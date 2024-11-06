import torch
import triton
import triton.language as tl

@triton.jit  # 对核函数进行即时编译
def add_kernel(x_ptr,        # 指向第一个输入向量的指针（大小：n_elements）
               y_ptr,        # 指向第二个输入向量的指针（大小：n_elements）
               output_ptr,   # 指向输出向量的指针（大小：n_elements）
               n_elements,   # 向量中的元素总数
               BLOCK_SIZE: tl.constexpr,  # 每个程序（块）处理的元素数量
               ):
    # 每个“程序”（块）处理数据的不同部分。
    # 通过获取程序ID来确定我们是哪一个程序。
    pid = tl.program_id(axis=0)  # 获取第0轴上的程序ID（我们使用1D网格）>> 4个program
    # 计算该程序将处理的块的起始索引。
    block_start = pid * BLOCK_SIZE  # 该块的起始索引
    # 生成该块将处理的元素的偏移。
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # 偏移：从block_start到block_start+BLOCK_SIZE-1
    # 例如，对于pid=1和BLOCK_SIZE=256：
    # block_start = 1 * 256 = 256
    # offsets = 256 + [0, 1, ..., 255] = [256, 257, ..., 511]
    
    # 创建一个掩码，确保在n_elements不是BLOCK_SIZE的倍数时，不会越界访问。
    mask = offsets < n_elements  # 布尔掩码，标记offsets中有效的索引
    # 在计算的偏移位置，从x和y中加载数据，应用掩码。
    x = tl.load(x_ptr + offsets, mask=mask)  # 从x加载元素；形状：(BLOCK_SIZE,)
    y = tl.load(y_ptr + offsets, mask=mask)  # 从y加载元素；形状：(BLOCK_SIZE,)
    # 执行元素级加法。
    output = x + y  # 结果向量；形状：(BLOCK_SIZE,),triton内部处理
    # 将结果存回输出向量，应用掩码。
    tl.store(output_ptr + offsets, output, mask=mask)

if __name__ == '__main__':
    # 使用随机值初始化输入向量。
    n = 1024  # 向量x和y中的元素总数
    x = torch.randn(n, device='cuda')  # x：形状为(1024,)的张量，存储在GPU上，元素为随机浮点数
    y = torch.randn(n, device='cuda')  # y：形状为(1024,)的张量，存储在GPU上，元素为随机浮点数
    output = torch.empty_like(x)  # 在GPU上分配形状为(1024,)的输出张量

    BLOCK_SIZE = 256  # 每个程序（块）处理的元素数量
    # 计算需要的程序（块）数量；网格大小为：(4,)
    # triton.cdiv(a, b)计算ceil(a / b)
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    # 对于n=1024和BLOCK_SIZE=256：
    # grid = (triton.cdiv(1024, 256),) = (4,)

    # 使用指定的网格大小启动Triton核函数。
    add_kernel[grid](
        x,          # 指向x的指针（大小：1024）
        y,          # 指向y的指针（大小：1024）
        output,     # 指向输出的指针（大小：1024）
        n,          # 元素总数
        BLOCK_SIZE=BLOCK_SIZE  # 块大小（作为constexpr传递）
    )

    # 通过与PyTorch的加法结果比较来验证结果。
    expected = x + y  # 使用PyTorch计算期望结果
    if torch.allclose(output, expected):
        print("向量加法成功！")
    else:
        print("向量加法结果不正确。")
