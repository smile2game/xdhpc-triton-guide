import torch

import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr,  # 第一个输入向量的指针。
               y_ptr,  # 第二个输入向量的指针。
               output_ptr,  # 输出向量的指针。
               n_elements,  # 向量的大小。
               BLOCK_SIZE: tl.constexpr,  # 每个程序应该处理的元素数量。
               # 注意：`constexpr` 可以作为形状值使用。
               ):
    # 有多个'程序'处理不同的数据。我们在这里标识我们是哪个程序：
    pid = tl.program_id(axis=0)  # 我们使用 1D launch 网格，因此 axis 是 0。
    # 该程序将处理与初始数据偏移的输入。
    # 例如，如果您有长度为 256 的向量和块大小为 64，程序
    # 将分别访问元素[0:64, 64:128, 128:192, 192:256]。
    # 请注意，偏移量是指针的列表：
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # 创建一个掩码以防止内存操作超出范围。
    mask = offsets < n_elements
    # 从 DRAM 加载 x 和 y，以掩盖掉输入不是块大小的倍数的任何额外元素。
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # 将 x + y 写回到 DRAM。
    tl.store(output_ptr + offsets, output, mask=mask)

if __name__ == '__main__':
    # 初始化输入向量
    n = 1024  # 向量大小
    x = torch.randn(n, device='cuda')
    y = torch.randn(n, device='cuda')
    output = torch.empty_like(x)  # 分配输出向量空间

    BLOCK_SIZE = 256  # 每个程序处理的元素数量
    grid = (triton.cdiv(n, BLOCK_SIZE),)  # 计算网格大小

    # 调用 Triton 核函数
    add_kernel[grid](
        x, y, output, 
        n, 
        BLOCK_SIZE=BLOCK_SIZE
    )

    # 验证结果
    expected = x + y
    if torch.allclose(output, expected):
        print("向量加法成功！")
    else:
        print("向量加法结果不正确。")
