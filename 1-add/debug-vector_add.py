import torch
import triton
import triton.language as tl

@triton.jit  # Just-In-Time 编译器
def add_kernel(
    x_ptr,        # 第一个输入向量的指针
    y_ptr,        # 第二个输入向量的指针
    output_ptr,   # 输出向量的指针
    debug_ptr,    # 调试缓冲区的指针，用于存储 pid
    n_elements,   # 向量的大小
    BLOCK_SIZE: tl.constexpr,  # 每个程序处理的元素数量
):
    pid = tl.program_id(axis=0)  # 获取当前程序的ID（块编号）
    block_start = pid * BLOCK_SIZE  # 计算当前块的起始索引
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # 计算当前块内的索引偏移
    mask = offsets < n_elements  # 创建掩码，防止越界访问

    # 从全局内存加载x和y的对应元素
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    output = x + y  # 执行向量加法

    # 将结果存储回全局内存
    tl.store(output_ptr + offsets, output, mask=mask)

    # 将 pid 存储到调试缓冲区中
    # 通过将 pid 转换为 float32
    pid_f32 = pid + 0.0  # 将 pid 转换为 float32
    tl.store(debug_ptr + pid, pid_f32)  # 存储到调试缓冲区

if __name__ == '__main__':
    # 初始化输入向量
    n = 1024  # 向量大小
    x = torch.randn(n, device='cuda')  # 随机生成向量x
    y = torch.randn(n, device='cuda')  # 随机生成向量y
    output = torch.empty_like(x)  # 分配输出向量空间

    BLOCK_SIZE = 256  # 每个程序处理的元素数量
    grid = (triton.cdiv(n, BLOCK_SIZE),)  # 计算网格大小

    # 分配调试缓冲区，大小等于网格大小，每个元素为 float32
    debug = torch.empty(grid, dtype=torch.float32, device='cuda')

    # 调用 Triton 核函数
    add_kernel[grid](
        x, y, output,
        debug,  # 传递调试缓冲区
        n,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # 等待核函数执行完成
    torch.cuda.synchronize()

    # 检查CUDA错误
    error = torch.cuda.get_last_error()
    if error != torch.cuda.Error.Success:
        print(f"CUDA 错误: {error}")
    else:
        # 读取并打印调试信息
        debug_cpu = debug.cpu().tolist()
        print(f"Program IDs (pid) values: {debug_cpu}")

        # 验证结果
        expected = x + y  # 使用PyTorch自带的向量加法作为参考
        if torch.allclose(output, expected):
            print("向量加法成功！")
        else:
            print("向量加法结果不正确。")
