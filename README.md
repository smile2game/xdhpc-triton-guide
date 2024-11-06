# 西电超算队的triton教程

- 为什么要有triton?
cuda编程难度大,学术界工业界都对GPU编程领域特定语言(DSL,Domain-Specific Language)感兴趣。

- triton的优势?
编写灵活且速度高于已有的 DSL 如 polyhedral machinery (Tiramisu/Tensor Comprehensions)、scheduling languages (Halide、TVM) 

- triton的核心理念:
基于分块的编程范式,能促进神经网络高性能计算kernel的构建

- triton与cuda的区别:
cuda是更细的线程粒度,triton是 分块的粒度。都是 “单程序，多数据”

- 硬件映射关系：
        计算单元（CUDA Cores）： GPU 的计算核心会并行执行这些程序，每个程序在硬件上被分配到一个或多个计算单元。

        寄存器（Registers）： 每个线程需要使用寄存器来存储临时变量，如 x、y、output。

        共享内存（Shared Memory）： 在这个示例中，未显式使用共享内存，但如果需要在程序内的线程间共享数据，可以利用共享
        内存。

        全局内存（Global Memory）： 输入和输出张量 x、y、output 存储在全局内存中，程序通过指针访问它们。

        指令并行性： GPU 利用 SIMD（单指令多数据）架构，同时对多个数据元素执行相同的指令。