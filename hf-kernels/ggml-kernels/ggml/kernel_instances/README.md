# GGML MMQ 内核拆分说明

## 概述

这次重构将原本在 `mmq.cu` 中的 `mul_mat_q_case` 函数拆分为独立的 CUDA 文件，每个量化类型对应一个文件。这样做的好处包括：

1. **并行编译**：每个量化类型可以独立编译，提高构建速度
2. **模块化**：代码结构更清晰，便于维护
3. **选择性链接**：可以选择性地链接需要的量化类型

## 文件结构

### 模板文件
- `kernel_instances/mmq_kernel_template.h` - 包含宏定义和模板实现
- `kernel_instances/mmq_kernels.h` - 函数声明头文件

### 量化类型实现文件
- `kernel_instances/mmq_q4_0.cu` - Q4_0 量化类型实现
- `kernel_instances/mmq_q4_1.cu` - Q4_1 量化类型实现  
- `kernel_instances/mmq_q5_0.cu` - Q5_0 量化类型实现
- `kernel_instances/mmq_q5_1.cu` - Q5_1 量化类型实现
- `kernel_instances/mmq_q8_0.cu` - Q8_0 量化类型实现
- `kernel_instances/mmq_q2_k.cu` - Q2_K 量化类型实现
- `kernel_instances/mmq_q3_k.cu` - Q3_K 量化类型实现
- `kernel_instances/mmq_q4_k.cu` - Q4_K 量化类型实现
- `kernel_instances/mmq_q5_k.cu` - Q5_K 量化类型实现
- `kernel_instances/mmq_q6_k.cu` - Q6_K 量化类型实现

## 主要更改

### 1. 主文件 `mmq.cu` 
- 添加了 `#include "kernel_instances/mmq_kernels.h"`
- 将 switch 语句中的调用改为新的函数名：
  ```cpp
  // 原来：
  mul_mat_q_case<scalar_t, GGML_TYPE_Q4_0>(kernel_args, stream);
  
  // 现在：
  mul_mat_q_case_Q4_0<scalar_t>(kernel_args, stream);
  ```

### 2. 头文件 `mmq.cuh`
- 移除了原始的 `mul_mat_q_case` 模板函数实现

### 3. 构建配置 `CMakeLists.txt`
- 在 `ggml_SRC` 中添加了所有新的 .cu 和 .h 文件

## 使用方式

拆分后的使用方式与原来完全相同，对外接口没有任何变化：

```cpp
torch::Tensor result = ggml_mul_mat_a8(weight, input, quantization_type, num_rows);
```

## 编译优势

1. **并行编译**：CUDA 编译器可以并行编译各个 .cu 文件
2. **增量编译**：修改某个量化类型的实现只需重新编译对应的文件
3. **编译时间**：大型项目中可显著减少总编译时间

## 扩展性

添加新的量化类型非常简单：

1. 创建新的 `.cu` 文件，例如 `mmq_qnew.cu`
2. 使用 `IMPLEMENT_MUL_MAT_Q_CASE(QNEW)` 宏
3. 在 `mmq_kernels.h` 中添加声明
4. 在主 `mmq.cu` 文件的 switch 语句中添加 case
5. 更新 `CMakeLists.txt`