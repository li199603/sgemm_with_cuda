#pragma once
#include <functional>

struct CostTime {
    float total = 0.0f, kernel = 0.0f; // millisecond
};

struct Performance {
    CostTime cost_time;
    float tflops = 0.0f; // 1 tflops = 10^12 flops
};

// 暴力三层循环
CostTime sgemm_cpu(float *A, float *B, float *C, const int M, const int N,
                   const int K);

// gpu并行去掉两层循环
CostTime sgemm_gpu_v1(float *A, float *B, float *C, const int M, const int N,
                      const int K);

// 使用共享内存来优化
CostTime sgemm_gpu_v2(float *A, float *B, float *C, const int M, const int N,
                      const int K);

// 使用寄存器来优化
CostTime sgemm_gpu_v3(float *A, float *B, float *C, const int M, const int N,
                      const int K);

// 优化bank冲突——s_a列优先
CostTime sgemm_gpu_v4(float *A, float *B, float *C, const int M, const int N,
                      const int K);

// 优化bank冲突——s_b读取位置重排
CostTime sgemm_gpu_v5(float *A, float *B, float *C, const int M, const int N,
                      const int K);

// 指令并行化——预取
CostTime sgemm_gpu_v6(float *A, float *B, float *C, const int M, const int N,
                      const int K);

// cublas  英伟达官方实现
CostTime sgemm_cublas(float *A, float *B, float *C, const int M, const int N,
                      const int K);

using SgemmFunc = std::function<CostTime(float *, float *, float *, const int,
                                         const int, const int)>;
float test_error(SgemmFunc func);
Performance test_performance(SgemmFunc func, const int M, const int N,
                             const int K, const int test_num);
