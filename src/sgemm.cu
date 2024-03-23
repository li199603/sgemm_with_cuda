/*
    A -- [M, K]
    B -- [K, N]
    C -- [M, N] = A * B
*/

#include "sgemm.cuh"
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cublas_v2.h>
#include <random>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(float_var) (reinterpret_cast<float4 *>(&(float_var))[0])
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            printf("CUDA Error: \n");                                          \
            printf("    File:       %s\n", __FILE__);                          \
            printf("    Line:       %d\n", __LINE__);                          \
            printf("    Error Code: %d\n", err);                               \
            printf("    Error Text: %s\n", cudaGetErrorString(err));           \
            exit(1);                                                           \
        }                                                                      \
    } while (0)
#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t err = call;                                             \
        if (err != CUBLAS_STATUS_SUCCESS) {                                    \
            printf("cuBLAS Error: \n");                                        \
            printf("    File:       %s\n", __FILE__);                          \
            printf("    Line:       %d\n", __LINE__);                          \
            printf("    Error Code: %d\n", err);                               \
            printf("    Error Text: %s\n", cublasGetStatusString(err));        \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

void data_init(float *data, const int num) {
    std::uniform_real_distribution<float> float_gen(-1.0f, 1.0f);
    std::default_random_engine rand_engine(time(nullptr));
    for (int i = 0; i < num; i++) {
        data[i] = float_gen(rand_engine);
    }
}

class TotalTimer {
    using Clock = std::chrono::high_resolution_clock;

  private:
    Clock::time_point m_start_point, m_end_point;

  public:
    void start() { m_start_point = Clock::now(); };
    void end() { m_end_point = Clock::now(); };
    float cost() {
        std::chrono::duration<float, std::milli> dur =
            m_end_point - m_start_point;
        return dur.count();
    };
};

class KernelTimer {
  private:
    cudaEvent_t m_start_event, m_end_event;

  public:
    KernelTimer() {
        CUDA_CHECK(cudaEventCreate(&m_start_event));
        CUDA_CHECK(cudaEventCreate(&m_end_event));
    };
    ~KernelTimer() {
        CUDA_CHECK(cudaEventDestroy(m_start_event));
        CUDA_CHECK(cudaEventDestroy(m_end_event));
    };
    void start() { CUDA_CHECK(cudaEventRecord(m_start_event)); };
    void end() {
        CUDA_CHECK(cudaEventRecord(m_end_event));
        CUDA_CHECK(cudaEventSynchronize(m_end_event));
    };
    float cost() {
        float kernel_cost;
        CUDA_CHECK(
            cudaEventElapsedTime(&kernel_cost, m_start_event, m_end_event));
        return kernel_cost;
    };
};

float test_error(SgemmFunc func) {
    const int M = 512, N = 1024, K = 128;

    float *A, *B, *C1, *C2;
    const size_t size_A = M * K * sizeof(float);
    const size_t size_B = K * N * sizeof(float);
    const size_t size_C = M * N * sizeof(float);
    A = (float *)malloc(size_A);
    B = (float *)malloc(size_B);
    C1 = (float *)malloc(size_C);
    C2 = (float *)malloc(size_C);

    data_init(A, M * K);
    data_init(B, K * N);

    sgemm_cpu(A, B, C1, M, N, K);
    func(A, B, C2, M, N, K);

    float max_error = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float this_error = std::abs(C1[i] - C2[i]);
        max_error = std::max(max_error, this_error);
    }

    free(A);
    free(B);
    free(C1);
    free(C2);

    return max_error;
}

Performance test_performance(SgemmFunc func, const int M, const int N,
                             const int K, const int test_num) {
    float *A, *B, *C;
    const size_t size_A = M * K * sizeof(float);
    const size_t size_B = K * N * sizeof(float);
    const size_t size_C = M * N * sizeof(float);
    A = (float *)malloc(size_A);
    B = (float *)malloc(size_B);
    C = (float *)malloc(size_C);
    data_init(A, M * K);
    data_init(B, K * N);

    CostTime avg_cost_time;
    for (int i = 0; i < test_num; i++) {
        CostTime cost_time = func(A, B, C, M, N, K);
        avg_cost_time.total += cost_time.total;
        avg_cost_time.kernel += cost_time.kernel;
    }
    avg_cost_time.total /= test_num;
    avg_cost_time.kernel /= test_num;
    float flops = 2.0f * M * N * K / (avg_cost_time.kernel / 1e3);

    Performance performance;
    performance.cost_time = avg_cost_time;
    performance.tflops = flops / 1e12;

    free(A);
    free(B);
    free(C);

    return performance;
}

CostTime sgemm_cpu(float *A, float *B, float *C, const int M, const int N,
                   const int K) {
    CostTime cost_time;
    TotalTimer total_timer;
    total_timer.start();

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float value = 0.0f;
            for (int k = 0; k < K; k++) {
                value += A[OFFSET(m, k, K)] * B[OFFSET(k, n, N)];
            }
            C[OFFSET(m, n, N)] = value;
        }
    }

    total_timer.end();
    cost_time.total = total_timer.cost();
    cost_time.kernel = cost_time.total;
    return cost_time;
}

__global__ void sgemm_gpu_kernel_v1(float *__restrict__ A,
                                    float *__restrict__ B,
                                    float *__restrict__ C, const int M,
                                    const int N, const int K) {
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    float value = 0.0f;
    for (int k = 0; k < K; k++) {
        value += A[OFFSET(m, k, K)] * B[OFFSET(k, n, N)];
    }
    C[OFFSET(m, n, N)] = value;
}

CostTime sgemm_gpu_v1(float *A, float *B, float *C, const int M, const int N,
                      const int K) {
    CostTime cost_time;
    TotalTimer total_timer;
    total_timer.start();

    const int BM = 16, BN = 16; // 受线程块最大线程数限制

    assert(M % BM == 0 && N % BN == 0); // 核函数不处理边界情况
    const dim3 block_size(BN, BM);
    const dim3 grid_size(N / BN, M / BM);

    float *d_A, *d_B, *d_C;
    const size_t size_A = M * K * sizeof(float);
    const size_t size_B = K * N * sizeof(float);
    const size_t size_C = M * N * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));

    KernelTimer kernel_timer;
    kernel_timer.start();

    sgemm_gpu_kernel_v1<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    kernel_timer.end();
    cost_time.kernel = kernel_timer.cost();

    CUDA_CHECK(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    total_timer.end();
    cost_time.total = total_timer.cost();

    return cost_time;
}

__global__ void sgemm_gpu_kernel_v2(float *__restrict__ A,
                                    float *__restrict__ B,
                                    float *__restrict__ C, const int M,
                                    const int N, const int K) {
    const int BM = 16, BN = 16;
    const int BK = 64;
    __shared__ float s_a[BM][BK], s_b[BK][BN];
    float c = 0.0f;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    // 每次从全局内存加载到共享内存，每个线程都负责一个float4。以下是当前线程负责的这个float4的索引
    const int row_s_a = tid / 16;
    const int col_s_a = (tid % 16) * 4;
    const int row_s_b = tid / 4;
    const int col_s_b = (tid % 4) * 4;
    // 每个线程从读取的全局内存的位置，在A上的行是固定不变的，在B上列是固定不变的
    const int row_A = blockIdx.y * BM + row_s_a;
    const int col_B = blockIdx.x * BN + col_s_b;

    for (int step = 0; step < K / BK; step++) {
        // 从A加载到s_a
        const int col_A = step * BK + col_s_a;
        const int index_A = OFFSET(row_A, col_A, K);
        FETCH_FLOAT4(s_a[row_s_a][col_s_a]) = FETCH_FLOAT4(A[index_A]);
        // 从B加载到s_b
        const int row_B = step * BK + row_s_b;
        const int index_B = OFFSET(row_B, col_B, N);
        FETCH_FLOAT4(s_b[row_s_b][col_s_b]) = FETCH_FLOAT4(B[index_B]);
        __syncthreads();
        // 计算
        for (int k = 0; k < BK; k++) {
            const float a = s_a[threadIdx.y][k];
            const float b = s_b[k][threadIdx.x];
            c += a * b;
        }
        __syncthreads();
    }
    // 写入C
    const int row_C = blockIdx.y * BM + threadIdx.y;
    const int col_C = blockIdx.x * BN + threadIdx.x;
    const int index_C = OFFSET(row_C, col_C, N);
    C[index_C] = c;
}

CostTime sgemm_gpu_v2(float *A, float *B, float *C, const int M, const int N,
                      const int K) {
    CostTime cost_time;
    TotalTimer total_timer;
    total_timer.start();

    const int BM = 16, BN = 16; // 受线程块最大线程数限制
    // 理论上其大小不影响计算速度。为了每个线程刚好加载一个float4
    const int BK = 64;

    assert(M % BM == 0 && N % BN == 0 && K % BK == 0); // 核函数不处理边界情况
    const dim3 block_size(BN, BM);
    const dim3 grid_size(N / BN, M / BM);

    float *d_A, *d_B, *d_C;
    const size_t size_A = M * K * sizeof(float);
    const size_t size_B = K * N * sizeof(float);
    const size_t size_C = M * N * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));

    KernelTimer kernel_timer;
    kernel_timer.start();

    sgemm_gpu_kernel_v2<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    kernel_timer.end();
    cost_time.kernel = kernel_timer.cost();

    CUDA_CHECK(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    total_timer.end();
    cost_time.total = total_timer.cost();

    return cost_time;
}

__global__ void sgemm_gpu_kernel_v3(float *__restrict__ A,
                                    float *__restrict__ B,
                                    float *__restrict__ C, const int M,
                                    const int N, const int K) {
    const int TM = 8, TN = 8;
    const int BM = 128, BN = 128;
    const int BK = 8;

    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];
    float r_a[TM];
    float r_b[TN];
    float r_c[TM][TN] = {0.0f};

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    // 每次从全局内存加载到共享内存，每个线程都负责一个float4。以下是当前线程负责的这个float4的索引
    const int row_s_a = tid / 2;
    const int col_s_a = (tid % 2) * 4;
    const int row_s_b = tid / 32;
    const int col_s_b = (tid % 32) * 4;
    // 每个线程从读取的全局内存的位置，在A上的行是固定不变的，在B上列是固定不变的
    const int row_A = blockIdx.y * BM + row_s_a;
    const int col_B = blockIdx.x * BN + col_s_b;

    for (int step = 0; step < K / BK; step++) {
        // 从A加载到s_a
        const int col_A = step * BK + col_s_a;
        const int index_A = OFFSET(row_A, col_A, K);
        FETCH_FLOAT4(s_a[row_s_a][col_s_a]) = FETCH_FLOAT4(A[index_A]);
        // 从B加载到s_b
        const int row_B = step * BK + row_s_b;
        const int index_B = OFFSET(row_B, col_B, N);
        FETCH_FLOAT4(s_b[row_s_b][col_s_b]) = FETCH_FLOAT4(B[index_B]);
        __syncthreads();
        for (int k = 0; k < BK; k++) {
            // 从s_a加载到r_a
            const int row_start = threadIdx.y * TM;
            for (int i = 0; i < TM; i++) {
                r_a[i] = s_a[row_start + i][k];
            }
            // 从s_b加载到r_b
            const int col_start = threadIdx.x * TN;
            for (int i = 0; i < TN; i++) {
                r_b[i] = s_b[k][col_start + i];
            }
            // 计算
            for (int m = 0; m < TM; m++) {
                for (int n = 0; n < TN; n++) {
                    r_c[m][n] += r_a[m] * r_b[n];
                }
            }
        }
        __syncthreads();
    }

    // 写入C
    for (int m = 0; m < TM; m++) {
        for (int n = 0; n < TN; n += 4) {
            const int row = blockIdx.y * BM + threadIdx.y * TM + m;
            const int col = blockIdx.x * BN + threadIdx.x * TN + n;
            const int index_C = OFFSET(row, col, N);
            FETCH_FLOAT4(C[index_C]) = FETCH_FLOAT4(r_c[m][n]);
        }
    }
}

CostTime sgemm_gpu_v3(float *A, float *B, float *C, const int M, const int N,
                      const int K) {
    CostTime cost_time;
    TotalTimer total_timer;
    total_timer.start();

    const int TM = 8, TN = 8;     // 受线程最大寄存器数限制
    const int BM = 128, BN = 128; // 受线程块最大线程数限制
    // 理论上其大小不影响计算速度。为了每个线程刚好加载一个float4
    const int BK = 8;

    assert(M % BM == 0 && N % BN == 0 && K % BK == 0); // 核函数不处理边界情况
    const dim3 block_size(BN / TN, BM / TM);
    const dim3 grid_size((N + BN - 1) / BN, (M + BM - 1) / BM);

    float *d_A, *d_B, *d_C;
    const size_t size_A = M * K * sizeof(float);
    const size_t size_B = K * N * sizeof(float);
    const size_t size_C = M * N * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));

    KernelTimer kernel_timer;
    kernel_timer.start();

    sgemm_gpu_kernel_v3<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    kernel_timer.end();
    cost_time.kernel = kernel_timer.cost();

    CUDA_CHECK(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    total_timer.end();
    cost_time.total = total_timer.cost();

    return cost_time;
}

__global__ void sgemm_gpu_kernel_v4(float *__restrict__ A,
                                    float *__restrict__ B,
                                    float *__restrict__ C, const int M,
                                    const int N, const int K) {
    const int TM = 8, TN = 8;
    const int BM = 128, BN = 128;
    const int BK = 8;

    __shared__ float s_a[BK][BM]; // 相比v3，s_a改为列优先
    __shared__ float s_b[BK][BN];
    float r_a[TM];
    float r_b[TN];
    float r_c[TM][TN] = {0.0f};

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    // 每次从全局内存加载到共享内存，每个线程都负责一个float4。以下是当前线程负责的这个float4的索引
    const int row_s_a = tid / 2;
    const int col_s_a = (tid % 2) * 4;
    const int row_s_b = tid / 32;
    const int col_s_b = (tid % 32) * 4;
    // 每个线程从读取的全局内存的位置，在A上的行是固定不变的，在B上列是固定不变的
    const int row_A = blockIdx.y * BM + row_s_a;
    const int col_B = blockIdx.x * BN + col_s_b;

    for (int step = 0; step < K / BK; step++) {
        // 从A加载到s_a
        const int col_A = step * BK + col_s_a;
        const int index_A = OFFSET(row_A, col_A, K);
        FETCH_FLOAT4(r_a[0]) = FETCH_FLOAT4(A[index_A]); // 借用r_a[0]中转
        s_a[col_s_a + 0][row_s_a] = r_a[0];
        s_a[col_s_a + 1][row_s_a] = r_a[1];
        s_a[col_s_a + 2][row_s_a] = r_a[2];
        s_a[col_s_a + 3][row_s_a] = r_a[3];
        // 从B加载到s_b
        const int row_B = step * BK + row_s_b;
        const int index_B = OFFSET(row_B, col_B, N);
        FETCH_FLOAT4(s_b[row_s_b][col_s_b]) = FETCH_FLOAT4(B[index_B]);
        __syncthreads();
        for (int k = 0; k < BK; k++) {
            // 从s_a加载到r_a
            const int row_start = threadIdx.y * TM;
            FETCH_FLOAT4(r_a[0]) = FETCH_FLOAT4(s_a[k][row_start]);
            FETCH_FLOAT4(r_a[4]) = FETCH_FLOAT4(s_a[k][row_start + 4]);
            // 从s_b加载到r_b
            const int col_start = threadIdx.x * TN;
            FETCH_FLOAT4(r_b[0]) = FETCH_FLOAT4(s_b[k][col_start]);
            FETCH_FLOAT4(r_b[4]) = FETCH_FLOAT4(s_b[k][col_start + 4]);
            // 计算
            for (int m = 0; m < TM; m++) {
                for (int n = 0; n < TN; n++) {
                    r_c[m][n] += r_a[m] * r_b[n];
                }
            }
        }
        __syncthreads();
    }

    // 写入C
    for (int m = 0; m < TM; m++) {
        for (int n = 0; n < TN; n += 4) {
            const int row = blockIdx.y * BM + threadIdx.y * TM + m;
            const int col = blockIdx.x * BN + threadIdx.x * TN + n;
            const int index_C = OFFSET(row, col, N);
            FETCH_FLOAT4(C[index_C]) = FETCH_FLOAT4(r_c[m][n]);
        }
    }
}

CostTime sgemm_gpu_v4(float *A, float *B, float *C, const int M, const int N,
                      const int K) {
    // 除核函数，其他相比v3没有改动
    CostTime cost_time;
    TotalTimer total_timer;
    total_timer.start();

    const int TM = 8, TN = 8;     // 受线程最大寄存器数限制
    const int BM = 128, BN = 128; // 受线程块最大线程数限制
    // 理论上其大小不影响计算速度。为了每个线程刚好加载一个float4
    const int BK = 8;

    assert(M % BM == 0 && N % BN == 0 && K % BK == 0); // 核函数不处理边界情况
    const dim3 block_size(BN / TN, BM / TM);
    const dim3 grid_size((N + BN - 1) / BN, (M + BM - 1) / BM);

    float *d_A, *d_B, *d_C;
    const size_t size_A = M * K * sizeof(float);
    const size_t size_B = K * N * sizeof(float);
    const size_t size_C = M * N * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));

    KernelTimer kernel_timer;
    kernel_timer.start();

    sgemm_gpu_kernel_v4<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    kernel_timer.end();
    cost_time.kernel = kernel_timer.cost();

    CUDA_CHECK(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    total_timer.end();
    cost_time.total = total_timer.cost();

    return cost_time;
}

__global__ void sgemm_gpu_kernel_v5(float *__restrict__ A,
                                    float *__restrict__ B,
                                    float *__restrict__ C, const int M,
                                    const int N, const int K) {
    const int TM = 8, TN = 8;
    const int BM = 128, BN = 128;
    const int BK = 8;

    __shared__ float s_a[BK][BM]; // 相比v3，s_a改为列优先
    __shared__ float s_b[BK][BN];
    float r_a[TM];
    float r_b[TN];
    float r_c[TM][TN] = {0.0f};

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    // 每次从全局内存加载到共享内存，每个线程都负责一个float4。以下是负责的这个float4的索引
    const int row_s_a = tid / 2;
    const int col_s_a = (tid % 2) * 4;
    const int row_s_b = tid / 32;
    const int col_s_b = (tid % 32) * 4;
    // 每个线程从读取的全局内存的位置，在A上的行是固定不变的，在B上列是固定不变的
    const int row_A = blockIdx.y * BM + row_s_a;
    const int col_B = blockIdx.x * BN + col_s_b;

    for (int step = 0; step < K / BK; step++) {
        // 从A加载到s_a
        const int col_A = step * BK + col_s_a;
        const int index_A = OFFSET(row_A, col_A, K);
        FETCH_FLOAT4(r_a[0]) = FETCH_FLOAT4(A[index_A]); // 借用r_a[0]中转
        s_a[col_s_a + 0][row_s_a] = r_a[0];
        s_a[col_s_a + 1][row_s_a] = r_a[1];
        s_a[col_s_a + 2][row_s_a] = r_a[2];
        s_a[col_s_a + 3][row_s_a] = r_a[3];
        // 从B加载到s_b
        const int row_B = step * BK + row_s_b;
        const int index_B = OFFSET(row_B, col_B, N);
        FETCH_FLOAT4(s_b[row_s_b][col_s_b]) = FETCH_FLOAT4(B[index_B]);
        __syncthreads();
        for (int k = 0; k < BK; k++) {
            // 从s_a加载到r_a
            const int row_start = threadIdx.y * TM;
            FETCH_FLOAT4(r_a[0]) = FETCH_FLOAT4(s_a[k][row_start]);
            FETCH_FLOAT4(r_a[4]) = FETCH_FLOAT4(s_a[k][row_start + 4]);
            // 从s_b加载到r_b，相比v4，这里读取的位置变了
            const int col_start = threadIdx.x * (TN / 2);
            FETCH_FLOAT4(r_b[0]) = FETCH_FLOAT4(s_b[k][col_start]);
            FETCH_FLOAT4(r_b[4]) = FETCH_FLOAT4(s_b[k][col_start + BN / 2]);
            // 计算
            for (int m = 0; m < TM; m++) {
                for (int n = 0; n < TN; n++) {
                    r_c[m][n] += r_a[m] * r_b[n];
                }
            }
        }
        __syncthreads();
    }

    // 写入C，相比v4，写入位置也变了，因为操作的数据位置变了
    for (int m = 0; m < TM; m++) {
        const int row = blockIdx.y * BM + threadIdx.y * TM + m;
        const int col1 = blockIdx.x * BN + threadIdx.x * (TN / 2);
        const int col2 = blockIdx.x * BN + threadIdx.x * (TN / 2) + BN / 2;
        const int index1_C = OFFSET(row, col1, N);
        const int index2_C = OFFSET(row, col2, N);
        FETCH_FLOAT4(C[index1_C]) = FETCH_FLOAT4(r_c[m][0]);
        FETCH_FLOAT4(C[index2_C]) = FETCH_FLOAT4(r_c[m][4]);
    }
}

CostTime sgemm_gpu_v5(float *A, float *B, float *C, const int M, const int N,
                      const int K) {
    // 除核函数，其他相比v3没有改动
    CostTime cost_time;
    TotalTimer total_timer;
    total_timer.start();

    const int TM = 8, TN = 8;     // 受线程最大寄存器数限制
    const int BM = 128, BN = 128; // 受线程块最大线程数限制
    // 理论上其大小不影响计算速度。为了每个线程刚好加载一个float4
    const int BK = 8;

    assert(M % BM == 0 && N % BN == 0 && K % BK == 0); // 核函数不处理边界情况
    const dim3 block_size(BN / TN, BM / TM);
    const dim3 grid_size((N + BN - 1) / BN, (M + BM - 1) / BM);

    float *d_A, *d_B, *d_C;
    const size_t size_A = M * K * sizeof(float);
    const size_t size_B = K * N * sizeof(float);
    const size_t size_C = M * N * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));

    KernelTimer kernel_timer;
    kernel_timer.start();

    sgemm_gpu_kernel_v5<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    kernel_timer.end();
    cost_time.kernel = kernel_timer.cost();

    CUDA_CHECK(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    total_timer.end();
    cost_time.total = total_timer.cost();

    return cost_time;
}

__global__ void sgemm_gpu_kernel_v6(float *__restrict__ A,
                                    float *__restrict__ B,
                                    float *__restrict__ C, const int M,
                                    const int N, const int K) {
    const int TM = 8, TN = 8;
    const int BM = 128, BN = 128;
    const int BK = 8;

    // 相比v5，s_a, s_b变为double buffer
    __shared__ float s_a[2][BK][BM];
    __shared__ float s_b[2][BK][BN];
    float r_a[TM];
    float r_b[TN];
    float r_c[TM][TN] = {0.0f};

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    // 每次从全局内存加载到共享内存，每个线程都负责一个float4。以下是负责的这个float4的索引
    const int row_s_a = tid / 2;
    const int col_s_a = (tid % 2) * 4;
    const int row_s_b = tid / 32;
    const int col_s_b = (tid % 32) * 4;
    // 每个线程从读取的全局内存的位置，在A上的行是固定不变的，在B上列是固定不变的
    const int row_A = blockIdx.y * BM + row_s_a;
    const int col_B = blockIdx.x * BN + col_s_b;

    // 第一次加载  全局 --> 共享
    // 从A加载到s_a
    const int step = 0;
    const int col_A = step * BK + col_s_a;
    const int index_A = OFFSET(row_A, col_A, K);
    FETCH_FLOAT4(r_a[0]) = FETCH_FLOAT4(A[index_A]); // 借用r_a[0]中转
    s_a[0][col_s_a + 0][row_s_a] = r_a[0];
    s_a[0][col_s_a + 1][row_s_a] = r_a[1];
    s_a[0][col_s_a + 2][row_s_a] = r_a[2];
    s_a[0][col_s_a + 3][row_s_a] = r_a[3];
    // 从B加载到s_b
    const int row_B = step * BK + row_s_b;
    const int index_B = OFFSET(row_B, col_B, N);
    FETCH_FLOAT4(s_b[0][row_s_b][col_s_b]) = FETCH_FLOAT4(B[index_B]);
    __syncthreads();

    for (int step = 1; step < K / BK; step++) {
        const int lbi = step % 2; // load_buffer_index
        // 加载下一次迭代需要的  全局 --> 共享
        // 从A加载到s_a
        const int col_A = step * BK + col_s_a;
        const int index_A = OFFSET(row_A, col_A, K);
        FETCH_FLOAT4(r_a[0]) = FETCH_FLOAT4(A[index_A]); // 借用r_a[0]中转
        s_a[lbi][col_s_a + 0][row_s_a] = r_a[0];
        s_a[lbi][col_s_a + 1][row_s_a] = r_a[1];
        s_a[lbi][col_s_a + 2][row_s_a] = r_a[2];
        s_a[lbi][col_s_a + 3][row_s_a] = r_a[3];
        // 从B加载到s_b
        const int row_B = step * BK + row_s_b;
        const int index_B = OFFSET(row_B, col_B, N);
        FETCH_FLOAT4(s_b[lbi][row_s_b][col_s_b]) = FETCH_FLOAT4(B[index_B]);
        // 相比v5，此处不再需要同步。因为加载的数据本轮迭代用不到
        // __syncthreads();

        // 使用上一次加载的做运算
        const int cbi = (step - 1) % 2; // compute_buffer_index
        for (int k = 0; k < BK; k++) {
            // 从s_a加载到r_a
            const int row_start = threadIdx.y * TM;
            FETCH_FLOAT4(r_a[0]) = FETCH_FLOAT4(s_a[cbi][k][row_start]);
            FETCH_FLOAT4(r_a[4]) = FETCH_FLOAT4(s_a[cbi][k][row_start + 4]);
            // 从s_b加载到r_b，相比v4，这里读取的位置变了
            const int col_start = threadIdx.x * (TN / 2);
            FETCH_FLOAT4(r_b[0]) = FETCH_FLOAT4(s_b[cbi][k][col_start]);
            FETCH_FLOAT4(r_b[4]) =
                FETCH_FLOAT4(s_b[cbi][k][col_start + BN / 2]);
            // 计算
            for (int m = 0; m < TM; m++) {
                for (int n = 0; n < TN; n++) {
                    r_c[m][n] += r_a[m] * r_b[n];
                }
            }
        }
        __syncthreads();
    }

    // 补充最后一次加载到共享内存的数据对应的计算
    const int cbi = (K / BK - 1) % 2; // compute_buffer_index
    for (int k = 0; k < BK; k++) {
        // 从s_a加载到r_a
        const int row_start = threadIdx.y * TM;
        FETCH_FLOAT4(r_a[0]) = FETCH_FLOAT4(s_a[cbi][k][row_start]);
        FETCH_FLOAT4(r_a[4]) = FETCH_FLOAT4(s_a[cbi][k][row_start + 4]);
        // 从s_b加载到r_b，相比v4，这里读取的位置变了
        const int col_start = threadIdx.x * (TN / 2);
        FETCH_FLOAT4(r_b[0]) = FETCH_FLOAT4(s_b[cbi][k][col_start]);
        FETCH_FLOAT4(r_b[4]) = FETCH_FLOAT4(s_b[cbi][k][col_start + BN / 2]);
        // 计算
        for (int m = 0; m < TM; m++) {
            for (int n = 0; n < TN; n++) {
                r_c[m][n] += r_a[m] * r_b[n];
            }
        }
    }

    // 写入C，相比v4，写入位置也变了，因为操作的数据位置变了
    for (int m = 0; m < TM; m++) {
        const int row = blockIdx.y * BM + threadIdx.y * TM + m;
        const int col1 = blockIdx.x * BN + threadIdx.x * (TN / 2);
        const int col2 = blockIdx.x * BN + threadIdx.x * (TN / 2) + BN / 2;
        const int index1_C = OFFSET(row, col1, N);
        const int index2_C = OFFSET(row, col2, N);
        FETCH_FLOAT4(C[index1_C]) = FETCH_FLOAT4(r_c[m][0]);
        FETCH_FLOAT4(C[index2_C]) = FETCH_FLOAT4(r_c[m][4]);
    }
}

CostTime sgemm_gpu_v6(float *A, float *B, float *C, const int M, const int N,
                      const int K) {
    // 除核函数，其他相比v3没有改动
    CostTime cost_time;
    TotalTimer total_timer;
    total_timer.start();

    const int TM = 8, TN = 8;     // 受线程最大寄存器数限制
    const int BM = 128, BN = 128; // 受线程块最大线程数限制
    // 理论上其大小不影响计算速度。为了每个线程刚好加载一个float4
    const int BK = 8;

    assert(M % BM == 0 && N % BN == 0 && K % BK == 0); // 核函数不处理边界情况
    const dim3 block_size(BN / TN, BM / TM);
    const dim3 grid_size((N + BN - 1) / BN, (M + BM - 1) / BM);

    float *d_A, *d_B, *d_C;
    const size_t size_A = M * K * sizeof(float);
    const size_t size_B = K * N * sizeof(float);
    const size_t size_C = M * N * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));

    KernelTimer kernel_timer;
    kernel_timer.start();

    sgemm_gpu_kernel_v6<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    kernel_timer.end();
    cost_time.kernel = kernel_timer.cost();

    CUDA_CHECK(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    total_timer.end();
    cost_time.total = total_timer.cost();

    return cost_time;
}

CostTime sgemm_cublas(float *A, float *B, float *C, const int M, const int N,
                      const int K) {
    CostTime cost_time;
    TotalTimer total_timer;
    total_timer.start();

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float *d_A, *d_B, *d_C;
    const size_t size_A = M * K * sizeof(float);
    const size_t size_B = K * N * sizeof(float);
    const size_t size_C = M * N * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));

    float cublas_alpha = 1.0;
    float cublas_beta = 0;

    KernelTimer kernel_timer;
    kernel_timer.start();

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &cublas_alpha, d_B,
                N, d_A, K, &cublas_beta, d_C, N);

    kernel_timer.end();
    cost_time.kernel = kernel_timer.cost();

    CUDA_CHECK(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUBLAS_CHECK(cublasDestroy(handle));

    total_timer.end();
    cost_time.total = total_timer.cost();

    return cost_time;
}