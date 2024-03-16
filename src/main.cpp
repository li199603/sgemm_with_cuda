#include "sgemm.cuh"
#include <cstdio>
#include <map>
#include <string>

int main(int argc, char **argv) {
    std::map<std::string, SgemmFunc> func_map = {
        {"sgemm_gpu_v1", sgemm_gpu_v1}, {"sgemm_gpu_v2", sgemm_gpu_v2},
        {"sgemm_gpu_v3", sgemm_gpu_v3}, {"sgemm_gpu_v4", sgemm_gpu_v4},
        {"sgemm_gpu_v5", sgemm_gpu_v5}, {"sgemm_gpu_v6", sgemm_gpu_v6},
        {"sgemm_cublas", sgemm_cublas},
    };
    // test_error
    for (auto &kv : func_map) {
        float max_error = test_error(kv.second);
        printf("%12s: max_error=%8.6f\n", kv.first.c_str(), max_error);
    }
    printf("\n");
    // test_performance
    int M_list[] = {1024, 8192, 12288};
    int N_list[] = {1024, 8192, 12288};
    int K_list[] = {1024, 1024, 1024};
    int length = sizeof(M_list) / sizeof(M_list[0]);
    int test_num = 10;

    for (int i = 0; i < length; i++) {
        int M = M_list[i], N = N_list[i], K = K_list[i];
        printf("--------------------------------- ");
        printf("M=%5d, N=%5d, K=%5d", M, N, K);
        printf(" ---------------------------------\n");
        for (auto &kv : func_map) {
            Performance performance =
                test_performance(kv.second, M, N, K, test_num);
            printf(
                "%12s: total_time=%8.3fms, kernel_time=%8.3fms, tflops=%6.3f\n",
                kv.first.c_str(), performance.cost_time.total,
                performance.cost_time.kernel, performance.tflops);
        }
        printf("\n");
    }

    return 0;
}