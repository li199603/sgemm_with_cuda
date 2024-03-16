cmake_minimum_required(VERSION 3.9)
project(sgmm_test)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
find_package(CUDA REQUIRED)

set(INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/include)

set(CPP_SRC
    ${PROJECT_SOURCE_DIR}/src/main.cpp)

set(CUDA_SRC
    ${PROJECT_SOURCE_DIR}/src/sgemm.cu)

cuda_add_library(sgemm_core STATIC ${INCLUDE_DIRS} ${CUDA_SRC})

include_directories(${INCLUDE_DIRS})
add_executable(sgemm ${CPP_SRC})
target_link_libraries(sgemm sgemm_core cublas)
