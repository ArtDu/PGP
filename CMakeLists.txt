cmake_minimum_required(VERSION 3.17)
project(main CUDA)

set(CMAKE_CUDA_STANDARD 11)

add_executable(main labs/lab3/main.cu)

set_target_properties(
        main
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CUDA_FLAGS "-Werror")
set(CMAKE_CUDA_FLAGS "cross-execution-space-call")
set(CMAKE_CUDA_FLAGS "-lm")