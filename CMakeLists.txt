cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(lenet5_mnist_mpi)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 ${TORCH_CXX_FLAGS}")

option(DOWNLOAD_LENET5_MNIST "Download the LENET5_MNIST dataset from the internet" ON)
if (DOWNLOAD_LENET5_MNIST)
  message(STATUS "Downloading LENET5_MNIST dataset")
  execute_process(
    COMMAND python ${CMAKE_CURRENT_LIST_DIR}/download_mnist.py
      -d ${CMAKE_BINARY_DIR}/data
    ERROR_VARIABLE DOWNLOAD_ERROR)
  if (DOWNLOAD_ERROR)
    message(FATAL_ERROR "Error downloading MNIST dataset: ${DOWNLOAD_ERROR}")
  endif()
endif()

add_executable(lenet5_mnist_mpi lenet5_mnist_mpi.cpp)
target_compile_features(lenet5_mnist_mpi PUBLIC cxx_range_for)
target_link_libraries(lenet5_mnist_mpi ${TORCH_LIBRARIES})
set_property(TARGET lenet5_mnist_mpi PROPERTY CXX_STANDARD 14)
