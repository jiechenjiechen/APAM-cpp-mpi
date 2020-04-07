cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(apam_mpi)

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

add_executable(apam_mpi_logit_mnist apam_mpi_logit_mnist.cpp)
target_compile_features(apam_mpi_logit_mnist PUBLIC cxx_range_for)
target_link_libraries(apam_mpi_logit_mnist ${TORCH_LIBRARIES})
set_property(TARGET apam_mpi_logit_mnist PROPERTY CXX_STANDARD 14)

add_executable(apam_mpi_mlp_mnist apam_mpi_mlp_mnist.cpp)
target_compile_features(apam_mpi_mlp_mnist PUBLIC cxx_range_for)
target_link_libraries(apam_mpi_mlp_mnist ${TORCH_LIBRARIES})
set_property(TARGET apam_mpi_mlp_mnist PROPERTY CXX_STANDARD 14)

add_executable(apam_mpi_lenet5_mnist apam_mpi_lenet5_mnist.cpp)
target_compile_features(apam_mpi_lenet5_mnist PUBLIC cxx_range_for)
target_link_libraries(apam_mpi_lenet5_mnist ${TORCH_LIBRARIES})
set_property(TARGET apam_mpi_lenet5_mnist PROPERTY CXX_STANDARD 14)
