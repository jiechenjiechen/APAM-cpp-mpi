cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(apam_mpi)

find_package(Torch REQUIRED)

option(USE_OPENMP "If has OpenMP and use OpenMP" ON)

if(USE_OPENMP)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp ${TORCH_CXX_FLAGS}")
  add_definitions(-DUSE_OPENMP)
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 ${TORCH_CXX_FLAGS}")
endif()

option(DOWNLOAD_MNIST "Download the MNIST dataset from the internet" ON)
if (DOWNLOAD_MNIST)
  message(STATUS "Downloading MNIST dataset")
  execute_process(
    COMMAND python ${CMAKE_CURRENT_LIST_DIR}/download_mnist.py
      -d ${CMAKE_BINARY_DIR}/data
    ERROR_VARIABLE DOWNLOAD_ERROR)
  if (DOWNLOAD_ERROR)
    message(FATAL_ERROR "Error downloading MNIST dataset: ${DOWNLOAD_ERROR}")
  endif()
endif()

add_executable(apam_mpi_logit_mnist apam_mpi_main.cpp)
target_compile_features(apam_mpi_logit_mnist PUBLIC cxx_range_for)
target_compile_definitions(apam_mpi_logit_mnist PRIVATE ARCHITECTURE=LOGIT)
target_link_libraries(apam_mpi_logit_mnist -lc ${TORCH_LIBRARIES})
set_property(TARGET apam_mpi_logit_mnist PROPERTY CXX_STANDARD 14)

add_executable(apam_mpi_mlp_mnist apam_mpi_main.cpp)
target_compile_features(apam_mpi_mlp_mnist PUBLIC cxx_range_for)
target_compile_definitions(apam_mpi_mlp_mnist PRIVATE ARCHITECTURE=MLP)
target_link_libraries(apam_mpi_mlp_mnist -lc ${TORCH_LIBRARIES})
set_property(TARGET apam_mpi_mlp_mnist PROPERTY CXX_STANDARD 14)

add_executable(apam_mpi_lenet5_mnist apam_mpi_main.cpp)
target_compile_features(apam_mpi_lenet5_mnist PUBLIC cxx_range_for)
target_compile_definitions(apam_mpi_lenet5_mnist PRIVATE ARCHITECTURE=LeNet5)
target_link_libraries(apam_mpi_lenet5_mnist -lc ${TORCH_LIBRARIES})
set_property(TARGET apam_mpi_lenet5_mnist PROPERTY CXX_STANDARD 14)
