# APAM: Asynchronous Parallel Adaptive stochastic gradient Method

This repository contains a C++-MPI implementation of APAM [(Xu et al. 2020)](#Xu2020). It supports MPI/OpenMP hybrid parallelism.

For the C++-OpenMP code, see [https://github.com/xu-yangyang/APAM](https://github.com/xu-yangyang/APAM).

## Prerequisites

- CMake, version >= 3.1
- C++ compiler with C++14 standard support
- (optional) C++ compiler with OpenMP support

The configuration by default assumes using OpenMP. If you do not want to, or cannot, use OpenMP, change the following line in `CMakeLists.txt` from

```
option(USE_OPENMP "If has OpenMP and use OpenMP" ON)
```

to

```
option(USE_OPENMP "If has OpenMP and use OpenMP" OFF)
```

Hint: The purpose of using OpenMP is to allow multithreading for the master. If you use only one thread for the master, not invoking OpenMP will save you from threading overhead.

## Usage

Install MPI as needed. Example on Ubuntu Linux with OpenMPI:

```sh
sudo apt-get install openmpi-bin libopenmpi-dev
```

Install libtorch as needed. Go to [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) and get the correct link to download. Example on Linux without GPU support:

```sh
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.4.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.4.0+cpu.zip
```

Under the code directory, generate makefile. The `/absolute/path/to/libtorch` below is where you unzip in the last step, concatenated with the folder name.

```sh
mkdir build
cd build
cmake -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
```

Compile the code.  Example with `apam_mpi_lenet5_mnist`:

```sh
make apam_mpi_lenet5_mnist
```

or, if you want to compile all executables:

```sh
make
```

Run the code. Example running `apam_mpi_lenet5_mnist`, by using 4 mpi processes (1 master and 3 workers) with an initial learning rate 1e-4 and 40 epochs:

```sh
OMP_NUM_THREADS=1 mpirun -np 4 apam_mpi_lenet5_mnist --lr 1e-4 --num_epochs 40
```

The purpose of setting the environment variable `OMP_NUM_THREADS=1` is to ensure that any libtorch related function call uses one thread. Besides libtorch calls, the master and each worker may use several threads to perform other calculations. In practice, each worker uses one thread, but the master may need to use several threads to empower itself for faster gradient digestion and communication. The number of threads for the master may be set through command line option `--master_num_threads`.

For all available command line options, see the beginning of `apam_mpi_main.cpp`. They are self-explanatory. The command line option syntax follows the POSIX standard.

Note: If the compiler does not have OpenMP support, setting `--master_num_threads` takes no effect.

## Performance

Performance of `apam_mpi_lenet5_mnist` on a few machines is listed in the following. The option list is `--lr 1e-4 --num_epochs 40`.

On Ubuntu Linux 5.3.0-28.30~18.04.1, Intel(R) Xeon(R) CPU X5677 @ 3.47GHz, 8 cores, no GPU, gcc 7.5.0 with OpenMP, OpenMPI 2.1.1:

| # Workers | Test loss | Test accuracy | Train time (sec) | Speedup |
| :-------: | :-------: | :-----------: | :--------------: | :-----: |
| 1         | 0.0336    | 98.96%        | 1114.4           | 1.00    |
| 2         | 0.0339    | 98.93%        | 567.16           | 1.96    |
| 3         | 0.0355    | 98.88%        | 434.35           | 2.56    |
| 4         | 0.0346    | 98.98%        | 359.02           | 3.10    |
| 5         | 0.0351    | 99.02%        | 319.17           | 3.49    |
| 6         | 0.0367    | 98.90%        | 289.73           | 3.84    |

On Ubuntu Linux 4.4.0-169, Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz, 64 cores, no GPU, gcc 5.4.0 with OpenMP, MPICH 3.2:

| # Workers | Test loss | Test accuracy | Train time (sec) | Speedup |
| :-------: | :-------: | :-----------: | :--------------: | :-----: |
| 1         | 0.0327    | 99.01%        | 460.58           | 1.00    |
| 2         | 0.0389    | 98.86%        | 228.72           | 2.01    |
| 4         | 0.0359    | 98.95%        | 121.85           | 3.77    |
| 8         | 0.0343    | 98.99%        | 60.894           | 7.56    |
| 16        | 0.0365    | 98.86%        | 32.764           | 14.05   |
| 32        | 0.0468    | 98.50%        | 17.856           | 25.79   |

On macOS 10.15.1, 2.7 GHz Quad-Core Intel Core i7, no GPU, Apple clang 11.0.3 without OpenMP, OpenMPI 4.0.4:

| # Workers | Test loss | Test accuracy | Train time (sec) | Speedup |
| :-------: | :-------: | :-----------: | :--------------: | :-----: |
| 1         | 0.0329    | 99.03%        | 395.60           | 1.00    |
| 2         | 0.0365    | 98.89%        | 211.69           | 1.86    |
| 3         | 0.0364    | 98.83%        | 157.65           | 2.50    |

<!--- More results here

Performance of `apam_mpi_mlp_mnist ` on a few machines is listed in the following. The option `--master_num_threads` is set as the number of processes minus 1.

On Ubuntu Linux 5.3.0-28.30~18.04.1, Intel(R) Xeon(R) CPU X5677 @ 3.47GHz, 8 cores, no GPU, gcc 7.5.0 with OpenMP, OpenMPI 2.1.1, `--master_num_threads` set as np-1:

| # Workers | Test loss | Test accuracy | Train time (sec) | Speedup |
| :-------: | :-------: | :-----------: | :--------------: | :-----: |
| 1         | 0.0650    | 98.21%        | 192.37           | 1.00    |
| 2         | 0.0629    | 98.26%        | 91.125           | 2.11    |
| 3         | 0.0613    | 98.38%        | 72.611           | 2.64    |

On Ubuntu Linux 4.4.0-169, Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz, 64 cores, no GPU, gcc 5.4.0 with OpenMP, MPICH 3.2, `--master_num_threads` set as min(np-1,8):

| # Workers | Test loss | Test accuracy | Train time (sec) | Speedup |
| :-------: | :-------: | :-----------: | :--------------: | :-----: |
| 1         | 0.0655    | 98.31%        | 97.739           | 1.00    |
| 2         | 0.0639    | 98.40%        | 49.955           | 1.95    |
| 4         | 0.0619    | 98.41%        | 23.543           | 4.15    |
| 8         | 0.0681    | 98.16%        | 12.639           | 7.73    |
| 16        | 0.0780    | 97.63%        | 16.604           | 5.88    |
| 32        | 0.1325    | 95.87%        | 25.474           | 3.83    |

On macOS 10.15.1, 2.7 GHz Quad-Core Intel Core i7, no GPU, Apple clang 11.0.3 without OpenMP, OpenMPI 4.0.4, `--master_num_threads` set as 1:

| # Workers | Test loss | Test accuracy | Train time (sec) | Speedup |
| :-------: | :-------: | :-----------: | :--------------: | :-----: |
| 1         | 0.0644    | 98.26%        | 86.399           | 1.00    |
| 2         | 0.0617    | 98.43%        | 45.326           | 1.90    |
| 3         | 0.0618    | 98.37%        | 37.985           | 2.27    |

Performance of `apam_mpi_logit_mnist ` on a few machines is listed in the following. The option list is `--lr 1e-4`.

On Ubuntu Linux 5.3.0-28.30~18.04.1, Intel(R) Xeon(R) CPU X5677 @ 3.47GHz, 8 cores, no GPU, gcc 7.5.0 with OpenMP, OpenMPI 2.1.1:

| # Workers | Test loss | Test accuracy | Train time (sec) | Speedup |
| :-------: | :-------: | :-----------: | :--------------: | :-----: |
| 1         | 0.2769    | 92.15%        | 44.669           | 1.00    |
| 2         | 0.2770    | 92.26%        | 22.508           | 1.98    |
| 3         | 0.2787    | 92.08%        | 17.768           | 2.51    |
| 4         | 0.2776    | 92.13%        | 15.154           | 2.94    |
| 5         | 0.2776    | 92.22%        | 13.047           | 3.42    |
| 6         | 0.2784    | 92.12%        | 11.810           | 3.78    |

On Ubuntu Linux 4.4.0-169, Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz, 64 cores, no GPU, gcc 5.4.0 with OpenMP, MPICH 3.2:

| # Workers | Test loss | Test accuracy | Train time (sec) | Speedup |
| :-------: | :-------: | :-----------: | :--------------: | :-----: |
| 1         | 0.2769    | 92.15%        | 45.956           | 1.00    |
| 2         | 0.2771    | 92.17%        | 23.699           | 1.93    |
| 4         | 0.2779    | 92.17%        | 11.114           | 4.13    |
| 8         | 0.2777    | 92.10%        | 5.5981           | 8.20    |
| 16        | 0.2794    | 92.23%        | 3.0756           | 14.94   |
| 32        | 0.2843    | 92.08%        | 1.9339           | 23.76   |

On macOS 10.15.1, 2.7 GHz Quad-Core Intel Core i7, no GPU, Apple clang 11.0.3 without OpenMP, OpenMPI 4.0.4:

| # Workers | Test loss | Test accuracy | Train time (sec) | Speedup |
| :-------: | :-------: | :-----------: | :--------------: | :-----: |
| 1         | 0.2769    | 92.15%        | 26.983           | 1.00    |
| 2         | 0.2775    | 92.22%        | 13.800           | 1.95    |
| 3         | 0.2788    | 92.13%        | 9.6875           | 2.78    |

-->

## Reference

- <a name="Xu2020"></a>Yangyang Xu, Colin Sutcher-Shepard, Yibo Xu, and Jie Chen. [Asynchronous parallel adaptive stochastic gradient methods](https://arxiv.org/abs/2002.09095). Preprint arXiv:2002.09095, 2020.

