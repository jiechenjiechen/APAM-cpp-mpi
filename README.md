# APAM: Asynchronous Parallel Adaptive stochastic gradient Method

This repository contains a C++-MPI implementation of APAM [(Xu et al. 2020)](#Xu2020).

For the C++-OpenMP code, see [https://github.com/xu-yangyang/APAM](https://github.com/xu-yangyang/APAM).

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
make
```

Compile the code.  Example with `apam_mpi_logit_mnist`:

```sh
make apam_mpi_logit_mnist
```

or, if you want to compile all executables:

```sh
make
```

Run the code. Example running `apam_mpi_logit_mnist`, by using 4 mpi processes (1 master and 3 workers), each of which uses only 1 thread, under bash:

```sh
OMP_NUM_THREADS=1 mpirun -np 4 apam_mpi_logit_mnist
```

## Performance

Performance of `apam_mpi_logit_mnist` on a few machines:

On Ubuntu Linux 5.3.0-28.30~18.04.1, Intel(R) Xeon(R) CPU X5677 @ 3.47GHz, 8 cores, no GPU, OpenMPI 2.1.1

| # Workers | Test loss | Test accuracy | Train time (sec) | Speedup |
| :-------: | :-------: | :-----------: | :--------------: | :-----: |
| 1         | 0.2754    | 92.15%        | 43.417           | 1.00    |
| 2         | 0.2768    | 92.33%        | 22.105           | 1.96    |
| 3         | 0.2774    | 92.27%        | 16.979           | 2.55    |
| 4         | 0.2761    | 92.22%        | 14.467           | 3.00    |
| 5         | 0.2761    | 92.26%        | 12.516           | 3.46    |
| 6         | 0.2773    | 92.11%        | 11.209           | 3.87    |
| 7         | 0.2754    | 92.18%        | 10.223           | 4.24    |

On Ubuntu Linux 4.4.0-169, Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz, 64 cores, no GPU, MPICH 3.2

| # Workers | Test loss | Test accuracy | Train time (sec) | Speedup |
| :-------: | :-------: | :-----------: | :--------------: | :-----: |
| 1         | 0.2754    | 92.15%        | 44.441           | 1.00    |
| 2         | 0.2758    | 92.35%        | 22.406           | 1.98    |
| 4         | 0.2766    | 92.06%        | 11.465           | 3.87    |
| 8         | 0.2754    | 92.22%        | 5.5909           | 7.94    |
| 16        | 0.2784    | 92.21%        | 3.0306           | 14.66   |
| 32        | 0.2837    | 91.97%        | 1.7604           | 25.24   |

On macOS 10.15.1, 2.7 GHz Quad-Core Intel Core i7, no GPU, OpenMPI 4.0.2

| # Workers | Test loss | Test accuracy | Train time (sec) | Speedup |
| :-------: | :-------: | :-----------: | :--------------: | :-----: |
| 1         | 0.2754    | 92.15%        | 26.333           | 1.00    |
| 2         | 0.2757    | 92.30%        | 13.563           | 1.94    |
| 3         | 0.2781    | 92.17%        | 9.2720           | 2.84    |

<!--- More results here

Performance of `apam_mpi_mlp_mnist` on a few machines

On Ubuntu Linux 5.3.0-28.30~18.04.1, Intel(R) Xeon(R) CPU X5677 @ 3.47GHz, 8 cores, no GPU, OpenMPI 2.1.1

| # Workers | Test loss | Test accuracy | Train time (sec) | Speedup |
| :-------: | :-------: | :-----------: | :--------------: | :-----: |
| 1         | 0.0648    | 98.28%        | 286.66           | 1.00    |
| 2         | 0.0645    | 98.24%        | 159.13           | 1.80    |
| 3         | 0.0631    | 98.40%        | 160.37           | 1.78    |
| 4         | 0.0611    | 98.34%        | 168.99           | 1.69    |
| 5         | 0.0677    | 98.21%        | 176.55           | 1.62    |
| 6         | 0.0642    | 98.10%        | 191.19           | 1.49    |
| 7         | 0.0719    | 98.04%        | 223.47           | 1.28    |

On Ubuntu Linux 4.4.0-169, Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz, 64 cores, no GPU, MPICH 3.2

| # Workers | Test loss | Test accuracy | Train time (sec) | Speedup |
| :-------: | :-------: | :-----------: | :--------------: | :-----: |
| 1         | 0.0641    | 98.30%        | 98.186           | 1.00    |
| 2         | 0.0680    | 98.33%        | 51.333           | 1.91    |
| 4         | 0.0650    | 98.32%        | 29.692           | 3.30    |
| 8         | 0.0713    | 97.94%        | 27.691           | 3.54    |
| 16        | 0.1240    | 96.33%        | 28.456           | 3.45    |
| 32        | 0.1668    | 94.99%        | 32.120           | 3.05    |

On macOS 10.15.1, 2.7 GHz Quad-Core Intel Core i7, no GPU, OpenMPI 4.0.2

| # Workers | Test loss | Test accuracy | Train time (sec) | Speedup |
| :-------: | :-------: | :-----------: | :--------------: | :-----: |
| 1         | 0.0636    | 98.38%        | 82.041           | 1.00    |
| 2         | 0.0630    | 98.28%        | 42.654           | 1.92    |
| 3         | 0.0602    | 98.44%        | 32.748           | 2.50    |

Performance of `apam_mpi_lenet5_mnist` on a few machines

On Ubuntu Linux 5.3.0-28.30~18.04.1, Intel(R) Xeon(R) CPU X5677 @ 3.47GHz, 8 cores, no GPU, OpenMPI 2.1.1

| # Workers | Test loss | Test accuracy | Train time (sec) | Speedup |
| :-------: | :-------: | :-----------: | :--------------: | :-----: |
| 1         | 0.0455    | 98.52%        | 293.11           | 1.00    |
| 2         | 0.0469    | 98.60%        | 152.00           | 1.92    |
| 3         | 0.0442    | 98.62%        | 106.59           | 2.74    |
| 4         | 0.0630    | 98.00%        | 94.541           | 3.10    |
| 5         | 0.0694    | 97.76%        | 84.735           | 3.45    |
| 6         | 0.1642    | 94.64%        | 76.735           | 3.81    |
| 7         | 0.0773    | 97.72%        | 69.393           | 4.22    |

On Ubuntu Linux 4.4.0-169, Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz, 64 cores, no GPU, MPICH 3.2

| # Workers | Test loss | Test accuracy | Train time (sec) | Speedup |
| :-------: | :-------: | :-----------: | :--------------: | :-----: |
| 1         | 0.0458    | 98.59%        | 113.63           | 1.00    |
| 2         | 0.0460    | 98.65%        | 59.038           | 1.92    |
| 4         | 0.0474    | 98.48%        | 30.110           | 3.77    |
| 8         | 0.0954    | 97.00%        | 16.354           | 6.94    |
| 16        | 2.3010    | 11.35%        | 8.5322           | 13.31   |
| 32        | 2.3009    | 11.35%        | 4.9129           | 23.12   |

On macOS 10.15.1, 2.7 GHz Quad-Core Intel Core i7, no GPU, OpenMPI 4.0.2

| # Workers | Test loss | Test accuracy | Train time (sec) | Speedup |
| :-------: | :-------: | :-----------: | :--------------: | :-----: |
| 1         | 0.0445    | 98.62%        | 92.800           | 1.00    |
| 2         | 0.0507    | 98.38%        | 49.340           | 1.88    |
| 3         | 0.0373    | 98.84%        | 35.956           | 2.58    |

-->

## Reference

- <a name="Xu2020"></a>Yangyang Xu, Colin Sutcher-Shepard, Yibo Xu, and Jie Chen. [Asynchronous parallel adaptive stochastic gradient methods](https://arxiv.org/abs/2002.09095). Preprint arXiv:2002.09095, 2020.

