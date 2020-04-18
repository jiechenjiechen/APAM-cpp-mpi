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
| 1         | 0.2769    | 92.15%        | 43.465           | 1.00    |
| 2         | 0.2776    | 92.18%        | 22.344           | 1.94    |
| 3         | 0.2780    | 92.05%        | 16.456           | 2.64    |
| 4         | 0.2777    | 92.13%        | 14.411           | 3.01    |
| 5         | 0.2786    | 92.10%        | 12.635           | 3.44    |
| 6         | 0.2793    | 92.10%        | 11.308           | 3.84    |
| 7         | 0.2779    | 92.14%        | 10.155           | 4.28    |

On Ubuntu Linux 4.4.0-169, Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz, 64 cores, no GPU, MPICH 3.2

| # Workers | Test loss | Test accuracy | Train time (sec) | Speedup |
| :-------: | :-------: | :-----------: | :--------------: | :-----: |
| 1         | 0.2769    | 92.15%        | 43.999           | 1.00    |
| 2         | 0.2776    | 92.09%        | 21.388           | 2.05    |
| 4         | 0.2787    | 92.23%        | 11.366           | 3.87    |
| 8         | 0.2769    | 92.23%        | 5.8701           | 7.49    |
| 16        | 0.2797    | 91.98%        | 2.7817           | 15.81   |
| 32        | 0.2873    | 91.95%        | 1.7131           | 25.68   |

On macOS 10.15.1, 2.7 GHz Quad-Core Intel Core i7, no GPU, OpenMPI 4.0.2

| # Workers | Test loss | Test accuracy | Train time (sec) | Speedup |
| :-------: | :-------: | :-----------: | :--------------: | :-----: |
| 1         | 0.2769    | 92.15%        | 26.661           | 1.00    |
| 2         | 0.2774    | 92.11%        | 13.531           | 1.97    |
| 3         | 0.2779    | 92.05%        | 9.4176           | 2.83    |

<!--- More results here

Performance of `apam_mpi_mlp_mnist` on a few machines

On Ubuntu Linux 5.3.0-28.30~18.04.1, Intel(R) Xeon(R) CPU X5677 @ 3.47GHz, 8 cores, no GPU, OpenMPI 2.1.1

| # Workers | Test loss | Test accuracy | Train time (sec) | Speedup |
| :-------: | :-------: | :-----------: | :--------------: | :-----: |
| 1         | 0.0650    | 98.21%        | 194.00           | 1.00    |
| 2         | 0.0620    | 98.41%        | 106.26           | 1.82    |
| 3         | 0.0624    | 98.33%        | 83.592           | 2.32    |
| 4         | 0.0623    | 98.33%        | 98.080           | 1.97    |
| 5         | 0.0629    | 98.38%        | 117.23           | 1.65    |
| 6         | 0.0611    | 98.26%        | 126.05           | 1.53    |
| 7         | 0.0659    | 98.19%        | 146.99           | 1.31    |

On Ubuntu Linux 4.4.0-169, Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz, 64 cores, no GPU, MPICH 3.2

| # Workers | Test loss | Test accuracy | Train time (sec) | Speedup |
| :-------: | :-------: | :-----------: | :--------------: | :-----: |
| 1         | 0.0655    | 98.31%        | 97.672           | 1.00    |
| 2         | 0.0620    | 98.40%        | 51.653           | 1.89    |
| 4         | 0.0619    | 98.37%        | 29.737           | 3.28    |
| 8         | 0.0744    | 98.01%        | 28.667           | 3.40    |
| 16        | 0.0740    | 97.86%        | 29.776           | 3.28    |
| 32        | 0.1546    | 95.27%        | 32.792           | 2.97    |

On macOS 10.15.1, 2.7 GHz Quad-Core Intel Core i7, no GPU, OpenMPI 4.0.2

| # Workers | Test loss | Test accuracy | Train time (sec) | Speedup |
| :-------: | :-------: | :-----------: | :--------------: | :-----: |
| 1         | 0.0644    | 98.26%        | 78.676           | 1.00    |
| 2         | 0.0624    | 98.40%        | 42.913           | 1.83    |
| 3         | 0.0617    | 98.38%        | 32.674           | 2.40    |

Performance of `apam_mpi_lenet5_mnist` on a few machines

On Ubuntu Linux 5.3.0-28.30~18.04.1, Intel(R) Xeon(R) CPU X5677 @ 3.47GHz, 8 cores, no GPU, OpenMPI 2.1.1

| # Workers | Test loss | Test accuracy | Train time (sec) | Speedup |
| :-------: | :-------: | :-----------: | :--------------: | :-----: |
| 1         | 0.0436    | 98.91%        | 558.33           | 1.00    |
| 2         | 0.0373    | 99.10%        | 289.96           | 1.92    |
| 3         | 0.0412    | 98.97%        | 216.46           | 2.57    |
| 4         | 0.0464    | 98.94%        | 174.65           | 3.19    |
| 5         | 0.0453    | 98.83%        | 162.76           | 3.43    |
| 6         | 0.0556    | 98.41%        | 143.95           | 3.87    |
| 7         | 0.0461    | 98.75%        | 134.34           | 4.15    |

On Ubuntu Linux 4.4.0-169, Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz, 64 cores, no GPU, MPICH 3.2

| # Workers | Test loss | Test accuracy | Train time (sec) | Speedup |
| :-------: | :-------: | :-----------: | :--------------: | :-----: |
| 1         | 0.0431    | 99.02%        | 231.92           | 1.00    |
| 2         | 0.0428    | 98.82%        | 117.22           | 1.97    |
| 4         | 0.0402    | 99.02%        | 62.368           | 3.71    |
| 8         | 0.0370    | 98.87%        | 31.682           | 7.32    |
| 16        | 0.0587    | 98.34%        | 16.412           | 14.13   |
| 32        | 2.3013    | 11.35%        | 9.0915           | 25.50   |

On macOS 10.15.1, 2.7 GHz Quad-Core Intel Core i7, no GPU, OpenMPI 4.0.2

| # Workers | Test loss | Test accuracy | Train time (sec) | Speedup |
| :-------: | :-------: | :-----------: | :--------------: | :-----: |
| 1         | 0.0440    | 98.91%        | 185.08           | 1.00    |
| 2         | 0.0442    | 98.96%        | 98.923           | 1.87    |
| 3         | 0.0410    | 98.96%        | 71.099           | 2.60    |

-->

## Reference

- <a name="Xu2020"></a>Yangyang Xu, Colin Sutcher-Shepard, Yibo Xu, and Jie Chen. [Asynchronous parallel adaptive stochastic gradient methods](https://arxiv.org/abs/2002.09095). Preprint arXiv:2002.09095, 2020.

