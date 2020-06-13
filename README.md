# APAM: Asynchronous Parallel Adaptive stochastic gradient Method

This repository contains a C++-MPI implementation of APAM [(Xu et al. 2020)](#Xu2020).

For the C++-OpenMP code, see [https://github.com/xu-yangyang/APAM](https://github.com/xu-yangyang/APAM).

## Usage

Make sure you have cmake with version >= 3.1 and a C++ compiler that supports C++14 standard.

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

Run the code. Example running `apam_mpi_lenet5_mnist`, by using 4 mpi processes (1 master and 3 workers), each of which uses only 1 thread, under bash:

```sh
OMP_NUM_THREADS=1 mpirun -np 4 apam_mpi_lenet5_mnist
```

## Performance

Performance of `apam_mpi_lenet5_mnist` on a few machines:

On Ubuntu Linux 5.3.0-28.30~18.04.1, Intel(R) Xeon(R) CPU X5677 @ 3.47GHz, 8 cores, no GPU, OpenMPI 2.1.1

| # Workers | Test loss | Test accuracy | Train time (sec) | Speedup |
| :-------: | :-------: | :-----------: | :--------------: | :-----: |
| 1         | 0.0336    | 98.96%        | 1104.6           | 1.00    |
| 2         | 0.0396    | 98.93%        | 580.97           | 1.90    |
| 3         | 0.0344    | 98.94%        | 422.01           | 2.61    |
| 4         | 0.0344    | 98.92%        | 348.05           | 3.17    |
| 5         | 0.0351    | 98.94%        | 324.27           | 3.40    |
| 6         | 0.0357    | 98.91%        | 292.87           | 3.77    |
| 7         | 0.0337    | 98.90%        | 270.57           | 4.08    |

On Ubuntu Linux 4.4.0-169, Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz, 64 cores, no GPU, MPICH 3.2

| # Workers | Test loss | Test accuracy | Train time (sec) | Speedup |
| :-------: | :-------: | :-----------: | :--------------: | :-----: |
| 1         | 0.0327    | 99.01%        | 451.86           | 1.00    |
| 2         | 0.0357    | 98.82%        | 232.47           | 1.94    |
| 4         | 0.0375    | 98.90%        | 125.20           | 3.60    |
| 8         | 0.0347    | 98.95%        | 64.225           | 7.03    |
| 16        | 0.0342    | 98.85%        | 33.285           | 13.57   |
| 32        | 0.0406    | 98.67%        | 17.882           | 25.26   |

On macOS 10.15.1, 2.7 GHz Quad-Core Intel Core i7, no GPU, OpenMPI 4.0.2

| # Workers | Test loss | Test accuracy | Train time (sec) | Speedup |
| :-------: | :-------: | :-----------: | :--------------: | :-----: |
| 1         | 0.0329    | 99.03%        | 376.11           | 1.00    |
| 2         | 0.0378    | 98.87%        | 200.80           | 1.87    |
| 3         | 0.0334    | 98.95%        | 149.17           | 2.52    |

<!--- More results here

Performance of `apam_mpi_mlp_mnist` on a few machines

On Ubuntu Linux 5.3.0-28.30~18.04.1, Intel(R) Xeon(R) CPU X5677 @ 3.47GHz, 8 cores, no GPU, OpenMPI 2.1.1

| # Workers | Test loss | Test accuracy | Train time (sec) | Speedup |
| :-------: | :-------: | :-----------: | :--------------: | :-----: |
| 1         | 0.0650    | 98.21%        | 193.54           | 1.00    |
| 2         | 0.0611    | 98.42%        | 103.31           | 1.87    |
| 3         | 0.0607    | 98.38%        | 86.580           | 2.23    |
| 4         | 0.0617    | 98.36%        | 95.263           | 2.03    |
| 5         | 0.0615    | 98.29%        | 122.34           | 1.58    |
| 6         | 0.0653    | 98.28%        | 128.94           | 1.50    |
| 7         | 0.0646    | 98.19%        | 130.90           | 1.47    |

On Ubuntu Linux 4.4.0-169, Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz, 64 cores, no GPU, MPICH 3.2

| # Workers | Test loss | Test accuracy | Train time (sec) | Speedup |
| :-------: | :-------: | :-----------: | :--------------: | :-----: |
| 1         | 0.0655    | 98.31%        | 97.639           | 1.00    |
| 2         | 0.0629    | 98.43%        | 50.844           | 1.92    |
| 4         | 0.0630    | 98.42%        | 28.892           | 3.37    |
| 8         | 0.0647    | 98.25%        | 27.784           | 3.51    |
| 16        | 0.0727    | 97.81%        | 27.966           | 3.49    |
| 32        | 0.1615    | 95.09%        | 32.528           | 3.00    |

On macOS 10.15.1, 2.7 GHz Quad-Core Intel Core i7, no GPU, OpenMPI 4.0.2

| # Workers | Test loss | Test accuracy | Train time (sec) | Speedup |
| :-------: | :-------: | :-----------: | :--------------: | :-----: |
| 1         | 0.0644    | 98.26%        | 79.470           | 1.00    |
| 2         | 0.0650    | 98.29%        | 43.496           | 1.82    |
| 3         | 0.0605    | 98.29%        | 34.900           | 2.27    |

Performance of `apam_mpi_logit_mnist` on a few machines

On Ubuntu Linux 5.3.0-28.30~18.04.1, Intel(R) Xeon(R) CPU X5677 @ 3.47GHz, 8 cores, no GPU, OpenMPI 2.1.1

| # Workers | Test loss | Test accuracy | Train time (sec) | Speedup |
| :-------: | :-------: | :-----------: | :--------------: | :-----: |
| 1         | 0.2769    | 92.15%        | 43.241           | 1.00    |
| 2         | 0.2778    | 92.17%        | 22.890           | 1.88    |
| 3         | 0.2792    | 92.06%        | 16.946           | 2.55    |
| 4         | 0.2783    | 92.22%        | 14.309           | 3.02    |
| 5         | 0.2782    | 92.13%        | 12.622           | 3.42    |
| 6         | 0.2787    | 92.13%        | 11.332           | 3.81    |
| 7         | 0.2776    | 92.24%        | 10.186           | 4.24    |

On Ubuntu Linux 4.4.0-169, Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz, 64 cores, no GPU, MPICH 3.2

| # Workers | Test loss | Test accuracy | Train time (sec) | Speedup |
| :-------: | :-------: | :-----------: | :--------------: | :-----: |
| 1         | 0.2769    | 92.15%        | 43.633           | 1.00    |
| 2         | 0.2783    | 92.18%        | 23.890           | 1.82    |
| 4         | 0.2791    | 91.93%        | 11.608           | 3.75    |
| 8         | 0.2782    | 92.18%        | 5.8105           | 7.50    |
| 16        | 0.2784    | 92.25%        | 2.8638           | 15.23   |
| 32        | 0.2869    | 92.02%        | 1.8435           | 23.66   |

On macOS 10.15.1, 2.7 GHz Quad-Core Intel Core i7, no GPU, OpenMPI 4.0.2

| # Workers | Test loss | Test accuracy | Train time (sec) | Speedup |
| :-------: | :-------: | :-----------: | :--------------: | :-----: |
| 1         | 0.2769    | 92.15%        | 26.245           | 1.00    |
| 2         | 0.2773    | 92.12%        | 13.471           | 1.94    |
| 3         | 0.2787    | 92.14%        | 9.3800           | 2.79    |

-->

## Reference

- <a name="Xu2020"></a>Yangyang Xu, Colin Sutcher-Shepard, Yibo Xu, and Jie Chen. [Asynchronous parallel adaptive stochastic gradient methods](https://arxiv.org/abs/2002.09095). Preprint arXiv:2002.09095, 2020.

