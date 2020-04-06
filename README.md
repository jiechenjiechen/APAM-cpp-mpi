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

Under the code directory, compile code. The `/absolute/path/to/libtorch` below is where you unzip in the last step, concatenated with the folder name.

```sh
mkdir build
cd build
cmake -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
make
```

Run the code. Example by using 4 mpi processes (1 master and 3 workers), each of which uses only 1 thread, under bash:

```sh
OMP_NUM_THREADS=1 mpirun -np 4 lenet5_mnist_mpi
```

## Performance

On Ubuntu Linux 5.3.0-28.30~18.04.1, Intel(R) Xeon(R) CPU X5677 @ 3.47GHz, 8 cores, no GPU, OpenMPI 2.1.1

| # Workers | Test loss | Test accuracy | Train time (sec) | Speedup |
| :-------: | :-------: | :-----------: | :--------------: | :-----: |
| 1         | 0.0455    | 98.52%        | 292.64           | 1.00    |
| 2         | 0.0480    | 98.62%        | 149.63           | 1.95    |
| 3         | 0.0386    | 98.72%        | 109.13           | 2.68    |
| 4         | 0.0499    | 98.38%        | 90.013           | 3.25    |
| 5         | 0.0908    | 97.10%        | 84.985           | 3.44    |
| 6         | 0.0921    | 97.11%        | 76.063           | 3.84    |
| 7         | 0.0956    | 97.03%        | 71.415           | 4.09    |

On Ubuntu Linux 4.4.0-169, Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz, 64 cores, no GPU, MPICH 3.2

| # Workers | Test loss | Test accuracy | Train time (sec) | Speedup |
| :-------: | :-------: | :-----------: | :--------------: | :-----: |
| 1         | 0.0458    | 98.59%        | 113.24           | 1.00    |
| 2         | 0.0441    | 98.61%        | 59.630           | 1.89    |
| 4         | 0.0491    | 98.40%        | 30.249           | 3.74    |
| 8         | 0.0920    | 97.28%        | 15.561           | 7.27    |
| 16        | 2.3010    | 11.35%        | 8.2346           | 13.75   |
| 32        | 2.3009    | 11.35%        | 4.6895           | 24.14   |

On macOS 10.15.1, 2.7 GHz Quad-Core Intel Core i7, no GPU, OpenMPI 4.0.2

| # Workers | Test loss | Test accuracy | Train time (sec) | Speedup |
| :-------: | :-------: | :-----------: | :--------------: | :-----: |
| 1         | 0.0445    | 98.62%        | 93.10            | 1.00    |
| 2         | 0.0392    | 98.73%        | 49.92            | 1.86    |
| 3         | 0.0429    | 98.62%        | 36.64            | 2.54    |

## Reference

- <a name="Xu2020"></a>Yangyang Xu, Colin Sutcher-Shepard, Yibo Xu, and Jie Chen. [Asynchronous parallel adaptive stochastic gradient methods](https://arxiv.org/abs/2002.09095). Preprint arXiv:2002.09095, 2020.

