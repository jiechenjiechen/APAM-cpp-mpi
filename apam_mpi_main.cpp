// This code trains a neural network on a data set by using the APAM
// optimizer. It supports MPI/OpenMP hybrid parallelism.
//
// Currently the data set is hard-coded as MNIST.
//
// For a list of implemented neural networks, see nets.hpp. To use
// one, define the macro ARCHITECTURE by using the name of the
// architecture at compile time.
//
// For a list of changeable parameters, see the beginning of the main
// function. They are self-explanatory and they can be changed through
// command line options.

#include <getopt.h>
#include "apam_mpi_core.hpp"
#include "nets.hpp"

int main(int argc, char **argv) {

  // default parameters
  double lr = 1e-3;
  bool use_amsgrad = true;
  size_t train_batch_size = 128;
  size_t test_batch_size = 1000;
  size_t num_epochs = 20;
  int master_num_threads = 2;
  int mpi_thread_level = MPI_THREAD_MULTIPLE;
  bool use_gpu = false;
  bool use_sparse = false;     // sparse gradient
  bool debug_comm = false;     // if true, debugging info will print to screen
  char* debug_sparse_outfile = NULL; // if !NULL, will debug and print to file
  bool debug_sparse = false;         // determined by existence of outfile name
  char* debug_time_outfile = NULL;   // if !NULL, will debug and print to file
  bool debug_time = false;           // determined by existence of outfile name
  char* debug_grad_outfile = NULL;   // if !NULL, will debug and print to file
  bool debug_grad = false;           // determined by existence of outfile name
  bool timing = true;      // if false, will compute training error each epoch

  // process command line options (minimal error check)
  struct option long_options[] = {
    {"lr",                   required_argument, 0, 'l'},
    {"use_amsgrad",          required_argument, 0, 'a'},
    {"train_batch_size",     required_argument, 0, 'B'},
    {"test_batch_size",      required_argument, 0, 'b'},
    {"num_epochs",           required_argument, 0, 'e'},
    {"master_num_threads",   required_argument, 0, 'n'},
    {"mpi_thread_level",     required_argument, 0, 'm'},
    {"use_gpu",              required_argument, 0, 'G'},
    {"use_sparse",           required_argument, 0, 's'},
    {"debug_comm",           required_argument, 0, 'c'},
    {"debug_sparse_outfile", required_argument, 0, 'S'},
    {"debug_time_outfile",   required_argument, 0, 't'},
    {"debug_grad_outfile",   required_argument, 0, 'g'},
    {"timing",               required_argument, 0, 'T'},
    {0,                      0,                 0, 0}
  };
  char short_options[] = "a:b:B:c:e:g:G:l:m:n:s:S:t:T:";
  while (1) {
    int option_idx = 0;
    int c = getopt_long(argc, argv, short_options, long_options, &option_idx);
    if (c == -1) { // end of options
      break;
    }
    switch(c) {
    case 'l':  lr = atof(optarg);                                     break;
    case 'a':  use_amsgrad = atoi(optarg) ? true : false;             break;
    case 'B':  train_batch_size = atoi(optarg);                       break;
    case 'b':  test_batch_size = atoi(optarg);                        break;
    case 'e':  num_epochs = atoi(optarg);                             break;
    case 'n':  master_num_threads = atoi(optarg);                     break;
    case 'm':
      if (strcmp(optarg, "MPI_THREAD_SINGLE") == 0) {
        mpi_thread_level = MPI_THREAD_SINGLE; }
      else if (strcmp(optarg, "MPI_THREAD_FUNNELED") == 0) {
        mpi_thread_level = MPI_THREAD_FUNNELED; }
      else if (strcmp(optarg, "MPI_THREAD_SERIALIZED") == 0) {
        mpi_thread_level = MPI_THREAD_SERIALIZED; }
      else /* (strcmp(optarg, "MPI_THREAD_MULTIPLE") == 0) */ {
        mpi_thread_level = MPI_THREAD_MULTIPLE; }                     break;
    case 'G':  use_gpu = atoi(optarg) ? true : false;                 break;
    case 's':  use_sparse = atoi(optarg) ? true : false;              break;
    case 'c':  debug_comm = atoi(optarg) ? true : false;              break;
    case 'S':  debug_sparse_outfile = optarg; debug_sparse = true;    break;
    case 't':  debug_time_outfile = optarg;   debug_time = true;      break;
    case 'g':  debug_grad_outfile = optarg;   debug_grad = true;      break;
    case 'T':  timing = atoi(optarg) ? true : false;                  break;
    }
  }
  
  // mpi initialization
#ifdef USE_OPENMP
  int mpi_thread_provided;
  MPI_Init_thread(&argc, &argv, mpi_thread_level, &mpi_thread_provided);
#else
  MPI_Init(&argc, &argv);
#endif
  int myrank, nranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  // print mpi threading diagnostics (done by only the master)
#ifdef USE_OPENMP
  if (myrank == ROOT) {
    printf("process %d: mpi_thread_level = %d, mpi_thread_provided = %d\n",
           myrank, mpi_thread_level, mpi_thread_provided);
  }
#endif

  // set num threads
#ifdef USE_OPENMP
  if (myrank == ROOT) {
    omp_set_num_threads(master_num_threads);
  }
  else {
    omp_set_num_threads(1);
  }
#endif

  // pytorch setup
  size_t seed = myrank;                   // seed
  torch::manual_seed(seed);
  torch::DeviceType device_type =         // device
    (torch::cuda::is_available() && use_gpu) ? torch::kCUDA : torch::kCPU;
  torch::Device device(device_type);
  ARCHITECTURE model;                     // move net to device
  model.to(device);
  APAM<ARCHITECTURE> optimizer(model, lr, use_amsgrad, use_sparse); //optimizer

  // print gpu usage (done by only the master)
  if (myrank == ROOT) {
    printf("process %d: Use gpu = %s; gpu available = %s\n", myrank,
           use_gpu ? "true" : "false",
           torch::cuda::is_available() ? "true" : "false");
  }
  
  // training dataset and data loader (each process has a copy)
  auto train_dataset = torch::data::datasets::MNIST("./data")
    .map(torch::data::transforms::Normalize<>(0.1307, 0.3081)) // preprocessing
    .map(torch::data::transforms::Stack<>());
  const size_t train_dataset_size = train_dataset.size().value();
  auto train_loader =
    torch::data::make_data_loader<torch::data::samplers::RandomSampler>
    (std::move(train_dataset), train_batch_size); // RandomSampler will shuffle

  // other parameters
  int num_iter_per_epoch = (int)ceil((double)train_dataset_size /
                                     train_batch_size);
  int maxiter = num_epochs * num_iter_per_epoch;
  bool check_progress = timing ? false : true;

  // training
  double time_start = MPI_Wtime();
  if (myrank == ROOT) {
    train_master(model, device, *train_loader, train_dataset_size, optimizer,
                 maxiter, num_iter_per_epoch, use_sparse, debug_comm,
                 debug_sparse, debug_sparse_outfile, debug_time,
                 debug_time_outfile, debug_grad, debug_grad_outfile,
                 check_progress);
  }
  else {
    train_worker(model, device, *train_loader, train_dataset_size, optimizer,
                 use_sparse, debug_comm, check_progress);
  }
  double time_end = MPI_Wtime();

  // print timing (done by only the master)
  if (timing && myrank == ROOT) {
    double time_elapsed = time_end - time_start;
    printf("process %d: Training time %g seconds (with %d workers)\n",
           myrank, time_elapsed, nranks-1);
  }

  // testing (done by only the master)
  if (myrank == ROOT) {

    // testing data and data loader
    auto test_dataset = torch::data::datasets::
      MNIST("./data", torch::data::datasets::MNIST::Mode::kTest)
      .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
      .map(torch::data::transforms::Stack<>());
    const size_t test_dataset_size = test_dataset.size().value();
    auto test_loader =
      torch::data::make_data_loader(std::move(test_dataset), test_batch_size);

    // testing (done by only the master)
    test(model, device, *test_loader, test_dataset_size, "test set");
  }

  // clean up and return
  MPI_Finalize();
  return 0;
}

