// Logistic regression on MNIST data set.

#include "apam_mpi.hpp"

//-----------------------------------------------------------------------------

// define the network architecture
struct LOGIT : torch::nn::Module {
  torch::nn::Linear fc{nullptr};
  
  LOGIT() {
    fc = register_module("fc", torch::nn::Linear(784, 10));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::flatten(x, 1);
    x = fc->forward(x);
    x = torch::log_softmax(x, 1);
    return x;  // output
  }
};

//-----------------------------------------------------------------------------

int main(int argc, char **argv) {

  // mpi initialization
  MPI_Init(&argc, &argv);
  int myrank, nranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  // parameters and options
  double lr = 1e-4;
  bool use_amsgrad = true;
  size_t train_batch_size = 128;
  size_t test_batch_size = 1000;
  size_t num_epochs = 20;
  bool use_gpu = false;
  bool debug_gpu = false;    // if true, debugging info will print to screen
  bool use_sparse = false;   // sparse gradient
  bool debug_sparse = false; // if true, debugging info will print to file
  std::string debug_sparse_outfile = "debug_logit_sparse.txt";
  bool debug_comm = false;   // if true, debugging info will print to screen
  bool debug_time = false;   // if true, debugging info will print to file
  std::string debug_time_outfile = "debug_logit_time.txt";
  bool debug_grad = false;   // if true, debugging info will print to file
  std::string debug_grad_outfile = "debug_logit_grad.txt";
  bool timing = true; // if false, will compute training error each epoch

  // pytorch setup
  size_t seed = myrank;                   // seed
  torch::manual_seed(seed);
  torch::DeviceType device_type =         // device
    (torch::cuda::is_available() && use_gpu) ? torch::kCUDA : torch::kCPU;
  torch::Device device(device_type);
  LOGIT model;                            // move net to device
  model.to(device);
  APAM<LOGIT> optimizer(model, lr, use_amsgrad, use_sparse); // set optimizer

  // print gpu usage (done by only the master)
  if (debug_gpu && myrank == ROOT) {
    printf("process %d: Use gpu (%s); gpu available (%s)\n", myrank,
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
  clock_t time_start = clock();
  if (myrank == ROOT) {
    train_master(model, device, *train_loader, train_dataset_size, optimizer,
                 maxiter, num_iter_per_epoch, use_sparse, debug_sparse,
                 debug_sparse_outfile, debug_comm, debug_time,
                 debug_time_outfile, debug_grad, debug_grad_outfile,
                 check_progress);
  }
  else {
    train_worker(model, device, *train_loader, train_dataset_size, optimizer,
                 use_sparse, debug_comm, check_progress);
  }
  clock_t time_end = clock();

  // print timing (done by only the master)
  if (timing && myrank == ROOT) {
    double time_elapsed = (double)(time_end - time_start) / CLOCKS_PER_SEC;
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

