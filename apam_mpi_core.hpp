// This code implements the APAM optimizer with MPI parallelization.
//
// For C++, we manipulate primitive data types (e.g., float), because
// mpi can only handle these types; but for python, we can manipulate
// pytorch data types (e.g., torch::Tensor), because pytorch has an
// mpi interface handling such.

#ifndef _APAM_MPI_CORE_
#define _APAM_MPI_CORE_

#include <mpi.h>
#ifdef USE_OPENMP
#include <omp.h>
#endif
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <torch/torch.h>

// global constant
const int ROOT = 0;       // this is the master
const int DONE = 999999;  // the "done" flag. choose some special number
const int NOT_DONE = 1;   // some flag different from "done"

//-----------------------------------------------------------------------------

// sparse gradient g
// =================
// The gradient g may be sparse, due to, for example, the use of
// ReLU. We design a sparse data structure to store a gradient. The
// storage may be substantially reduced if g is sufficiently sparse
// (however, it is also possible that the storage will increase if
// there is not much sparsity).
//
// Specifically, we use a block sparse data structure. The array reads
// like the following:
//
// [length of this array
//  starting_location, num_nonzero (M), val1, val2, ..., valM,
//  ...
//  starting_location', num_nonzero (M'), val1, val2, ..., valM'].
//
// For better logic and nterpretability, we implement operations with
// sparse gradient inside the optimizer.

//-----------------------------------------------------------------------------

// define the optimizer. the official APAM is based on AMSGRAD, but it
// should also work on the base of ADAM. AMSGRAD differs from ADAM in
// only one line
template <typename Net>
class APAM {

public:
  APAM(Net &model,
       double alpha = 1e-3,
       bool amsgrad = true,
       bool sparse_g = false,    // whether use sparse gradient
       double beta1 = 0.9,
       double beta2 = 0.999,
       double epsilon = 1e-8);
  ~APAM();

  // the meat
  void unpack_w(float *w);       // unpack from float array to tensor
  void pack_w(float *w);         // pack from tensor to float array
  void pack_g(float *g);         // pack from tensor to float array
  void zero_grad(void);          // called by worker. conform with pytorch
  void step(float *g, float *w); // called by master. conform with pytorch

  // utility functions
  int get_num_param(void);

private:
  Net& _model;
  double _alpha;
  double _beta1;
  double _beta2;
  double _epsilon;
  bool _amsgrad;
  bool _sparse_g;
  int num_set;   // as in "number of weight matrices"
  int *set_size; // as in "number of elements in a weight matrix"
  int num_param; // = sum(set_size)
  float *m;      // storage
  float *v;      // storage
  float *v_hat;  // storage (only used when amsgrad is true)
  int t;         // iteration number
  float *full_g; // full gradient (only used when sparse_g is true)

  // not sure if this function should be made public or not
  void reset(void);
};

template <typename Net>
APAM<Net>::
APAM(Net &model,
     double alpha,
     bool amsgrad,
     bool sparse_g,
     double beta1,
     double beta2,
     double epsilon) : _model(model) {

  _alpha = alpha;
  _beta1 = beta1;
  _beta2 = beta2;
  _epsilon = epsilon;
  _amsgrad = amsgrad;
  _sparse_g = sparse_g;
  num_set = _model.parameters().size();
  set_size = new int [num_set];
  num_param = 0;
  for (int i = 0; i < num_set; i++) {
    set_size[i] = _model.parameters()[i].numel();
    num_param += set_size[i];
  }
  m = new float [num_param];
  v = new float [num_param];
  if (_amsgrad) {
    v_hat = new float [num_param];
  }
  else {
    v_hat = NULL;
  }
  if (_sparse_g) {
    full_g = new float [num_param];
  }
  else {
    full_g = NULL;
  }
  reset();
}

template <typename Net>
APAM<Net>::
~APAM() {
  delete [] set_size;
  delete [] m;
  delete [] v;
  if (_amsgrad) {
    delete [] v_hat;
  }
  if (_sparse_g) {
    delete [] full_g;
  }
}

template <typename Net>
void APAM<Net>::
reset(void) {
  t = 0;
  memset(m, 0, num_param*sizeof(float));
  memset(v, 0, num_param*sizeof(float));
  if (_amsgrad) {
    memset(v_hat, 0, num_param*sizeof(float));
  }
  if (_sparse_g) {
    memset(full_g, 0, num_param*sizeof(float));
  }
}

template <typename Net>
void APAM<Net>::
unpack_w(float *w) {
  float *tw = NULL;
  int offset = 0;
  for (int i = 0; i < num_set; i++) {
    tw = static_cast<float*>(_model.parameters()[i].storage().data());
    memcpy(tw, w+offset, set_size[i]*sizeof(float));
    offset += set_size[i];
  }
}

template <typename Net>
void APAM<Net>::
pack_w(float *w) {
  float *tw = NULL;
  int offset = 0;
  for (int i = 0; i < num_set; i++) {
    tw = static_cast<float*>(_model.parameters()[i].storage().data());
    memcpy(w+offset, tw, set_size[i]*sizeof(float));
    offset += set_size[i];
  }
}

template <typename Net>
void APAM<Net>::
pack_g(float *g) {
  float *tg = NULL;
  int offset = 0;
  float *g_ptr = _sparse_g ? full_g : g;
  for (int i = 0; i < num_set; i++) {
    tg = static_cast<float*>(_model.parameters()[i].grad().storage().data());
    memcpy(g_ptr+offset, tg, set_size[i]*sizeof(float));
    offset += set_size[i];
  }

  // convert from full array to sparse array
  if (_sparse_g) {
    bool nonzero = false;
    int full_g_current = 0;
    int nz_count = 0;
    int g_current = 1;
    // loop over the entries of the full array
    while (full_g_current < num_param) {
      if (nonzero == true && full_g[full_g_current] != 0.0) {
        // past element is nonzero and current element is nonzero:
        // just routine bookkeeping
        nz_count++;
      }
      else if (nonzero == true && full_g[full_g_current] == 0.0) {
        // past element is nonzero and current element is zero: a
        // nonzero segment is complete. copy to sparse array
        g[g_current] = full_g_current - nz_count;
        g[g_current+1] = nz_count;
        memcpy(g + g_current + 2, full_g + full_g_current - nz_count,
               nz_count*sizeof(float));
        nonzero = false;
        g_current += 2 + nz_count;
        nz_count = 0;
      }
      else if (nonzero == false && full_g[full_g_current] != 0.0) {
        // past element is zero and current element is nonzero: start
        // bookkeeping a new nonzero segment
        nonzero = true;
        nz_count = 1;
      }
      else { // (nonzero == false && full_g[full_g_current] == 0.0)
        // past element is zero and current element is zero: do
        // nothing
      }
      full_g_current++; // increase full array index
    }
    // now reaching the end of the full array. if past element is
    // nonzero, the corresponding segment has not been recorded and
    // will need to be recorded
    if (nonzero == true) {
      g[g_current] = full_g_current - nz_count;
      g[g_current+1] = nz_count;
      memcpy(g + g_current + 2, full_g + full_g_current - nz_count,
             nz_count*sizeof(float));
      g_current += 2 + nz_count;
    }
    // finally, set g[0]
    g[0] = g_current;
  }
}

template <typename Net>
void APAM<Net>::
zero_grad(void) {
  float *tg = NULL;
  for (int i = 0; i < num_set; i++) {
    if (!_model.parameters()[i].grad().defined()) {
      continue;
    }
    tg = static_cast<float*>(_model.parameters()[i].grad().storage().data());
    memset(tg, 0, set_size[i]*sizeof(float));
  }
}

template <typename Net>
void APAM<Net>::
step(float *g, float *w) {
  
  // note two differences from the APAM paper: (1) the calculation of
  // lr; (2) the use of "+ _epsilon" in the denominator
  t++;
  double lr = _alpha * sqrt(1 - pow(_beta2,t)) / (1 - pow(_beta1,t));
  double one_minus_beta1 = 1 - _beta1;
  double one_minus_beta2 = 1 - _beta2;
  if (_sparse_g == false) { // full gradient
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < num_param; i++) {
      m[i] = _beta1 * m[i] + one_minus_beta1 * g[i];
      v[i] = _beta2 * v[i] + one_minus_beta2 * g[i] * g[i];
      if (_amsgrad) {
        v_hat[i] = std::max(v_hat[i], v[i]);
        w[i] -= lr * m[i] / (sqrt(v_hat[i]) + _epsilon);
      }
      else {
        w[i] -= lr * m[i] / (sqrt(v[i]) + _epsilon);
      }
    }
  }
  else { // sparse gradient
    bool nonzero = (g[0] == 1 || g[1] != 0) ? false : true;
    int g_current = (g[0] == 1) ? -1 : 3;
    int i_start = (g[0] == 1) ? -1 : (int)g[1];
    int i_end = (g[0] == 1) ? -1 : (int)(g[1]+g[2]);
    for (int i = 0; i < num_param; i++) {
      if (nonzero) {
        m[i] = _beta1 * m[i] + one_minus_beta1 * g[g_current];
        v[i] = _beta2 * v[i] + one_minus_beta2 * g[g_current] * g[g_current];
        if (i+1 == i_end) { // reaching the end of the nonzero segment
          nonzero = false;
          if (g_current+1 < (int)g[0]) {
            i_start = (int)g[g_current+1];
            i_end = (int)(g[g_current+1]+g[g_current+2]);
            g_current += 3;
          }
        }
        else {
          g_current++;
        }
      }
      else {
        m[i] = _beta1 * m[i];
        v[i] = _beta2 * v[i];
        if (i+1 == i_start) { // reaching the start of a nonzero segment
          nonzero = true;
        }
      }
      if (_amsgrad) {
        v_hat[i] = std::max(v_hat[i], v[i]);
        w[i] -= lr * m[i] / (sqrt(v_hat[i]) + _epsilon);
      }
      else {
        w[i] -= lr * m[i] / (sqrt(v[i]) + _epsilon);
      }
    }
  }
}

template <typename Net>
int APAM<Net>::
get_num_param(void) {
  return num_param;
}

//-----------------------------------------------------------------------------

// training and testing codes

template <typename Net, typename DataLoader>
void train_master(Net& model,
                  torch::Device device,
                  DataLoader& dataloader,
                  size_t dataset_size,
                  APAM<Net>& optimizer,
                  int maxiter,
                  int num_iter_per_epoch,
                  bool use_sparse,
                  bool debug_comm,
                  bool debug_sparse,
                  const char* debug_sparse_outfile,
                  bool debug_time,
                  const char* debug_time_outfile,
                  bool debug_grad,
                  const char* debug_grad_outfile,
                  bool check_progress) {

  // mpi context
  int myrank, nranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  // allocate storage
  int N = optimizer.get_num_param();
  int buf_sz = use_sparse ? 2*N+1 : N; // gradient buffer size
  if (N == 1) { buf_sz = 4; }
  // for sparse gradient, more space needed in worst case
  float **gg = NULL; // gradient (one slot for each worker)
  gg = new float* [nranks];
  for (int i = 0; i < nranks; i++) {
    gg[i] = new float [buf_sz];
  }
  float *w = NULL;   // model parameter
  w = new float [N];

  // debug timing (will output timing information to file)
  double time1, time2;
  int *elapse = NULL;
  int elapse_count = 0;
  if (debug_time) {
    time1 = MPI_Wtime();
    elapse = new int [(maxiter+nranks)*3];
  }
  
  // master and workers should use the same initialization
  optimizer.pack_w(w); // tensor -> float*
  MPI_Bcast(w, N, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
  
  // initiate nonblocking receives from all workers
  MPI_Request *recv_request = NULL;
  recv_request = new MPI_Request [nranks];
  int *recv_count = NULL;
  recv_count = new int [nranks];
  for (int i = 0; i < nranks; i++) {
    if (i == ROOT) {
      recv_request[i] = MPI_REQUEST_NULL;
    }
    else {
      if (MPI_Irecv(gg[i], buf_sz, MPI_FLOAT, i, MPI_ANY_TAG, MPI_COMM_WORLD,
                    &recv_request[i]) != MPI_SUCCESS) {
        printf("master %d: Error in MPI_Irecv!", myrank); exit(1);
      }
      // on completed receive, if we do
      // MPI_Get_count(&recv_request[i], MPI_INT, &count), count must
      // be the same as (int)gg[i][0]
    }
  }

  // print training set loss and accuracy (time consuming)
  if (check_progress) {
    printf("process %d: Train epoch 0 [0/%d]\n", myrank, maxiter);
    test(model, device, dataloader, dataset_size, "train set");
  }
  
  // debug gradient history (will output gradient history to file)
  int *which_worker = NULL;
  int *num_grad_this_wait = NULL;
  int num_wait = 0;
  if (debug_grad) {
    which_worker = new int [maxiter+nranks];
    num_grad_this_wait = new int [maxiter+nranks];
  }

  // debug gradient sparsity (will output information to file)
  float *rel_msg_len = NULL;
  if (use_sparse && debug_sparse) {
    rel_msg_len = new float [maxiter+nranks];
  }
  
  // prepare for while loop
  int num_active_workers = nranks - 1;
  int counter = 0;
  int nreceived;
  int *idx_of_received = NULL;
  idx_of_received = new int [nranks];
  MPI_Status *status_of_received = NULL;
  status_of_received = new MPI_Status [nranks];

  // while loop
  while (num_active_workers > 0) {

    // wait for new g
    if (MPI_Waitsome(nranks, recv_request, &nreceived, idx_of_received,
                     status_of_received) != MPI_SUCCESS) {
      printf("master %d: Error in MPI_Waitsome!", myrank); exit(1);
    }

    // debug timing
    if (debug_time) {
      time2 = MPI_Wtime();
      elapse[elapse_count++] = time2 - time1;
      time1 = time2;
    }

    // debug communication
    if (debug_comm) {
      printf("master %d: received %d new g from ranks [ ", myrank, nreceived);
      for (int i = 0; i < nreceived; i++) {
        printf("%d ", idx_of_received[i]);
      }
      printf("]\n");
      if (nreceived <= 0) {
        nreceived = 0;
      }
    }
    
    // debug gradient history
    if (debug_grad) {
      num_grad_this_wait[num_wait++] = nreceived;
      for (int i = 0; i < nreceived; i++) {
        which_worker[counter+i] = idx_of_received[i];
      }
    }

    // debug gradient sparsity
    if (use_sparse && debug_sparse) {
      for (int i = 0; i < nreceived; i++) {
        rel_msg_len[counter+i] = gg[idx_of_received[i]][0] / N;
      }
    }

    // compute new w based on received g
    for (int i = 0; i < nreceived; i++) {

      // compute new w based on received g
      optimizer.step(gg[idx_of_received[i]], w);
      ++counter; // bookkeep the number of digested g in total

      // print training set loss and accuracy (time consuming)
      if (check_progress && counter % num_iter_per_epoch == 0) {
        optimizer.unpack_w(w); // float* -> tensor
        int epoch_number = counter / num_iter_per_epoch;
        printf("process %d: Train epoch %d [%d/%d]\n",
               myrank, epoch_number, counter, maxiter);
        test(model, device, dataloader, dataset_size, "train set");
      }
    }
    
    // debug communication
    if (debug_comm) {
      printf("master %d: total number of digested g = %d\n", myrank, counter);
    }
    
    // debug timing
    if (debug_time) {
      time2 = MPI_Wtime();
      elapse[elapse_count++] = time2 - time1;
      time1 = time2;
    }
    
    // send new w to select workers and post new nonblocking receives
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < nreceived; i++) {
      if (counter < maxiter) { // if not done

        // send new w to worker
        if (MPI_Send(w, N, MPI_FLOAT, idx_of_received[i], NOT_DONE,
                     MPI_COMM_WORLD) != MPI_SUCCESS) {
          printf("master %d: Error in MPI_Send!", myrank); exit(1);
        }

        // post new nonblocking receive from worker
        if (MPI_Irecv(gg[idx_of_received[i]], buf_sz, MPI_FLOAT,
                      idx_of_received[i], MPI_ANY_TAG, MPI_COMM_WORLD,
                      &recv_request[idx_of_received[i]]) != MPI_SUCCESS) {
          printf("master %d: Error in MPI_Irecv!", myrank); exit(1);
        }
      }
      else { // if done

        // send new w to worker. also signal termination
        if (MPI_Send(w, N, MPI_FLOAT, idx_of_received[i], DONE,
                     MPI_COMM_WORLD) != MPI_SUCCESS) {
          printf("master %d: Error in MPI_Send!", myrank); exit(1);
        }

        // this worker is done
        num_active_workers--;
      }
    }

    // debug timing
    if (debug_time) {
      time2 = MPI_Wtime();
      elapse[elapse_count++] = time2 - time1;
      time1 = time2;
    }
  }

  // copy final w in the communication buffer to model (float* -> tensor)
  optimizer.unpack_w(w);

  // debug timing: output timing information to file
  FILE *fp = NULL;
  if (debug_time) {
    fp = fopen(debug_time_outfile, "w");
    for (int i = 0; i < elapse_count; i++) {
      fprintf(fp, "%d ", elapse[i]);
      if ((i+1)%3 == 0) {
        fprintf(fp, "\n");
      }
    }
    fclose(fp);
  }
  
  // debug gradient history: output gradient history to file
  if (debug_grad) {
    fp = fopen(debug_grad_outfile, "w");
    int *which_worker_ptr = which_worker;
    for (int i = 0; i < num_wait; i++) {
      for (int j = 0; j < num_grad_this_wait[i]; j++) {
        fprintf(fp, "%d ", *which_worker_ptr++);
      }
      fprintf(fp, "\n");
    }
    fclose(fp);
  }

  // debug gradient sparsity: output relative msg length to file
  if (use_sparse && debug_sparse) {
    fp = fopen(debug_sparse_outfile, "w");
    for (int i = 0; i < counter; i++) {
      fprintf(fp, "%f\n", rel_msg_len[i]);
    }
    fclose(fp);
  }
  
  // clean up
  if (debug_time) {
    delete [] elapse;
  }
  if (debug_grad) {
    delete [] which_worker;
    delete [] num_grad_this_wait;
  }
  if (use_sparse && debug_sparse) {
    delete [] rel_msg_len;
  }
  delete [] recv_request;
  delete [] recv_count;
  delete [] idx_of_received;
  delete [] status_of_received;
  
  for (int i = 0; i < nranks; i++) {
    delete [] gg[i];
  }
  delete [] gg;
  delete [] w;
}

//-----------------------------------------------------------------------------

template <typename Net, typename DataLoader>
void train_worker(Net& model,
                  torch::Device device,
                  DataLoader& dataloader,
                  size_t dataset_size,
                  APAM<Net>& optimizer,
                  bool use_sparse,
                  bool debug_comm,
                  bool check_progress) {

  // mpi context
  MPI_Status status;
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  // allocate storage
  int N = optimizer.get_num_param();
  float *g = NULL;   // gradient
  if (!use_sparse) {
    g = new float [N];
  }
  else { // for sparse gradient, more space needed in worst case
    g = new float [2*N];
  }
  float *w = NULL;   // model parameter
  w = new float [N];

  // master and workers should use the same initialization
  MPI_Bcast(w, N, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
  // optimizer.unpack_w(w); // no need; done in a few lines later
  
  // set the mode to train (affects dropout, batchnorm, etc)
  model.train();

  // while loop
  while (1) {

    // compute new g based on the current w
    for (auto& batch : dataloader) {

      // move data to device
      auto data = batch.data.to(device), target = batch.target.to(device);
      
      // use w in the communication buffer to set model (float* -> tensor)
      optimizer.unpack_w(w);
       
      // reset gradient
      optimizer.zero_grad();

      // forward
      auto output = model.forward(data);
      auto loss = torch::nll_loss(output, target);

      // backward (compute gradient)
      loss.backward();

      // extract gradient to communication buffer g (tensor -> float*)
      optimizer.pack_g(g);
      
      // ensure that the loop is iterated only once, because we need
      // only one random batch
      break;
    }

    // send new g to master
    int send_count = use_sparse ? (int)g[0] : N;
    if (MPI_Send(g, send_count, MPI_FLOAT, ROOT, NOT_DONE, MPI_COMM_WORLD)
        != MPI_SUCCESS) {
      printf("worker %d: Error in MPI_Send!", myrank); exit(1);
    }

    // receive new w from master
    if (MPI_Recv(w, N, MPI_FLOAT, ROOT, MPI_ANY_TAG, MPI_COMM_WORLD, &status)
        != MPI_SUCCESS) {
      printf("worker %d: Error in MPI_Recv!", myrank); exit(1);
    }

    // debug communication
    if (debug_comm) {
      printf("worker %d: received w from master. "
             "status.MPI_SOURCE = %d, status.MPI_TAG = %d\n",
             myrank, status.MPI_SOURCE, status.MPI_TAG);
    }
    
    // terminate iteration
    if (status.MPI_TAG == DONE) {
      break;
    }
  }

  // clean up
  delete [] g;
  delete [] w;
}

//-----------------------------------------------------------------------------

template <typename Net, typename DataLoader>
void test(Net& model,
          torch::Device device,
          DataLoader& dataloader,
          size_t dataset_size,
          std::string dataset_name) {
  
  int myrank, nranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  
  torch::NoGradGuard no_grad; // no gradient calculation needed
  model.eval(); // set the mode to eval (affects dropout, batchnorm, etc)
  double loss = 0;
  int64_t correct = 0;
  for (auto& batch : dataloader) {
    auto data = batch.data.to(device), target = batch.target.to(device);
    auto output = model.forward(data);
    loss += torch::nll_loss(output, target).template item<float>()
      * batch.data.size(0);
    auto pred = output.argmax(1);
    correct += pred.eq(target).sum().template item<int64_t>();
  }
  loss /= dataset_size;
  double accuracy = 100. * correct / dataset_size;
  printf("process %d: On %s: Average loss: %g Accuracy: %d/%d (%g%%)\n",
         myrank, dataset_name.c_str(), loss, (int)correct, (int)dataset_size,
         accuracy);
}

#endif

