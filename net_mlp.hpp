#ifndef _NET_MLP_
#define _NET_MLP_

#include <torch/torch.h>

struct MLP : torch::nn::Module {
  torch::nn::Linear fc1{nullptr}, fc2{nullptr};
  
  MLP() {
    fc1 = register_module("fc1", torch::nn::Linear(784, 390));
    fc2 = register_module("fc2", torch::nn::Linear(390, 10));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::flatten(x, 1);
    x = fc1->forward(x);
    x = torch::relu(x);
    x = fc2->forward(x);
    x = torch::log_softmax(x, 1);
    return x;
  }
};

#endif
