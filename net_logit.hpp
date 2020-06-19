#ifndef _NET_LOGIT_
#define _NET_LOGIT_

#include <torch/torch.h>

struct LOGIT : torch::nn::Module {
  torch::nn::Linear fc{nullptr};
  
  LOGIT() {
    fc = register_module("fc", torch::nn::Linear(784, 10));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::flatten(x, 1);
    x = fc->forward(x);
    x = torch::log_softmax(x, 1);
    return x;
  }
};

#endif
