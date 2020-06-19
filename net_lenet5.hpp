#ifndef _NET_LENET5_
#define _NET_LENET5_

#include <torch/torch.h>

struct LeNet5 : torch::nn::Module {
  torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
  torch::nn::MaxPool2d pool1{nullptr}, pool2{nullptr};
  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
  
  LeNet5() {
    conv1 = register_module("conv1", torch::nn::Conv2d(1, 6, 5));
    conv2 = register_module("conv2", torch::nn::Conv2d(6, 16, 5));
    pool1 = register_module("pool1", torch::nn::MaxPool2d(2));
    pool2 = register_module("pool2", torch::nn::MaxPool2d(2));
    fc1 = register_module("fc1", torch::nn::Linear(256, 120));
    fc2 = register_module("fc2", torch::nn::Linear(120, 84));
    fc3 = register_module("fc3", torch::nn::Linear(84, 10));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = conv1->forward(x);
    x = torch::relu(x);
    x = pool1->forward(x);
    x = conv2->forward(x);
    x = torch::relu(x);
    x = pool2->forward(x);
    x = torch::flatten(x, 1);
    x = fc1->forward(x);
    x = torch::relu(x);
    x = fc2->forward(x);
    x = torch::relu(x);
    x = fc3->forward(x);
    x = torch::log_softmax(x, 1);
    return x;
  }
};

#endif
