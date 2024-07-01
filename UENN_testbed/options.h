#pragma once

#include <torch/torch.h>
#include <string>

// optimizers
#define OPTIMIZER_SGD       0101
#define OPTIMIZER_ADAM      0102

// loss functions used in ENN
#define ENN_LOSS_DIGAMMA    0301
#define ENN_LOSS_LOG        0302
#define ENN_LOSS_MSE        0303

// loss functions used in CNN
#define CNN_NLL_LOSS            0401 // negative log likelihood loss
#define CNN_CROSS_ENTROPY_LOSS  0402 // cross entropy loss

const std::string Dataset_MNIST = "mnist";
const std::string Dataset_FashionMNIST = "FashionMNIST";
const std::string MNIST_PreTrainedModel = "lenet_mnist_eval.pt";
const std::string ROOT = "D:\\WorkSpace2024\\Project_Pytorch\\TorchENN\\";
const std::string Path_Datasets = "datasets";
const std::string Path_PreTrainedModels = "pretrained";

struct Options {
	std::string dataset = Dataset_MNIST;
	std::string dataset_path = ROOT + Path_Datasets + "\\" + Dataset_MNIST;
	std::string result_path = ROOT + Path_Datasets + "\\results";
	std::string logfile_path;

	int64_t train_batch_size = 128/*default*/;
	int64_t test_batch_size = 128/*default*/;
	int n_epochs = 60;
	int n_classes = 10;
	int n_loginterval = 10;
	bool use_dropout = true;

	unsigned long optimizer = OPTIMIZER_ADAM;
	unsigned long lossfunctype = ENN_LOSS_DIGAMMA;

	//float gamma = 0.01;
	//float grad_clip = 1;
	//int d_iters = 2;
	//int e_iters = 1;
	//int batch_size = 256;
	//float lambda_term = 10;
	//float weight_decay = 0.0001;
	//int nz = 128;
	//int log_interval = 5;
	//bool use_augment = true;
	//bool use_validation = false;
	//bool use_pretrain = true;
	//bool use_G_grad = false;
	//std::string path;
	//int num_classes = 10;
	//int split = 10000;
	//int in_channels = 1;

	torch::DeviceType device = torch::kCPU;
};
