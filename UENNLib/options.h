#pragma once

#include <string>

// optimizers
#define OPTIMIZER_SGD       0101
#define OPTIMIZER_ADAM      0102

// loss functions used in ENN
enum LossFunctionTypeEnum {
	ENN_LOSS_DIGAMMA,  // Assume associated with CrossEntropyBayesRisk
	ENN_LOSS_LOG,      // Assume associated with MaximumLikelihoodLoss
	ENN_LOSS_MSE       // Assume associated with SquaredErrorBayesRisk
};

// loss functions used in CNN
#define CNN_NLL_LOSS            0401 // negative log likelihood loss
#define CNN_CROSS_ENTROPY_LOSS  0402 // cross entropy loss

const std::string Dataset_MNIST = "mnist";
const std::string Dataset_FashionMNIST = "FashionMNIST";
const std::string MNIST_PreTrainedModel = "lenet_mnist_eval.pt";
const std::string Path_Datasets = "datasets";
const std::string Path_PreTrainedModels = "pretrained";

struct Options {
	std::string dataset = Dataset_MNIST;
	std::string dataset_path = "";
	std::string result_path = "";
	std::string logfile_path;

	int64_t train_batch_size = 128/*default*/;
	int64_t test_batch_size = 128/*default*/;

	int64_t train_dataset_size;
	int64_t test_dataset_size;

	int n_epochs = 10;
	int n_classes = 10;
	int n_loginterval = 10;
	bool use_dropout = true;

	unsigned long optimizer = OPTIMIZER_ADAM;
	unsigned long lossfunctype = ENN_LOSS_DIGAMMA;

	int8_t device = 0; // default: torch::kCPU;
};
