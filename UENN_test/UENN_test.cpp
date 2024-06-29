// UENN_test.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <eigen-3.4.0/Dense>
#include "UENNLib.h"

#define CPU 0
#define CUDA 1

int main()
{
	Options option;
	option.device = CUDA;
	option.n_epochs = 5;	// number of epochs to train

	// OPTIMIZER_SGD | OPTIMIZER_ADAM
	option.optimizer = OPTIMIZER_ADAM;

	// ENN - ENN_LOSS_DIGAMMA | ENN_LOSS_LOG | ENN_LOSS_MSE
	option.lossfunctype = ENN_LOSS_DIGAMMA;

	const std::string DATA_PATH = "D:\\WorkSpace2024\\Project_Pytorch\\TorchENN\\datasets";

	option.dataset_path = DATA_PATH + "\\mnist";
	option.result_path = DATA_PATH + "\\results";

	// Generate a random Eigen matrix
	Eigen::MatrixXd matrix1;

	fnUENNLib(option, matrix1);
}