/**
 * @file main.cpp
 *
 * @brief This is the main program source code.
 *
 */


#include <torch/torch.h>

#include <iostream>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

#include "lenet.h"
//#include "losses.h"
#include "cnn.h"
#include "options.h"

// Where to find the MNIST dataset.
//const char* kDataRoot = "./mnist";

int main()
{
    torch::manual_seed(1);  // Sets the seed for generating random numbers

    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    }
    else {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }

#ifdef _DEBUG
    // using CPU mode when debugging
    device_type = torch::kCPU;
#endif
    
	// The number of epochs to train.
	const int64_t kNumberOfEpochs = 10;

	Options options;
    options.device = device_type;
    options.n_epochs = kNumberOfEpochs;

    // OPTIMIZER_SGD | OPTIMIZER_ADAM
    options.optimizer = OPTIMIZER_ADAM;

    // ENN - ENN_LOSS_DIGAMMA | ENN_LOSS_LOG | ENN_LOSS_MSE
	// CNN - CNN_NLL_LOSS | CNN_CROSS_ENTROPY_LOSS
	options.lossfunctype = ENN_LOSS_DIGAMMA;

    CNNMNIST cnn(options);
    cnn.trainENN();

    cnn.testENN();

    return EXIT_SUCCESS;
}