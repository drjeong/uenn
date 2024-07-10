/**
 * @file main.cpp
 *
 * @brief This is the main program source code.
 *
 *
 * DLL needs to be copied.
 * Release Mode
 * xcopy $(SolutionDir)Lib\libtorch-win-shared-with-deps-2.0.1+cu118\libtorch\lib\*.dll $(SolutionDir)$(Platform)\$(Configuration)\ /c /y
 *
 * Debug Mode
 * xcopy $(SolutionDir)Lib\libtorch-win-shared-with-deps-debug-2.0.1+cu118\libtorch\lib\*.dll $(SolutionDir)$(Platform)\$(Configuration)\ /c /y
 */


#include <torch/torch.h>

#include <iostream>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

#include "lenet.h"
#include "enn.h"
#include "options.h"

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
	const int64_t kNumberOfEpochs = 2;

	Options options;
	options.device = device_type;
	options.n_epochs = kNumberOfEpochs;

	// available option: OPTIMIZER_SGD | OPTIMIZER_ADAM
	options.optimizer = OPTIMIZER_ADAM;

	// ENN options: ENN_LOSS_DIGAMMA | ENN_LOSS_LOG | ENN_LOSS_MSE
	// CNN options: CNN_NLL_LOSS | CNN_CROSS_ENTROPY_LOSS
	options.lossfunctype = ENN_LOSS_DIGAMMA;

	CENN uenn(options);
	uenn.trainENN();

	uenn.testENN();

	return EXIT_SUCCESS;
}