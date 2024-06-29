// UENNLib.cpp : Defines the exported functions for the DLL.
//

#include "pch.h"
#include "framework.h"
#include "UENNLib.h"
#include "ENN.h"

ENN enn;
std::unique_ptr<torch::data::datasets::MNIST> train_dataset;
std::unique_ptr< torch::data::datasets::MNIST> test_dataset;

int LoadMNISTDataset(Options option)
{
	try {
		// Attempt to create m_train_dataset
		train_dataset = std::make_unique<torch::data::datasets::MNIST>(option.dataset_path, torch::data::datasets::MNIST::Mode::kTrain);
	}
	catch (const std::exception& e) {
		std::cerr << "Error initializing m_train_dataset: " << e.what() << std::endl;
		// Handle the error appropriately
	}

	try {
		// Attempt to create m_train_dataset
		test_dataset = std::make_unique<torch::data::datasets::MNIST>(option.dataset_path, torch::data::datasets::MNIST::Mode::kTest);
	}
	catch (const std::exception& e) {
		std::cerr << "Error initializing m_test_dataset: " << e.what() << std::endl;
		// Handle the error appropriately
		return -1;
	}

	option.train_dataset_size = train_dataset->size().value();
	option.test_dataset_size = test_dataset->size().value();

	return 0;
}

// ENN MNST Training
UENNLIB_API int fnUENN_MNIST_Train(Options option)
{
	if (LoadMNISTDataset(option) < 0)
		return -1; // data loading error

	enn.Init(option);

	// 0.1307, 0.3081 are the mean and std deviation of the MNIST dataset.
	auto train_dataset_norm = train_dataset
		->map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
		.map(torch::data::transforms::Stack<>());

	auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
			std::move(train_dataset_norm), option.train_batch_size);
	
	// Determine which loss function to use beforehand
	auto chosenLossFunc = lossFunc(option.lossfunctype);

	enn.trainingModel(*train_loader, train_dataset->size().value(), chosenLossFunc);


    return 0;
}


// ENN MNST Training
UENNLIB_API int fnUENN_MNIST_Test(Options option, Eigen::MatrixXd& mat1)
{
	// 0.1307, 0.3081 are the mean and std deviation of the MNIST dataset.
	auto train_dataset_norm = train_dataset
		->map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
		.map(torch::data::transforms::Stack<>());
	auto test_dataset_norm = test_dataset
		->map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
		.map(torch::data::transforms::Stack<>());

	auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
		std::move(train_dataset_norm), option.train_batch_size);
	auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
		std::move(test_dataset_norm), option.test_batch_size);

	// Determine which loss function to use beforehand
	auto chosenLossFunc = lossFunc(option.lossfunctype);

	enn.evaluatingModel(*train_loader, train_dataset->size().value(), chosenLossFunc);

	enn.evaluatingModel(*test_loader, test_dataset->size().value(), chosenLossFunc);
}