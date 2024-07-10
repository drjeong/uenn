// UENNLib.cpp : Defines the exported functions for the DLL.
//

#include "pch.h"
#include "framework.h"
#include "UENNLib.h"
#include "ENN.h"

ENN enn;

/// <summary>
/// Loaidng MNIST Training Dataset
/// </summary>
/// <param name="option"></param>
/// <returns></returns>
std::unique_ptr<torch::data::datasets::MNIST> LoadMNIST_TrainDataset(const Options& option) {
	std::unique_ptr<torch::data::datasets::MNIST> train_dataset;

	try {
		// Attempt to create train_dataset
		train_dataset = std::make_unique<torch::data::datasets::MNIST>(option.dataset_path, torch::data::datasets::MNIST::Mode::kTrain);
		std::cerr << "Train dataset is loaded." << std::endl;
	}
	catch (const std::exception& e) {
		std::cerr << "Error initializing train dataset: " << e.what() << std::endl;
		throw std::runtime_error("Failed to initialize the train dataset");
	}

	return std::move(train_dataset);
}

/// <summary>
/// Loading MNIST Testing Dataset
/// </summary>
/// <param name="option"></param>
/// <returns></returns>
std::unique_ptr<torch::data::datasets::MNIST> LoadMNIST_TestDataset(const Options& option) {
	std::unique_ptr<torch::data::datasets::MNIST> test_dataset;

	try {
		// Attempt to create test_dataset
		test_dataset = std::make_unique<torch::data::datasets::MNIST>(option.dataset_path, torch::data::datasets::MNIST::Mode::kTest);
		std::cerr << "Test dataset is loaded." << std::endl;
	}
	catch (const std::exception& e) {
		std::cerr << "Error initializing test dataset: " << e.what() << std::endl;
		throw std::runtime_error("Failed to initialize the test dataset");
	}

	return std::move(test_dataset);
}


/// <summary>
/// Train UENN model with MNIST training dataset
/// </summary>
/// <param name="option"></param>
/// <returns></returns>
UENNLIB_API int fnUENN_MNIST_Train(Options option)
{
	try {
		auto train_dataset = LoadMNIST_TrainDataset(option);

		// set data size to options
		option.train_dataset_size = train_dataset->size().value();

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

	}
	catch (const std::exception& e) {
		std::cerr << "Failed to load datasets: " << e.what() << std::endl;
		// Handle error appropriately
		return -1;
	}

    return 0;
}


/// <summary>
/// Testing UENN model with Training dataset
/// </summary>
/// <param name="option"></param>
/// <returns></returns>
UENNLIB_API int fnUENN_MNIST_Test_w_TrainData(Options option)
{
	try {
		auto train_dataset = LoadMNIST_TrainDataset(option);

		// set data size to options
		option.train_dataset_size = train_dataset->size().value();

		// 0.1307, 0.3081 are the mean and std deviation of the MNIST dataset.
		auto train_dataset_norm = train_dataset
			->map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
			.map(torch::data::transforms::Stack<>());

		auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
			std::move(train_dataset_norm), option.train_batch_size);

		// Determine which loss function to use beforehand
		auto chosenLossFunc = lossFunc(option.lossfunctype);

		//enn.testingModel(*train_loader, train_dataset->size().value(), chosenLossFunc);
	}
	catch (const std::exception& e) {
		std::cerr << "Failed to load datasets: " << e.what() << std::endl;
		// Handle error appropriately
		return -1;
	}

	return 0;
}


/// <summary>
/// Test UENN model with MNIST Testing dataset
/// </summary>
/// <param name="option"></param>
/// <param name="mat_belief"></param>
/// <param name="mat_evidence"></param>
/// <param name="mat_strength"></param>
/// <param name="mat_uncertainty_mass"></param>
/// <param name="mat_belief_ent"></param>
/// <param name="mat_belief_tot_disagreement"></param>
/// <param name="mat_expected_probability_ent"></param>
/// <param name="mat_dissonance"></param>
/// <param name="mat_vacuity"></param>
/// <param name="mat_labels"></param>
/// <param name="mat_matches"></param>
/// <returns></returns>
UENNLIB_API int fnUENN_MNIST_Test_w_TestData(Options option, 
	Eigen::MatrixXd& mat_belief, Eigen::MatrixXd& mat_evidence, Eigen::MatrixXd& mat_strength, 
	Eigen::MatrixXd& mat_uncertainty_mass, Eigen::MatrixXd& mat_belief_ent,
	Eigen::MatrixXd& mat_belief_tot_disagreement, Eigen::MatrixXd& mat_expected_probability_ent,
	Eigen::MatrixXd& mat_dissonance, Eigen::MatrixXd& mat_vacuity,
	Eigen::MatrixXd& mat_labels, Eigen::MatrixXd& mat_matches)
{

	try {
		auto test_dataset = LoadMNIST_TestDataset(option);

		// set data size to options
		option.test_dataset_size = test_dataset->size().value();

		// 0.1307, 0.3081 are the mean and std deviation of the MNIST dataset.
		auto test_dataset_norm = test_dataset
			->map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
			.map(torch::data::transforms::Stack<>());

		auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
			std::move(test_dataset_norm), option.test_batch_size);

		// Determine which loss function to use beforehand
		auto chosenLossFunc = lossFunc(option.lossfunctype);

		enn.testingModel(*test_loader, test_dataset->size().value(), chosenLossFunc,
			mat_belief, mat_evidence, mat_strength,
			mat_uncertainty_mass, mat_belief_ent,
			mat_belief_tot_disagreement, mat_expected_probability_ent,
			mat_dissonance, mat_vacuity, mat_labels, mat_matches);

	}
	catch (const std::exception& e) {
		std::cerr << "Failed to load datasets: " << e.what() << std::endl;
		// Handle error appropriately
		return -1;
	}

	return 0;
}