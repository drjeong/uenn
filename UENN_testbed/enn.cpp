#include "enn.h"
#include "losses.h"
#include "entropy.h"

#include <filesystem>   // supported in C++17 (and later) 
#include <fstream>
#include <iostream>

CENN::CENN(Options& options) :
	m_options(options),
	m_Model(options.use_dropout)
{
	m_pOptimizer = NULL;

	try {
		// Attempt to create m_train_dataset
		m_train_dataset = std::make_unique<torch::data::datasets::MNIST>(options.dataset_path, torch::data::datasets::MNIST::Mode::kTrain);
	}
	catch (const std::exception& e) {
		std::cerr << "Error initializing m_train_dataset: " << e.what() << std::endl;
		// Handle the error appropriately
	}

	try {
		// Attempt to create m_train_dataset
		m_test_dataset = std::make_unique<torch::data::datasets::MNIST>(options.dataset_path, torch::data::datasets::MNIST::Mode::kTest);
	}
	catch (const std::exception& e) {
		std::cerr << "Error initializing m_test_dataset: " << e.what() << std::endl;
		// Handle the error appropriately
	}

	m_Model.to(options.device);

	CreateLogFile();

	SetOptimizer(options.optimizer);
}

CENN::~CENN()
{
	if (m_pOptimizer) delete m_pOptimizer;
	m_pOptimizer = NULL;
}

int CENN::CreateLogFile()
{
	// Get the current time as a time_t object
	time_t rawTime;
	time(&rawTime);

	// Initialize a tm structure to store the local time
	struct tm time_info;
	errno_t err = localtime_s(&time_info, &rawTime);

	char time_buffer[80];
	std::strftime(time_buffer, 80, "%Y-%m-%d_%H-%M-%S", &time_info);
	std::string current_time_str(time_buffer);

	std::string logfilename = m_options.dataset + "_time_" + current_time_str + ".log";

	try {
		std::filesystem::create_directories(m_options.result_path);
	}
	catch (std::exception& e) {
		std::cerr << "Creation of the directory " << m_options.result_path << " failed: " << e.what() << std::endl;
		return -1;
	}

	m_options.logfile_path = m_options.result_path + "\\" + logfilename;

	return 0;
}

/**
 * Define Optimizer
 * Supported optimizer: SGD, Adam
 */
void CENN::SetOptimizer(unsigned long optimizer)
{
	switch (optimizer)
	{
	case OPTIMIZER_SGD: // using SGD optimizer
		m_pOptimizer = new torch::optim::SGD(
			m_Model.parameters(),
			torch::optim::SGDOptions(0.01).momentum(0.5)
		);
		break;
	case OPTIMIZER_ADAM:	// using Adam optimizer
		m_pOptimizer = new torch::optim::Adam(
			m_Model.parameters(),
			torch::optim::AdamOptions(/*lr*/2e-4).betas(std::make_tuple(0.5, 0.5)).weight_decay(0.005)
		);
		break;
	}
}

void CENN::loadMNIST(const char* kDataRoot)
{
	//// 0.1307, 0.3081 are the mean and std deviation of the MNIST dataset.
	//auto train_dataset = torch::data::datasets::MNIST(
	//	kDataRoot, torch::data::datasets::MNIST::Mode::kTrain)
	//	.map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
	//	.map(torch::data::transforms::Stack<>());
	//m_train_dataset_size = train_dataset.size().value();

	//m_train_dataset = std::move(train_dataset);

	//auto test_dataset = torch::data::datasets::MNIST(
	//	kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
	//	.map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
	//	.map(torch::data::transforms::Stack<>());
	//m_test_dataset_size = test_dataset.size().value();
	//auto test_loader =
	//	torch::data::make_data_loader(std::move(test_dataset), m_kTestBatchSize);
	//m_test_loader = &test_loader;

}

/**
 * Training CNN
 * epochs: total number of epochs
 * lossfunctype: loss function type
 */
void CENN::trainCNN()
{
	// 0.1307, 0.3081 are the mean and std deviation of the MNIST dataset.
	auto train_dataset = m_train_dataset
		->map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
		.map(torch::data::transforms::Stack<>());

	auto train_loader =
		torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
			std::move(train_dataset), m_options.train_batch_size);

	for (size_t epoch = 1; epoch <= m_options.n_epochs; ++epoch) {
		trainCNN(epoch, *train_loader, m_options.lossfunctype);
	}
}

/**
 * Training ENN
 * epochs: total number of epochs
 * lossfunctype: loss function type (available types are listed in cnn.h)
 *
 */
void CENN::trainENN()
{
	// 0.1307, 0.3081 are the mean and std deviation of the MNIST dataset.
	auto train_dataset = m_train_dataset
		->map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
		.map(torch::data::transforms::Stack<>());

	auto train_loader =
		torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
			std::move(train_dataset), m_options.train_batch_size);

	for (size_t epoch = 1; epoch <= m_options.n_epochs; ++epoch) {
		switch (m_options.lossfunctype)
		{
		case ENN_LOSS_DIGAMMA:
			trainENN(epoch, *train_loader, CrossEntropyBayesRisk);
			break;
		case ENN_LOSS_LOG:
			trainENN(epoch, *train_loader, MaximumLikelihoodLoss);
			break;
		case ENN_LOSS_MSE:
			trainENN(epoch, *train_loader, SquaredErrorBayesRisk);
			break;
		}
		//break;
	}
}

/**
 * Training CNN with data loader
 * epoch: current epoch iterator
 * data_loader: dataset
 * lossfunctype: loss function type (available types are listed in cnn.h)
 *
 */
template <typename DataLoader>
void CENN::trainCNN(size_t epoch, DataLoader& data_loader, unsigned long lossfunctype)
{
	size_t num_classes = m_options.n_classes;
	size_t train_dataset_size = m_train_dataset->size().value();

	m_Model.train();
	size_t batch_idx = 0;
	int32_t corrects = 0;

	torch::Tensor loss;
	for (auto& batch : data_loader)
	{
		auto data = batch.data.to(m_options.device);
		auto targets = batch.target.to(m_options.device);

		// zero the parameter gradients
		m_pOptimizer->zero_grad();

		auto output = m_Model.forward(data);
		switch (lossfunctype)
		{
		case CNN_NLL_LOSS:
			loss = torch::nll_loss(output, targets);	//nll_loss: negative log likelihood loss
			break;
		case CNN_CROSS_ENTROPY_LOSS:
			loss = torch::cross_entropy_loss(output, targets);	//cross_entropy_loss: cross entropy loss
			break;
		}
		AT_ASSERT(!std::isnan(loss.template item<float>()));

		// Calculate the number of correct predictions and add it to 'corrects'
		auto pred = output.argmax(1);
		corrects += pred.eq(targets).sum().template item<int64_t>();

		loss.backward();
		m_pOptimizer->step();

		if (batch_idx++ % m_options.n_loginterval == 0) {
			std::printf(
				"\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f",
				epoch,
				batch_idx * batch.data.size(0),
				train_dataset_size,
				loss.template item<float>());
		}
	}

	std::printf(
		" | Accuracy: %.3f\n",
		static_cast<double>(corrects) / train_dataset_size);
}


/**
 * Training ENN
 * epoch: current epoch
 * data_loader: dataset
 * lossfunc: user defined loss function
 *
 */
template <typename DataLoader>
void CENN::trainENN(size_t epoch, DataLoader& data_loader,
	torch::Tensor(*lossfunc)(const torch::Tensor&, const torch::Tensor&,
		size_t, size_t, size_t, torch::Device))
{
	size_t num_classes = m_options.n_classes;
	size_t train_dataset_size = m_train_dataset->size().value();

	m_Model.train();
	size_t batch_idx = 0;
	int64_t corrects = 0;

	float avg_loss = 0;

	std::vector<torch::Tensor> computed_belief;
	std::vector<torch::Tensor> computed_uncertainty_mass;
	std::vector<torch::Tensor> computed_belief_ent;
	std::vector<torch::Tensor> computed_belief_tot_disagreement;
	std::vector<torch::Tensor> computed_expected_probability_ent;
	std::vector<torch::Tensor> computed_u_succ;
	std::vector<torch::Tensor> computed_u_fail;
	std::vector<torch::Tensor> computed_prob_succ;
	std::vector<torch::Tensor> computed_prob_fail;
	std::vector<torch::Tensor> computed_belief_succ;
	std::vector<torch::Tensor> computed_belief_fail;
	std::vector<torch::Tensor> computed_dissonance;
	std::vector<torch::Tensor> computed_vacuity;

	int batch_count = 0;
	for (auto& batch : data_loader)
	{
		auto data = batch.data.to(m_options.device);
		auto targets = batch.target.to(m_options.device);

		// zero the parameter gradients
		m_pOptimizer->zero_grad();

		auto output = m_Model.forward(data);

		// Compute the indices of the maximum values along dimension 1
		auto preds = output.argmax(1);

		// Convert to One Hot Encoding
		auto y = torch::one_hot(targets, num_classes).to(m_options.device);

		auto loss = lossfunc(
			output, y.to(torch::kFloat), epoch, num_classes, /*annealing_step*/10, m_options.device
		);

		// Compute element-wise equality between 'preds' and 'labels'
		torch::Tensor equality = torch::eq(preds, targets).to(torch::kFloat);

		// Reshape the tensor to (-1, 1)
		auto match = equality.view({ -1, 1 });
		auto match_bool = match.to(torch::kBool);  // convert match to a boolean tensor

		auto acc = torch::mean(match);
		auto evidence = relu_evidence(output);
		//auto evidence = elu_evidence(output); // when using elu, output must be positive values.
		//printTensor(evidence);

		// alpha size: batch size x # of classes
		auto alpha = evidence + 1;
		//printTensor(alpha);

		// strength size: batch size x 1
		auto strength /*alpha_sum*/ = torch::sum(alpha, /*dim=*/1, /*keepdim=*/true);
		//printTensor(strength);

		// uncertainty_mass size: batch size x 1
		auto uncertainty_mass = static_cast<float>(num_classes) / strength;
		auto u_succ = uncertainty_mass.masked_select(match_bool);	// Track u for correct predictions
		auto u_fail = uncertainty_mass.masked_select(~match_bool);   // Track u for incorrect predictions

		// vacuity
		// Note: Vacuity might not be correct because of incorrect when using OOD samples. 
		auto vacuity = ((int)num_classes) / strength;	// Calculate vacuity
		//printTensor(vacuity);

		// expected_probability size: batch size x # of classes
		auto expected_probability = alpha / strength;

		// Expected probability of the selected (greedy highset probability) class
		torch::Tensor expected_probability_max;
		std::tie(expected_probability_max, std::ignore) = expected_probability.max(-1);

		auto prob_succ = expected_probability_max.masked_select(match_bool); // Track p_k for correct predictions
		auto prob_fail = expected_probability_max.masked_select(~match_bool); // Track p_k for incorrect predictions

		auto total_evidence = torch::sum(evidence, 1, true);
		auto mean_evidence = torch::mean(total_evidence);
		auto mean_evidence_succ = torch::sum(total_evidence * match) / (torch::sum(match + 1e-20));
		auto mean_evidence_fail = torch::sum(total_evidence * (1 - match)) / (torch::sum(torch::abs(1 - match)) + 1e-20);

		auto pred = output.argmax(1);
		corrects += pred.eq(targets).sum().template item<int64_t>();

		// belief mass
		auto belief = evidence / strength;

		// Expected belief mass of the selected (greedy highset probability) class
		torch::Tensor belief_max;
		std::tie(belief_max, std::ignore) = belief.max(-1);

		auto belief_succ = belief_max.masked_select(match_bool); // Track belief for correct predictions
		auto belief_fail = belief_max.masked_select(~match_bool); // Track belief for incorrect predictions

		auto dissonance = getDisn(alpha, evidence, strength, belief);
		//printTensor(dissonance);

		auto belief_ent = shannon_entropy(belief);
		auto belief_tot_disagreement = total_disagreement(belief);
		auto expected_probability_ent = shannon_entropy(expected_probability); // Function to calculate entropy

		//auto vec_targets = tensorToVector(targets);
		//auto vec_belief = tensorToVector(belief);

		// statistics
		//torch::Tensor correct_preds = preds == targets;
		//running_corrects += correct_preds.sum();

		loss.backward();
		m_pOptimizer->step();

		if (batch_idx++ % m_options.n_loginterval == 0) {
			std::cout << "\rTrain Epoch: " << epoch;
			std::cout << "[" << batch_idx * batch.data.size(0);
			std::cout << "/" << train_dataset_size << "] ";
			std::cout << "Loss: " << loss.template item<float>();
			std::cout << " | Avg belief: " << belief.mean().item().toFloat();
			std::cout << " | Accuracy: " << static_cast<float>(corrects) / train_dataset_size;
			std::cout << std::flush;
		}

		//printTensorSize(belief);

		batch_count++;

		computed_belief.emplace_back(std::move(belief));
		computed_uncertainty_mass.emplace_back(std::move(uncertainty_mass));
		computed_belief_ent.emplace_back(std::move(belief_ent));
		computed_belief_tot_disagreement.emplace_back(std::move(belief_tot_disagreement));
		computed_expected_probability_ent.emplace_back(std::move(expected_probability_ent));

		computed_u_succ.emplace_back(std::move(u_succ));
		computed_u_fail.emplace_back(std::move(u_fail));
		computed_prob_succ.emplace_back(std::move(prob_succ));
		computed_prob_fail.emplace_back(std::move(prob_fail));
		computed_belief_succ.emplace_back(std::move(belief_succ));
		computed_belief_fail.emplace_back(std::move(belief_fail));

		computed_dissonance.emplace_back(std::move(dissonance));
		computed_vacuity.emplace_back(std::move(vacuity));

		avg_loss += loss.template item<float>();

		//if (batch_count > 3) break; // for debugging purpose
	}
	//std::cout << std::endl;

	std::stringstream ss; // Create a stringstream object
	ss << "\rTrain Epoch: " << epoch;
	ss << "[" << train_dataset_size;
	ss << "/" << train_dataset_size << "] ";
	ss << "Loss: " << avg_loss / batch_count;
	ss << " | m(u_succ): " << getMeanValue(computed_u_succ);
	ss << " | m(u_fail): " << getMeanValue(computed_u_fail);
	ss << " | m(uncertainty): " << getMeanValue(computed_uncertainty_mass);   // 6000 x 1
	ss << " | m(prob_succ): " << getMeanValue(computed_prob_succ);
	ss << " | m(prob_fail): " << getMeanValue(computed_prob_fail);
	ss << " | m(belief): " << getMeanValue(computed_belief);   // 6000 x 10
	ss << " | m(belief_succ): " << getMeanValue(computed_belief_succ);
	ss << " | m(belief_fail): " << getMeanValue(computed_belief_fail);
	ss << " | m(exp_p_entropy): " << getMeanValue(computed_expected_probability_ent);   // 6000 x 10
	ss << " | m(dissonance): " << getMeanValue(computed_dissonance);   // 6000 x 1
	ss << " | m(vacuity): " << getMeanValue(computed_vacuity);   // 6000 x 1
	ss << " | m(belief entropy): " << getMeanValue(computed_belief_ent);   // 6000
	ss << " | m(belief total disagreement): " << getMeanValue(computed_belief_tot_disagreement);   // 6000
	ss << " | Accuracy: " << static_cast<double>(corrects) / train_dataset_size;

	outputToFileAndConsole(m_options.logfile_path, ss.str());

	//auto epoch_loss = running_loss / m_kTrainBatchSize;
	//auto epoch_acc = running_corrects.to(torch::kDouble) / m_kTrainBatchSize;

	// Calculate 'epoch_loss' and 'epoch_acc'
	//double epoch_loss = running_loss.item<double>() / static_cast<double>(m_kTrainBatchSize);
	//double epoch_acc = static_cast<double>(running_corrects.item<int64_t>()) / static_cast<double>(m_kTrainBatchSize);

	//std::printf(
	//	"{} loss: {:.4f} acc: {:.4f}",
	//	epoch_loss, epoch_acc
	//);

	//Eigen::MatrixXd mat_belief = convertTensorVecToEigen(computed_belief, train_dataset_size, num_classes);
}


/**
 * print ENN training data to console out
 *
 */
void CENN::printENNTrainingData(size_t& epoch, size_t batch_size,
	size_t train_dataset_size, size_t corrects, torch::Tensor& loss)
{
	std::stringstream ss; // Create a stringstream object

	ss << "\rTrain Epoch: " << epoch;
	ss << "[" << batch_size;
	ss << "/" << train_dataset_size << "] ";
	ss << "Loss: " << loss.template item<float>();
	//ss << " | Avg belief: " << belief.mean().item().toDouble();
	//ss << " | Avg uncertainty: " << uncertainty.mean().item().toDouble();
	//ss << " | Avg Shannon entropy: " << belief_ent.mean().item().toDouble();
	//ss << " | Avg total disagreement: " << belief_tot_disagreement.mean().item().toDouble();
	ss << " | Accuracy: " << static_cast<double>(corrects) / train_dataset_size;

	std::cout << ss.str() << std::flush;

	//outputToFileAndConsole(ss.str(), m_options.logfile_path);

	//if (end ==false) std::cout << std::flush;
	//else  std::cout << std::endl;
}


/**
 * Testing CNN with different loss function
 * lossfunctype:
 *
 */
void CENN::testCNN(unsigned long lossfunctype)
{
	// 0.1307, 0.3081 are the mean and std deviation of the MNIST dataset.
	auto test_dataset = m_test_dataset
		->map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
		.map(torch::data::transforms::Stack<>());

	auto test_loader =
		torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
			std::move(test_dataset), m_options.train_batch_size);

	testCNN(*test_loader, lossfunctype);
}


/**
 * Testing ENN with different loss function
 * lossfunctype: loss function type
 *
 */
void CENN::testENN()
{
	// 0.1307, 0.3081 are the mean and std deviation of the MNIST dataset.
	auto test_dataset = m_test_dataset
		->map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
		.map(torch::data::transforms::Stack<>());

	auto test_loader =
		torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
			std::move(test_dataset), m_options.train_batch_size);

	switch (m_options.lossfunctype)
	{
	case ENN_LOSS_DIGAMMA:
		testENN(*test_loader, CrossEntropyBayesRisk);
		break;
	case ENN_LOSS_LOG:
		testENN(*test_loader, MaximumLikelihoodLoss);
		break;
	case ENN_LOSS_MSE:
		testENN(*test_loader, SquaredErrorBayesRisk);
		break;
	}
}

/**
 * Testing CNN
 * data_loader: dataset to be used in testing
 * lossfunctype: loss function definition - CNN_NLL_LOSS, CNN_CROSS_ENTROPY_LOSS (defined in cnn.h)
 *
 */
template <typename DataLoader>
void CENN::testCNN(DataLoader& data_loader, unsigned long lossfunctype)
{
	torch::NoGradGuard no_grad;
	size_t test_dataset_size = m_test_dataset->size().value();
	size_t num_classes = m_options.n_classes;

	m_Model.eval();
	double test_loss = 0;
	int32_t corrects = 0;
	torch::Tensor loss;

	for (const auto& batch : data_loader)
	{
		auto data = batch.data.to(m_options.device);
		auto targets = batch.target.to(m_options.device);

		auto output = m_Model.forward(data);	// forward processing

		switch (lossfunctype)
		{
		case CNN_NLL_LOSS:
			test_loss += torch::nll_loss(
				output,
				targets,
				/*weight=*/{},
				torch::Reduction::Sum)
				.template item<float>();
			break;
		case CNN_CROSS_ENTROPY_LOSS:
			test_loss += torch::cross_entropy_loss(
				output,
				targets,
				/*weight=*/{},
				torch::Reduction::Sum)
				.template item<float>();
			break;
		}

		// Calculate the number of correct predictions and add it to 'corrects'
		auto pred = output.argmax(1);
		corrects += pred.eq(targets).sum().template item<int64_t>();

	}

	test_loss /= test_dataset_size;
	std::printf(
		"\nTest set: Average loss: %.4f | Accuracy: %.3f\n",
		test_loss,
		static_cast<double>(corrects) / test_dataset_size);
}

/**
 * Testing ENN
 * data_loader: dataset to be used in testing
 * lossfunc: defined loss function (all loss functions are defined in losses.h)
 *
 */
template <typename DataLoader>
void CENN::testENN(DataLoader& data_loader,
	torch::Tensor(*lossfunc)(const torch::Tensor&, const torch::Tensor&,
		size_t, size_t, size_t, torch::Device))
{
	torch::NoGradGuard no_grad;
	size_t test_dataset_size = m_test_dataset->size().value();
	size_t num_classes = m_options.n_classes;

	m_Model.eval();
	double test_loss = 0;
	int32_t corrects = 0;
	torch::Tensor loss;

	for (const auto& batch : data_loader)
	{
		auto data = batch.data.to(m_options.device);
		auto targets = batch.target.to(m_options.device);

		auto output = m_Model.forward(data);

		// Compute the indices of the maximum values along dimension 1
		auto preds = output.argmax(1);

		// Convert to One Hot Encoding
		auto y = torch::one_hot(targets, num_classes).to(m_options.device);

		loss = lossfunc(
			output, y.to(torch::kFloat), 10, num_classes, /*annealing_step*/10, m_options.device
		);

		// Compute element-wise equality between 'preds' and 'labels'
		torch::Tensor equality = torch::eq(preds, targets).to(torch::kFloat);

		// Reshape the tensor to (-1, 1)
		auto match = equality.view({ -1, 1 });

		auto acc = torch::mean(match);
		auto evidence = relu_evidence(output);
		auto alpha = evidence + 1;
		auto u = (int)num_classes / torch::sum(alpha, /*dim=*/1, /*keepdim=*/true);

		auto total_evidence = torch::sum(evidence, 1, true);
		auto mean_evidence = torch::mean(total_evidence);
		auto mean_evidence_succ = torch::sum(total_evidence * match)
			/ (torch::sum(match + 1e-20));
		auto mean_evidence_fail = torch::sum(total_evidence * (1 - match))
			/ (torch::sum(torch::abs(1 - match)) + 1e-20);

		auto pred = output.argmax(1);
		corrects += pred.eq(targets).sum().template item<int64_t>();
	}

	test_loss /= test_dataset_size;
	std::printf(
		"\nTest set: Average loss: %.4f | Accuracy: %.3f\n",
		test_loss,
		static_cast<double>(corrects) / test_dataset_size);
}
