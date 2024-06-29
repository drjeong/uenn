#include "pch.h"
#include "ENN.h"
#include "entropy.h"

#include <filesystem>   // supported in C++17 (and later) 
#include <fstream>
#include <iostream>

// Undefine max & min because it is defined in C++ headers
#undef max
#undef min

ENN::ENN()
{
    m_pModel = NULL;
	m_pOptimizer = NULL;
}

ENN::~ENN()
{
    if (m_pOptimizer) delete m_pOptimizer;
    m_pOptimizer = NULL;

	if (m_pModel) delete m_pModel;
    m_pModel = NULL;
}

void ENN::Init(Options& options)
{
    m_options = options;
    m_pModel = new LeNet(options.use_dropout);

	// Allocating device type
	switch (options.device) {
	case 0: m_device = torch::kCPU; break;
	case 1: m_device = torch::kCUDA; break;
	}

	// Validating if CUDA is available
	if (m_device == torch::kCUDA) {
		torch::DeviceType device_type;
		if (torch::cuda::is_available()) {
			std::cout << "CUDA available! Training on GPU." << std::endl;
			m_device = torch::kCUDA;
		}
		else {
			std::cout << "CUDA not available! Training on CPU." << std::endl;
			m_device = torch::kCPU;
		}
	}

    m_pModel->to(m_device);

	CreateLogFile();

	SetOptimizer(options.optimizer);
}

int ENN::CreateLogFile()
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

    // Create a log file name
    std::string logfilename = m_options.dataset + "_time_" + current_time_str + ".log";

    try {
        // Create directories if not exists
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
void ENN::SetOptimizer(unsigned long optimizer)
{
    switch (optimizer)
    {
    case OPTIMIZER_SGD: // using SGD optimizer
        m_pOptimizer = new torch::optim::SGD(
            m_pModel->parameters(),
            torch::optim::SGDOptions(0.01).momentum(0.5)
        );
        break;
    case OPTIMIZER_ADAM:    // using Adam optimizer
        m_pOptimizer = new torch::optim::Adam(
            m_pModel->parameters(),
            torch::optim::AdamOptions(/*lr*/2e-4).betas(std::make_tuple(0.5, 0.5)).weight_decay(0.005)
        );
        break;
    }
}

/**
 * Testing ENN
 * data_loader: dataset to be used in testing
 * lossfunc: defined loss function (all loss functions are defined in losses.h)
 *
 */
//template <typename DataLoader>
//void ENN::testingModel(DataLoader& data_loader,
//    torch::Tensor(*lossfunc)(const torch::Tensor&, const torch::Tensor&,
//        size_t, size_t, size_t, torch::Device))
//{
//    torch::NoGradGuard no_grad;
//	size_t num_classes = m_options.n_classes;
//	size_t test_dataset_size = data_loader.dataset().size();
//
//    m_pModel->eval();
//    double test_loss = 0;
//    int32_t corrects = 0;
//    torch::Tensor loss;
//
//    for (const auto& batch : data_loader)
//    {
//        auto data = batch.data.to(m_device);
//        auto targets = batch.target.to(m_device);
//
//        auto output = m_pModel->forward(data);
//
//        // Compute the indices of the maximum values along dimension 1
//        auto preds = output.argmax(1);
//
//        // Convert to One Hot Encoding
//        auto y = torch::one_hot(targets, num_classes).to(m_device);
//
//        loss = lossfunc(
//            output, y.to(torch::kFloat), 10, num_classes, /*annealing_step*/10, m_device
//        );
//
//        // Compute element-wise equality between 'preds' and 'labels'
//        torch::Tensor equality = torch::eq(preds, targets).to(torch::kFloat);
//
//        // Reshape the tensor to (-1, 1)
//        auto match = equality.view({ -1, 1 });
//
//        auto acc = torch::mean(match);
//        auto evidence = relu_evidence(output);
//        auto alpha = evidence + 1;
//        auto u = (int)num_classes / torch::sum(alpha, /*dim=*/1, /*keepdim=*/true);
//
//        auto total_evidence = torch::sum(evidence, 1, true);
//        auto mean_evidence = torch::mean(total_evidence);
//        auto mean_evidence_succ = torch::sum(total_evidence * match)
//            / (torch::sum(match + 1e-20));
//        auto mean_evidence_fail = torch::sum(total_evidence * (1 - match))
//            / (torch::sum(torch::abs(1 - match)) + 1e-20);
//
//        auto pred = output.argmax(1);
//        corrects += pred.eq(targets).sum().template item<int64_t>();
//    }
//
//    test_loss /= test_dataset_size;
//    std::printf(
//        "\nTest set: Average loss: %.4f | Accuracy: %.3f\n",
//        test_loss,
//        static_cast<double>(corrects) / test_dataset_size);
//}
