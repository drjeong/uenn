#pragma once
#include <torch/torch.h>
#include <iostream>
#include <functional>

#include "lenet.h"
#include "losses.h"
#include "options.h"
#include "helper.h"
#include "entropy.h"

class ENN
{
private:

	torch::DeviceType m_device;
	torch::optim::Optimizer* m_pOptimizer;
	LeNet *m_pModel;

	// store all options
	Options m_options;

public:
    ENN();
	~ENN();

	void Init(Options& options);

	// Define Optimizer (default: SGD optimizer)
	void SetOptimizer(unsigned long optimizer = OPTIMIZER_SGD);

    /**
     * Evaluate Model
     * epoch: current epoch
     * data_loader: dataset
     * lossfunc: user defined loss function
     */
    template <typename DataLoader>
    void evaluatingModel(DataLoader& data_loader, size_t data_size,
		std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&, size_t, size_t, size_t, torch::Device)> lossfunc)
    {
		size_t num_classes = m_options.n_classes;

		// Set the model to evaluation mode
        m_pModel->eval();

		// Turn off gradient tracking
		torch::NoGradGuard no_grad;

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
            auto data = batch.data.to(m_device);
            auto targets = batch.target.to(m_device);

            auto output = m_pModel->forward(data);

            // Compute the indices of the maximum values along dimension 1
            auto preds = output.argmax(1);

            // Convert to One Hot Encoding
            auto y = torch::one_hot(targets, num_classes).to(m_device);

            auto loss = lossfunc(
                output, y.to(torch::kFloat), /*annealing_step*/10, num_classes, /*annealing_step*/10, m_device
            );

            // Compute element-wise equality between 'preds' and 'labels'
            torch::Tensor equality = torch::eq(preds, targets).to(torch::kFloat);

            // Reshape the tensor to (-1, 1)
            auto match = equality.view({ -1, 1 });
            auto match_bool = match.to(torch::kBool);  // convert match to a boolean tensor

            auto acc = torch::mean(match);
            auto evidence = relu_evidence(output);
            //auto evidence = elu_evidence(output);

            // alpha size: batch size x # of classes
            auto alpha = evidence + 1;

            // strength size: batch size x 1
            auto strength /*alpha_sum*/ = torch::sum(alpha, /*dim=*/1, /*keepdim=*/true);

            // uncertainty_mass size: batch size x 1
            auto uncertainty_mass = static_cast<float>(num_classes) / strength;
            auto u_succ = uncertainty_mass.masked_select(match_bool);    // Track u for correct predictions
            auto u_fail = uncertainty_mass.masked_select(~match_bool);   // Track u for incorrect predictions

    		// vacuity
			auto vacuity = ((int)num_classes) / strength;	// Calculate vacuity

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

            auto dissonance = getDisn(alpha);
            //printTensor(dissonance);

            auto belief_ent = shannon_entropy(belief);
            auto belief_tot_disagreement = total_disagreement(belief);
            auto expected_probability_ent = shannon_entropy(expected_probability); // Function to calculate entropy

            //auto vec_targets = tensorToVector(targets);
            //auto vec_belief = tensorToVector(belief);

            // statistics
            //torch::Tensor correct_preds = preds == targets;
            //running_corrects += correct_preds.sum();

            //if (batch_idx++ % m_options.n_loginterval == 0) {
            //    std::cout << "\rTrain Epoch: " << epoch;
            //    std::cout << "[" << batch_idx * batch.data.size(0);
            //    std::cout << "/" << train_dataset_size << "] ";
            //    std::cout << "Loss: " << loss.template item<float>();
            //    std::cout << " | Avg belief: " << belief.mean().item().toFloat();
            //    std::cout << " | Accuracy: " << static_cast<float>(corrects) / train_dataset_size;
            //    std::cout << std::flush;
            //}

            //printTensorSize(belief);
            batch_count++;

            computed_belief.push_back(belief);
            computed_uncertainty_mass.push_back(uncertainty_mass);
            computed_belief_ent.push_back(belief_ent);
            computed_belief_tot_disagreement.push_back(belief_tot_disagreement);
            computed_expected_probability_ent.push_back(expected_probability_ent);

            computed_u_succ.push_back(u_succ);
            computed_u_fail.push_back(u_fail);
            computed_prob_succ.push_back(prob_succ);
            computed_prob_fail.push_back(prob_fail);
            computed_belief_succ.push_back(belief_succ);
            computed_belief_fail.push_back(belief_fail);
            computed_dissonance.push_back(dissonance);
			computed_vacuity.push_back(vacuity);

            avg_loss += loss.template item<float>();
        }
        //std::cout << std::endl;

        std::stringstream ss; // Create a stringstream object
        ss << "\Evaluating: ";
        ss << "[" << data_size;
        ss << "/" << data_size << "] ";
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
        ss << " | Accuracy: " << static_cast<double>(corrects) / data_size;

        outputToFileAndConsole(m_options.logfile_path, ss.str());

        //auto epoch_loss = running_loss / m_kTrainBatchSize;
        //auto epoch_acc = running_corrects.to(torch::kDouble) / m_kTrainBatchSize;

        // Calculate 'epoch_loss' and 'epoch_acc'
        //double epoch_loss = running_loss.item<double>() / static_cast<double>(m_kTrainBatchSize);
        //double epoch_acc = static_cast<double>(running_corrects.item<int64_t>()) / static_cast<double>(m_kTrainBatchSize);

        //std::printf(
        //    "{} loss: {:.4f} acc: {:.4f}",
        //    epoch_loss, epoch_acc
        //);
    }

	/**
	* Training ENN
	* epoch: current epoch
	* data_loader: dataset
	* lossfunc: user defined loss function
	*/
	template <typename DataLoader>
	void trainingModel(DataLoader& data_loader, size_t data_size,
		std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&, size_t, size_t, size_t, torch::Device)> lossfunc)
	{
		size_t num_classes = m_options.n_classes;

		for (size_t epoch = 1; epoch <= m_options.n_epochs; ++epoch) {

			m_pModel->train();
			size_t batch_idx = 0;
			int64_t corrects = 0;
			float avg_loss = 0;
			int batch_count = 0;

			for (auto& batch : data_loader)
			{
				auto data = batch.data.to(m_device);
				auto targets = batch.target.to(m_device);

				// zero the parameter gradients
				m_pOptimizer->zero_grad();

				auto output = m_pModel->forward(data);

				// Compute the indices of the maximum values along dimension 1
				auto preds = output.argmax(1);

				// Convert to One Hot Encoding
				auto y = torch::one_hot(targets, num_classes).to(m_device);

				auto loss = lossfunc(
					output, y.to(torch::kFloat), epoch, num_classes, /*annealing_step*/10, m_device
				);

				auto pred = output.argmax(1);
				corrects += pred.eq(targets).sum().template item<int64_t>();

				// Backward pass
				loss.backward();

				//// zero the parameter gradients
				//m_pOptimizer->zero_grad();

				// Update weights
				m_pOptimizer->step();

				if (batch_idx++ % m_options.n_loginterval == 0) {
					std::cout << "\rTrain Epoch: " << epoch;
					std::cout << "[" << batch_idx * batch.data.size(0);
					std::cout << "/" << data_size << "] ";
					std::cout << "Loss: " << loss.template item<float>();
					std::cout << " | Accuracy: " << static_cast<float>(corrects) / data_size;
					std::cout << std::flush;
				}

				//printTensorSize(belief);
				batch_count++;

				avg_loss += loss.template item<float>();
			}
			//std::cout << std::endl;

			std::stringstream ss; // Create a stringstream object
			ss << "\rTrain Epoch: " << epoch;
			ss << "[" << data_size;
			ss << "/" << data_size << "] ";
			ss << "Loss: " << avg_loss / batch_count;
			ss << " | Accuracy: " << static_cast<double>(corrects) / data_size;

			outputToFileAndConsole(m_options.logfile_path, ss.str());
		}
	};

	int CreateLogFile();
};


