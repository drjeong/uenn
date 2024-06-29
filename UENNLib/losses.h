// https://nn.labml.ai/uncertainty/evidence/index.html

#pragma once

// Undefine max & min because it is defined in C++ headers
#undef max
#undef min

#include <torch/torch.h>
#include <functional>
#include "options.h"


#define TENSOR_TRACE(title, tensor) std::cout << "Tensor (" << title << "):" << std::endl << tensor << std::endl;

// Define a std::function type for your loss functions
using LossFunctionType = std::function<void(const torch::Tensor&, const torch::Tensor&, size_t, size_t, size_t, torch::Device)>;


torch::Tensor relu_evidence(const torch::Tensor& y);
torch::Tensor elu_evidence(const torch::Tensor& y);

// https://statproofbook.github.io/P/gam-kl.html
// torch.lgamma - Computes the natural logarithm of the absolute value of the gamma function on input.
// alpha: 1000 x 10

torch::Tensor kl_divergence(const torch::Tensor& alpha, size_t num_classes, torch::Device device);
torch::Tensor loglikelihood_loss(torch::Tensor y, torch::Tensor alpha, torch::Device device);

// Bayes Risk with Squared Error Loss
inline torch::Tensor SquaredErrorBayesRisk(
    const torch::Tensor& output, const torch::Tensor& target,
    size_t epoch_num, size_t num_classes,
    size_t annealing_step, torch::Device device)
{
    auto y = target.to(device);
    auto evidence = relu_evidence(output);
    auto alpha = evidence + 1;

    //auto loglikelihood = loglikelihood_loss(y, alpha, device);

    auto S = alpha.sum(1, true);

    auto loglikelihood_err = ((y - (alpha / S)).pow(2)).sum(1, true);

    auto loglikelihood_var = (alpha * (S - alpha) / (S * S * (S + 1))).sum(1, true);

    auto loglikelihood = loglikelihood_err + loglikelihood_var;

    {
		int epoch_num = 10; // Example epoch_num
		int annealing_step = 5; // Example annealing_step

		// Convert epoch_num and annealing_step to double and compute the ratio
		double ratio = static_cast<double>(epoch_num) / static_cast<double>(annealing_step);

		// Create tensors
		auto tensor_ratio = torch::tensor(ratio, torch::kFloat32);
		auto tensor_one = torch::tensor(1.0, torch::kFloat32);

		// Compute the minimum using element-wise min if necessary
		auto annealing_coef = torch::min(tensor_one, tensor_ratio);
    }

    auto annealing_coef = torch::min(
        torch::tensor(1.0, torch::kFloat32),
        torch::tensor((epoch_num / (double)annealing_step), torch::kFloat32)
    );

    auto kl_alpha = (alpha - 1) * (1 - y) + 1;
    auto kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device);

    return loglikelihood + kl_div;
};

// Type II Maximum Likelihood Loss
inline torch::Tensor MaximumLikelihoodLoss(
    const torch::Tensor& output, const torch::Tensor& target,
    size_t epoch_num, size_t num_classes,
    size_t annealing_step, torch::Device device)
{
    auto evidence = relu_evidence(output);
    auto alpha = evidence + 1;

    auto y = target.to(device);
    alpha = alpha.to(device);

    auto S/*strength*/ = torch::sum(alpha, /*dim=*/1, /*keepdim=*/true);
    auto A = torch::sum(y * (torch::log(S) - torch::log(alpha)), /*dim=*/1, /*keepdim=*/true);

    auto annealing_coef = torch::min(
        torch::tensor(1.0, torch::kFloat32),
        torch::tensor((epoch_num / (double)annealing_step), torch::kFloat32)
    );

    auto kl_alpha = (alpha - 1) * (1 - y) + 1;  // Remove non-misleading evidence
    auto kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device);
    auto loss = torch::mean(A + kl_div);

    return loss;
};

// Bayes Risk with Cross Entropy Loss
inline torch::Tensor CrossEntropyBayesRisk(
    const torch::Tensor& output, const torch::Tensor& target,
    size_t epoch_num, size_t num_classes,
    size_t annealing_step, torch::Device device)
{
    auto evidence = relu_evidence(output);
    auto alpha = evidence + 1;

    auto y = target.to(device);
    alpha = alpha.to(device);

    // uncertainty = # of classes / Dirichlet strength
    auto S = torch::sum(alpha, /*dim=*/1, /*keepdim=*/true);
    auto A = torch::sum(y * (torch::digamma(S) - torch::digamma(alpha)), /*dim=*/1, /*keepdim=*/true);

    // the KLD to gradually increase its influence (monotonic annealing)
    auto annealing_coef = torch::min(
        torch::tensor(1.0, torch::kFloat32),
        torch::tensor((epoch_num / (double)annealing_step), torch::kFloat32)
    );

    // Remove non-misleading evidence : target + (1 - target) * alpha
    auto alpha_tilde = (alpha - 1) * (1 - y) + 1;

	// Measuring the difference with KL divergence by computing first and second terms
    auto kl_div = annealing_coef * kl_divergence(alpha_tilde, num_classes, device);
    
	// Determine the mean loss over the batch
    auto loss = torch::mean(A + kl_div);

    return loss;
};


inline auto lossFunc = [=](unsigned long lossfunctype) ->std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&,
	size_t, size_t, size_t, torch::Device)> {
		if (lossfunctype == ENN_LOSS_DIGAMMA) return CrossEntropyBayesRisk;
		else if (lossfunctype == ENN_LOSS_LOG) return MaximumLikelihoodLoss;
		else return SquaredErrorBayesRisk;	// ENN_LOSS_MSE
};