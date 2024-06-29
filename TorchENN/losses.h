// https://nn.labml.ai/uncertainty/evidence/index.html

#pragma once
#include <torch/torch.h>
#include <functional>

#define TENSOR_TRACE(title, tensor) std::cout << "Tensor (" << title << "):" << std::endl << tensor << std::endl;

auto relu_evidence(const torch::Tensor& y)
{
    return torch::relu(y);
}

auto elu_evidence(const torch::Tensor& y)
{
	return torch::elu(y) + 1; // minimum value of the ELU function can be -1. It shifts the distribution to positive. 
}

// https://statproofbook.github.io/P/gam-kl.html
// torch.lgamma - Computes the natural logarithm of the absolute value of the gamma function on input.
// alpha: 1000 x 10

auto kl_divergence(
    const torch::Tensor &alpha, size_t num_classes, torch::Device device)
{
    // Create a tensor filled with ones
    torch::Tensor ones = torch::ones({ 1, (int)num_classes }, torch::kFloat32).to(device);

    auto sum_alpha = torch::sum(alpha, /*dim=*/1, /*keepdim=*/true);	// 1000 x 1
    auto first_term = (
        torch::lgamma(sum_alpha)
        - torch::lgamma(alpha).sum(/*dim=*/1, /*keepdim=*/true)	// output: 1000 x 1
        + torch::lgamma(ones).sum(/*dim=*/1, /*keepdim=*/true)
        - torch::lgamma(ones.sum(/*dim=*/1, /*keepdim=*/true))
        );
    auto second_term = (
        (alpha - ones)
        .mul(torch::digamma(alpha) - torch::digamma(sum_alpha))
        .sum(/*dim=*/1, /*keepdim=*/true)
        );
    auto kl = first_term + second_term;
    return kl;
}

//auto loglikelihood_loss(torch::Tensor y, torch::Tensor alpha, torch::Device device)
//{
//    y = y.to(device);
//    alpha = alpha.to(device);
//
//    torch::Tensor S = alpha.sum(1, true);
//
//    torch::Tensor loglikelihood_err = ((y - (alpha / S)).pow(2)).sum(1, true);
//
//    torch::Tensor loglikelihood_var = (alpha * (S - alpha) / (S * S * (S + 1))).sum(1, true);
//
//    torch::Tensor loglikelihood = loglikelihood_err + loglikelihood_var;
//
//    return loglikelihood;
//}

// Bayes Risk with Squared Error Loss
auto SquaredErrorBayesRisk = [](
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

    auto annealing_coef = torch::min(
        torch::tensor(1.0, torch::kFloat32),
        torch::tensor((epoch_num / (double)annealing_step), torch::kFloat32)
    );

    auto kl_alpha = (alpha - 1) * (1 - y) + 1;
    auto kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device);

    return loglikelihood + kl_div;
};

// Type II Maximum Likelihood Loss
auto MaximumLikelihoodLoss = [](
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
auto CrossEntropyBayesRisk = [](
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