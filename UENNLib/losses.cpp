#include "pch.h"
#include "losses.h"

torch::Tensor relu_evidence(const torch::Tensor& y)
{
	return torch::relu(y);
}

torch::Tensor elu_evidence(const torch::Tensor& y)
{
	return torch::elu(y) + 1; // minimum value of the ELU function can be -1. It shifts the distribution to positive. 
}

// https://statproofbook.github.io/P/gam-kl.html
// torch.lgamma - Computes the natural logarithm of the absolute value of the gamma function on input.
// alpha: 1000 x 10

torch::Tensor kl_divergence(const torch::Tensor& alpha, size_t num_classes, torch::Device device)
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

torch::Tensor loglikelihood_loss(torch::Tensor y, torch::Tensor alpha, torch::Device device)
{
	y = y.to(device);
	alpha = alpha.to(device);

	torch::Tensor S = alpha.sum(1, true);

	torch::Tensor loglikelihood_err = ((y - (alpha / S)).pow(2)).sum(1, true);

	torch::Tensor loglikelihood_var = (alpha * (S - alpha) / (S * S * (S + 1))).sum(1, true);

	torch::Tensor loglikelihood = loglikelihood_err + loglikelihood_var;

	return loglikelihood;
}