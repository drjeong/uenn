#pragma once

#include <torch/torch.h>
#include "helper.h"

// Undefine max & min because it is defined in C++ headers
#undef max
#undef min

/// <summary>
/// Shannon entropy measure
/// - This is a commonly used measure of entropy that can also be applied to belief functions in evidential neural networks.
/// - Shannon entropy quantifies the uncertainty associated with a belief function by considering 
/// the distribution of mass across its components.
/// </summary>
/// <param name="belief"></param>
/// <returns></returns>
inline torch::Tensor shannon_entropy(torch::Tensor belief, float epsilon = 1e-20) {
	// Add a small epsilon value to handle zero beliefs
	belief = belief.clamp_min(epsilon);

	// Compute entropy for each belief function
	torch::Tensor entropy = -torch::sum(belief * torch::log(belief), 1);
	//printTensor(entropy);
	return entropy;
}

/// <summary>
/// Total Disagreement (TD) measure
/// - This method computes entropy based on the total disagreement between the components of the belief function.
/// - The entropy is computed as a function of the distances between the components of the belief function.
/// - Higher disagreement leads to higher entropy and thus higher uncertainty.
/// </summary>
/// <param name="belief"></param>
/// <returns></returns>
inline torch::Tensor total_disagreement(torch::Tensor belief) {
	// Compute pairwise distances between belief functions
	torch::Tensor diff = belief.unsqueeze(1) - belief.unsqueeze(2);
	torch::Tensor distances = torch::norm(diff, /*p=*/2, /*dim=*/-1);

	// Compute TD measure
	torch::Tensor td_measure = torch::sum(distances, /*dim=*/-1) / (belief.size(1) * (belief.size(1) - 1));
	return td_measure;
}

/// <summary>
/// Maximum Entropy for a given belief structure
/// Maximum entropy achievable given the structure of the belief function.
/// </summary>
/// <param name="num_classes"></param>
/// <returns></returns>
inline torch::Tensor max_entropy(int num_classes) {
	// Compute the maximum entropy using the formula: log(num_classes)
	torch::Tensor max_entropy = torch::log(torch::tensor(num_classes, torch::kFloat));
	return max_entropy;
}

/// <summary>
/// Hartley entropy measure
/// Hartley entropy is a measure of uncertainty similar to Shannon entropy but is based on 
/// the logarithm to the base of the number of possible outcomes.
/// It can be used to assess the uncertainty associated with the beliefs in an evidential neural network.
/// </summary>
/// <param name="belief"></param>
/// <returns></returns>
inline torch::Tensor hartley_entropy(torch::Tensor belief) {
	// Compute the number of possible outcomes (classes)
	int num_classes = belief.size(1);

	// Compute entropy using Hartley formula: log2(num_classes)
	torch::Tensor entropy = torch::log2(torch::tensor(num_classes, torch::kFloat));
	return entropy;
}

/// <summary>
/// Hull entropy measure
/// Hull entropy is a measure of uncertainty that accounts for the convex hull of the belief functions.
/// It captures the spread of the belief functions and can provide insights into the reliability of predictions.
/// </summary>
/// <param name="belief"></param>
/// <returns></returns>
inline torch::Tensor hull_entropy(torch::Tensor belief) {
	// Compute the convex hull of the belief functions
	torch::Tensor max_values, min_values;
	std::tie(max_values, std::ignore) = torch::max(belief, /*dim=*/1);
	std::tie(min_values, std::ignore) = torch::min(belief, /*dim=*/1);
	torch::Tensor hull = max_values - min_values;

	// Compute entropy using the hull
	torch::Tensor entropy = -torch::sum(hull * torch::log(hull), 1);
	return entropy;
}

/// <summary>
/// Distance-based Entropy measure
/// Instead of computing entropy directly from the belief functions, you can compute entropy 
/// based on the distances between belief functions.
/// This can involve measuring the spread or separation between belief functions in the belief space.
/// </summary>
/// <param name="belief"></param>
/// <returns></returns>
inline torch::Tensor distance_based_entropy(torch::Tensor belief) {
	// Compute pairwise distances between belief functions
	torch::Tensor diff = belief.unsqueeze(1) - belief.unsqueeze(2);
	torch::Tensor distances = torch::norm(diff, /*p=*/2, /*dim=*/-1);

	// Compute entropy based on distances
	torch::Tensor entropy = torch::mean(distances, /*dim=*/-1);
	return entropy;
}

/// <summary>
/// Robustness Entropy measure
/// It measures the robustness of the beliefs to perturbations or variations in the input data.
/// It evaluates how stable the predictions are with respect to small changes in the input.
/// </summary>
/// <param name="belief"></param>
/// <param name="epsilon"></param>
/// <returns></returns>
inline torch::Tensor robustness_entropy(torch::Tensor belief, float epsilon = 1e-20) {
	// Add a small epsilon value to handle zero beliefs
	belief = belief.clamp_min(epsilon);

	// Compute entropy using the L2 norm
	torch::Tensor entropy = torch::norm(belief, /*p=*/2, /*dim=*/1);
	return entropy;
}

/// <summary>
/// Consistency Entropy measure
/// It assesses the consistency of the beliefs across different samples or instances.
/// It measures the variability or stability of the predictions across different inputs.
/// </summary>
/// <param name="belief"></param>
/// <returns></returns>
inline torch::Tensor consistency_entropy(torch::Tensor belief, float epsilon = 1e-20) {
	// Compute the standard deviation of belief functions across samples
	torch::Tensor std_dev = torch::std(belief, /*dim=*/0);

	// Compute entropy using the standard deviation
	torch::Tensor entropy = -torch::sum(std_dev * torch::log(std_dev + epsilon), /*dim=*/-1);
	return entropy;
}

/// <summary>
/// Information Gain
/// It measures the reduction in uncertainty (entropy) obtained by making a prediction.
/// It quantifies how much new information is provided by the prediction compared to the prior uncertainty.
/// </summary>
/// <param name="belief"></param>
/// <param name="true_labels"></param>
/// <returns></returns>
inline torch::Tensor information_gain(torch::Tensor belief, torch::Tensor true_labels, float epsilon = 1e-20) {
	// Compute cross entropy between belief functions and true labels
	torch::Tensor cross_entropy = -torch::sum(true_labels * torch::log(belief + epsilon), /*dim=*/1);

	// Compute entropy of true labels
	torch::Tensor true_entropy = -torch::sum(true_labels * torch::log(true_labels + epsilon), /*dim=*/1);

	// Compute information gain as the difference between cross entropy and true entropy
	torch::Tensor gain = true_entropy - cross_entropy;
	return gain;
}

// Calculate dissonance of a vector of alpha #
inline torch::Tensor getDisn(torch::Tensor alpha) {
	// Calculate evidence and sum of alpha along rows
	torch::Tensor evi = alpha - 1;
	torch::Tensor s = torch::sum(alpha, /*dim=*/1, /*keepdim=*/true);

	// Calculate belief
	torch::Tensor blf = evi / s;

	// Initialize diss tensor
	torch::Tensor diss = torch::zeros_like(blf.select(1, 0)); // Initialize diss tensor

	// Calculate balance function
	auto Bal = [](torch::Tensor bi, torch::Tensor bj) {
		return 1 - torch::abs(bi - bj) / (bi + bj + 1e-8);
	};

	// Calculate relative mass balance for other columns
	torch::Tensor classes = torch::arange(alpha.size(1));
	for (int i = 0; i < classes.size(0); ++i) {
		torch::Tensor score_j_bal_sum = torch::zeros_like(blf.select(1, 0)); // Initialize sum of score_j_bal
		torch::Tensor score_j_sum = torch::zeros_like(blf.select(1, 0)); // Initialize sum of score_j
		for (int j = 0; j < classes.size(0); ++j) {
			if (j != i) {
				score_j_bal_sum += blf.select(1, j) * Bal(blf.select(1, j), blf.select(1, i));
				score_j_sum += blf.select(1, j);
			}
		}
		diss += blf.select(1, i) * (score_j_bal_sum / (score_j_sum + 1e-8));
	}

	return diss;
}

// Function to compute the entropy of a tensor x
inline torch::Tensor entropy(const torch::Tensor& x) {
	return -torch::sum(x * torch::log(x));
}

// Function to compute the entropy of the expected value of p given alpha values
inline torch::Tensor entropy_expected_p(const torch::Tensor& alpha) {
	// Compute the log of the Dirichlet normalization term
	torch::Tensor log_B = torch::lgamma(torch::sum(alpha)) - torch::sum(torch::lgamma(alpha));

	// Compute the expected value of p
	torch::Tensor expected_p = torch::exp(alpha) / (torch::sum(torch::exp(alpha)) /*+ 1*/);
	// Optional: add 1 to denominator for smoothing

	// Compute the entropy of the expected value of p
	return entropy(expected_p);
}


