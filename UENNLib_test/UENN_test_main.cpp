// UENN_test.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include <fstream>

#include <eigen-3.4.0/Dense>
#include "UENNLib.h"
#include "fcsv.h"

#define CPU 0
#define CUDA 1


int main()
{
	Options option;
	option.device = CUDA;
	option.n_epochs = 2;	// number of epochs to train

	// OPTIMIZER: OPTIMIZER_SGD | OPTIMIZER_ADAM
	option.optimizer = OPTIMIZER_ADAM;

	// LOSSFUNCTYPE: ENN_LOSS_DIGAMMA | ENN_LOSS_LOG | ENN_LOSS_MSE
	option.lossfunctype = ENN_LOSS_DIGAMMA;

	const std::string DATA_PATH = "D:\\WorkSpace2024\\Project_Pytorch\\TorchENN\\UENN_Original\\datasets";

	option.dataset_path = DATA_PATH + "\\mnist";
	option.result_path = DATA_PATH + "\\results";

	fnUENN_MNIST_Train(option);

	// Generate a random Eigen matrix
	Eigen::MatrixXd mat_belief;
	Eigen::MatrixXd mat_evidence;
	Eigen::MatrixXd mat_strength;
	Eigen::MatrixXd mat_uncertainty_mass;
	Eigen::MatrixXd mat_belief_ent;
	Eigen::MatrixXd mat_belief_tot_disagreement;
	Eigen::MatrixXd mat_expected_probability_ent;
	Eigen::MatrixXd mat_dissonance;
	Eigen::MatrixXd mat_vacuity;
	Eigen::MatrixXd mat_matches;
	Eigen::MatrixXd mat_labels;

	fnUENN_MNIST_Test_w_TestData(option,
		mat_belief, mat_evidence, mat_strength, 
		mat_uncertainty_mass, mat_belief_ent,
		mat_belief_tot_disagreement, mat_expected_probability_ent,
		mat_dissonance, mat_vacuity, mat_labels, mat_matches);

	// Convert to Eigen::VectorXd
	std::vector<Eigen::VectorXd> vectors;
	vectors.emplace_back(Eigen::Map<Eigen::VectorXd>(mat_uncertainty_mass.data(), mat_uncertainty_mass.size()));
	vectors.emplace_back(Eigen::Map<Eigen::VectorXd>(mat_belief_ent.data(), mat_belief_ent.size()));
	vectors.emplace_back(Eigen::Map<Eigen::VectorXd>(mat_belief_tot_disagreement.data(), mat_belief_tot_disagreement.size()));
	vectors.emplace_back(Eigen::Map<Eigen::VectorXd>(mat_expected_probability_ent.data(), mat_expected_probability_ent.size()));
	vectors.emplace_back(Eigen::Map<Eigen::VectorXd>(mat_dissonance.data(), mat_dissonance.size()));
	vectors.emplace_back(Eigen::Map<Eigen::VectorXd>(mat_vacuity.data(), mat_vacuity.size()));
	vectors.emplace_back(Eigen::Map<Eigen::VectorXd>(mat_strength.data(), mat_strength.size()));
	vectors.emplace_back(Eigen::Map<Eigen::VectorXd>(mat_matches.data(), mat_matches.size()));
	vectors.emplace_back(Eigen::Map<Eigen::VectorXd>(mat_labels.data(), mat_labels.size()));

	// Column names
	std::vector<std::string> columnNames;
	columnNames.push_back("uncertainty_mass");
	columnNames.push_back("belief_ent");
	columnNames.push_back("belief_tot_disagreement");
	columnNames.push_back("expected_probability_ent");
	columnNames.push_back("dissonance");
	columnNames.push_back("vacuity");
	columnNames.push_back("strength");
	columnNames.push_back("match");
	columnNames.push_back("label");

	// Save to CSV
	saveVectorsToCSV("vectors.csv", vectors, columnNames);
	saveMatrixToCSV("belief.csv", mat_belief);
	saveMatrixToCSV("evidence.csv", mat_evidence);

	Eigen::VectorXd label = Eigen::Map<Eigen::VectorXd>(mat_labels.data(), mat_labels.size());
	appendVectorToCSV("belief.csv", label, "label");
	appendVectorToCSV("evidence.csv", label, "label");

	appendVectorToCSV("belief.csv", Eigen::Map<Eigen::VectorXd>(mat_strength.data(), mat_strength.size()), "strength");
	appendVectorToCSV("evidence.csv", Eigen::Map<Eigen::VectorXd>(mat_strength.data(), mat_strength.size()), "strength");
	
	appendTextureInfoToCSV(mat_labels.size(), "belief.csv", "texture", "png");

	return 0;
}