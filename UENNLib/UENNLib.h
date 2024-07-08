// The following ifdef block is the standard way of creating macros which make exporting
// from a DLL simpler. All files within this DLL are compiled with the UENNLIB_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see
// UENNLIB_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef UENNLIB_EXPORTS
#define UENNLIB_API __declspec(dllexport)
#else
#define UENNLIB_API __declspec(dllimport)
#endif

#include <eigen-3.4.0/Dense>
#include "options.h"

//extern UENNLIB_API int nUENNLib;

UENNLIB_API int fnUENN_MNIST_Train(Options option);
UENNLIB_API int fnUENN_MNIST_Test_w_TrainData(Options option);
UENNLIB_API int fnUENN_MNIST_Test_w_TestData(Options option,
	Eigen::MatrixXd& mat_belief, Eigen::MatrixXd& mat_uncertainty_mass, Eigen::MatrixXd& mat_belief_ent,
	Eigen::MatrixXd& mat_belief_tot_disagreement, Eigen::MatrixXd& mat_expected_probability_ent,
	Eigen::MatrixXd& mat_dissonance, Eigen::MatrixXd& mat_vacuity, 
	Eigen::MatrixXd& mat_labels, Eigen::MatrixXd& mat_match);
