/**
 * @file healper.h
 *
 * @brief This file is used to print values in tensor or std::vector to console output.
 *
 * 
 */

#pragma once

#include <torch/torch.h>
#include <iostream>
#include <string>
#include <eigen-3.4.0/Dense>

#define PRINT_VAR_NAME(x) std::cout << #x << ": " << std::endl

template<typename T>
void printVector(const std::vector<T>& vec)
{
	//PRINT_VAR_NAME(vec);

	// Print the elements of the vector
	for (const auto& value : vec) {
		std::cout << std::fixed << std::setprecision(4) << value << " ";
	}
	std::cout << std::endl;
};

void printTensor(const torch::Tensor& tensor);
void printTensorSize(const torch::Tensor& tensor, std::string label = "");
void printTensorType(const torch::Tensor& tensor, std::string label = "");


// Function to print tensor to cout
void printTensor(const torch::Tensor& tensor)
{
	// Convert tensor to CPU and to a float tensor if not already
	torch::Tensor cpuTensor = tensor.cpu();
	torch::Tensor floatTensor = cpuTensor.to(torch::kFloat32);

	printTensorSize(tensor);

	if (tensor.dim() == 0) {
		// Scalar Tensor
		std::cout << tensor.item<float>() << std::endl;
	}
	else if (tensor.dim() == 1) {
		for (int i = 0; i < tensor.size(0); ++i) {
			std::vector<float> row_vector = { tensor[i].item<float>() }; // create one-element vector
			printVector(row_vector);
		}
	}
	else {
		// Convert each row of the tensor into a vector of floats
		for (int i = 0; i < tensor.size(0); ++i) {
			std::vector<float> row_vector(tensor[i].data_ptr<float>(), tensor[i].data_ptr<float>() + tensor[i].size(0));
			printVector(row_vector);
		}
	}
}

// Function to print tensor to cout
void printTensorSize(const torch::Tensor& tensor, std::string label)
{
	std::cout << std::endl;
	// Print the sizes of each dimension
	if (label.size() > 0)
		std::cout << "Tensor (" << label << ") sizes: ";
	else
		std::cout << "Tensor sizes: ";
	
	if (tensor.dim() == 0) {
		std::cout << "scalarTensor is a scalar (0-dimensional)" << std::endl;
	}
	else {
		for (int64_t size : tensor.sizes().vec()) {
			std::cout << size << " ";
		}
	}
	
	std::cout << std::endl;
}

void printTensorType(const torch::Tensor& tensor, std::string label)
{
	// Check if the tensor is on CUDA
	if (tensor.device().is_cuda()) {
		std::cout << "Tensor";
		if (label.size() > 0) std::cout << " (" << label << ") ";
		std::cout << "is on CUDA(GPU)" << std::endl;
	}
	else if (tensor.device().is_cpu()) {
		std::cout << "Tensor";
		if (label.size() > 0) std::cout << " (" << label << ") ";
		std::cout << "is on CPU" << std::endl;
	}
}


class DualStreamBuf : public std::streambuf {
public:
	DualStreamBuf(std::ostream& stream1, std::ostream& stream2)
		: m_stream1(stream1), m_stream2(stream2) {}

	int overflow(int c) override {
		if (c != EOF) {
			if (m_stream1.rdbuf()->sputc(c) == EOF || m_stream2.rdbuf()->sputc(c) == EOF)
				return EOF;
		}
		return c;
	}

private:
	std::ostream& m_stream1;
	std::ostream& m_stream2;
};

class DualOstream : public std::ostream {
public:
	DualOstream(std::ostream& stream1, std::ostream& stream2)
		: std::ostream(&m_buf), m_buf(stream1, stream2) {}

private:
	DualStreamBuf m_buf;
};

void outputToFileAndConsole(const std::string& filename, const std::string& msg) {
	std::ofstream log_file(filename, std::ofstream::app);
	if (!log_file.is_open()) {
		std::cerr << "Error opening file!: " << filename << std::endl;
		return;
	}

	// Create a custom ostream that outputs to both std::cout and the file
	DualOstream dual_output(std::cout, log_file);

	// Output the message to both streams
	dual_output << msg << std::endl;

	// Close the file
	log_file.close();
}


float getMeanValue(std::vector<torch::Tensor> &tensor_vector)
{
	// Concatenate tensors along the specified dimension (e.g., dimension 0)
	torch::Tensor concatenatedTensor = torch::cat(tensor_vector, 0);

	// Compute the mean value of the concatenated tensor
	torch::Tensor meanValue = torch::mean(concatenatedTensor);

	return meanValue.item<float>();
}

// Function to flatten a tensor to a 1D vector
std::vector<float> flattenTensor(const torch::Tensor& tensor) {
	int64_t num_elements = tensor.numel();
	std::vector<float> result(num_elements);
	auto* tensor_ptr = tensor.data_ptr<float>();
	for (int i = 0; i < num_elements; ++i) {
		result[i] = *(tensor_ptr + i);
	}
	return result;
}

/// <summary>
/// Converting torch::Tensor to std::vector. It supports only 1D and 2D tensors
/// </summary>
/// <param name="tensor"></param>
/// <returns></returns>
std::vector<std::vector<float>> tensorToVector(const torch::Tensor& tensor)
{
	// Convert tensor to CPU and to a float tensor if not already
	torch::Tensor cpuTensor = tensor.cpu();
	torch::Tensor floatTensor = cpuTensor.to(torch::kFloat32);

	//printTensorSize(tensor);

	std::vector<std::vector<float>> result;

	// Check if the tensor is 1D or 2D
	assert((tensor.dim() == 1 || tensor.dim() == 2) && "Input tensor must be 1-dimensional or 2-dimensional.");

	// Check if the tensor is 2D
	if (tensor.dim() == 1) {
		// If the tensor is 1D, flatten it to a vector
		for (int i = 0; i < tensor.size(0); ++i) {
			std::vector<float> row_vector = { tensor[i].item<float>() }; // create one-element vector
			result.push_back(row_vector);
		}
	}
	else if (tensor.dim() == 2) {
		if (tensor.size(1) == 1) { // column size is 1
			// Iterate over rows of the tensor
			for (int i = 0; i < tensor.size(0); ++i) {
				// Access only the first element in the row
				std::vector<float> row_vector = { tensor[i][0].item<float>() }; // create one-element vector
				result.push_back(row_vector);
			}
		}
		else {
			// Iterate over rows of the tensor
			for (int i = 0; i < tensor.size(0); ++i) {
				// Access all elements in the row
				auto sliced_tensor = tensor[i];
				result.push_back(flattenTensor(sliced_tensor));
			}
		}
	}

	return result;
}


/// <summary>
/// Converting std::vector<torch::Tensor> to Eigen::MatrixXd.
/// </summary>
/// <param name="tensor"></param>
/// <returns></returns>
inline Eigen::MatrixXd convertTensorVecToEigen(const std::vector<torch::Tensor>& tensor_vector,
	size_t dataset_size, size_t num_classes) {
	// Assuming all tensors are of the same size and are 1D
	if (tensor_vector.empty()) return Eigen::MatrixXd();

	try {
		// Concatenate tensors along the specified dimension (e.g., dimension 0)
		torch::Tensor concatenatedTensor = torch::cat(tensor_vector, 0);

		// Reshape the tensor to data_size x num_class
		// Ensure the tensor is on CPU
		auto reshaped_tensor = concatenatedTensor.reshape({ (long)dataset_size, (long)num_classes }).to(torch::kCPU, torch::kDouble);

		// Ensure tensor is contiguous and on CPU
		if (!reshaped_tensor.is_contiguous()) {
			reshaped_tensor = reshaped_tensor.contiguous();
		}

		// Map the tensor data to Eigen::MatrixXd
		Eigen::Map<Eigen::MatrixXd> eigen_matrix(reshaped_tensor.data_ptr<double>(), 
			reshaped_tensor.size(0), reshaped_tensor.size(1));

		//// Print some matrix info to confirm
		//std::cout << "Eigen Matrix Rows: " << eigen_matrix.rows() << ", Columns: " << eigen_matrix.cols() << std::endl;
		//std::cout << "First element: " << eigen_matrix(0, 0) << std::endl;

		return eigen_matrix;
	}
	catch (const c10::Error& err) {
		std::cerr << "Error caught: " << err.what() << std::endl;
	}

	return Eigen::MatrixXd();
}