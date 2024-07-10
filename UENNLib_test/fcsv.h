#pragma once

#include <iostream>
#include <vector>
#include <fstream>
#include <eigen-3.4.0/Dense>

inline void saveMatrixToCSV(const std::string& filename, const Eigen::MatrixXd& matrix) {
	std::ofstream file(filename);
	if (!file.is_open()) {
		std::cerr << "Error opening file: " << filename << std::endl;
		return;
	}

	// Generate and write column names
	for (int i = 0; i < matrix.cols(); ++i) {
		file << "Col" << (i + 1);
		if (i != matrix.cols() - 1) file << ",";
	}
	file << "\n";

	// Write data
	for (int i = 0; i < matrix.rows(); ++i) {
		for (int j = 0; j < matrix.cols(); ++j) {
			file << matrix(i, j);
			if (j != matrix.cols() - 1) file << ",";
		}
		file << "\n";
	}

	file.close();
}

inline void saveVectorsToCSV(const std::string& filename, const std::vector<Eigen::VectorXd>& vectors, const std::vector<std::string>& columnNames) {
	if (vectors.empty()) return; // Nothing to do if there are no vectors

	std::ofstream file(filename);
	if (!file.is_open()) {
		std::cerr << "Failed to open file: " << filename << std::endl;
		return;
	}

	// Check if the number of column names matches the number of vectors
	if (columnNames.size() != vectors.size()) {
		std::cerr << "Column names size does not match number of vectors!" << std::endl;
		return;
	}

	// Write column names
	for (size_t j = 0; j < columnNames.size(); ++j) {
		file << columnNames[j];
		if (j < columnNames.size() - 1) file << ",";  // Comma between column names
	}
	file << "\n";  // Newline after header

	// Assume all vectors are the same length
	int numRows = vectors.front().size();

	// Write each row
	for (int i = 0; i < numRows; ++i) {
		for (size_t j = 0; j < vectors.size(); ++j) {
			file << vectors[j](i);
			if (j < vectors.size() - 1) file << ",";  // Comma between columns
		}
		file << "\n";  // Newline after each row
	}

	file.close();
}

inline void appendVectorToCSV(const std::string& filename, const Eigen::VectorXd& vector, const std::string& newColumnName) {
	std::ifstream fileIn(filename);
	std::vector<std::string> lines;
	std::string line;

	// Read existing data
	while (getline(fileIn, line)) {
		lines.push_back(line);
	}
	fileIn.close();

	// Check vector length matches the number of CSV data rows (excluding header)
	if (lines.size() - 1 != static_cast<size_t>(vector.size())) {
		std::cerr << "Error: The vector size does not match the number of data rows in the CSV." << std::endl;
		return;
	}

	std::ofstream fileOut(filename);

	// Handle header
	if (!lines.empty()) {
		fileOut << lines[0] << "," << newColumnName << "\n";  // Append new column name to the header
	}

	// Append vector data to each line and write back
	for (size_t i = 1; i < lines.size(); ++i) {
		fileOut << lines[i] << "," << vector(i - 1) << "\n";  // Note the index adjustment for the vector
	}

	fileOut.close();
}

inline void appendTextureInfoToCSV(const size_t& size, const std::string& filename, 
	const std::string& newColumnName, const std::string& imageExtension) {

	// Create a vector of the given size
	Eigen::VectorXd vector(size);

	// Initialize each element to its index
	for (int i = 0; i < size; ++i) {
		vector(i) = i;
	}

	std::ifstream fileIn(filename);
	std::vector<std::string> lines;
	std::string line;

	// Read existing data
	while (getline(fileIn, line)) {
		lines.push_back(line);
	}
	fileIn.close();

	// Check vector length matches the number of CSV data rows (excluding header)
	if (lines.size() - 1 != static_cast<size_t>(vector.size())) {
		std::cerr << "Error: The vector size does not match the number of data rows in the CSV." << std::endl;
		return;
	}

	std::ofstream fileOut(filename);

	// Handle header
	if (!lines.empty()) {
		fileOut << lines[0] << "," << newColumnName << "\n";  // Append new column name to the header
	}

	// Append vector data to each line and write back
	for (size_t i = 1; i < lines.size(); ++i) {
		fileOut << lines[i] << "," << vector(i - 1) << "." << imageExtension << "\n";  // Note the index adjustment for the vector
	}

	fileOut.close();
}