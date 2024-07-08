/**
 * Implementation of ENN class
 *
 *
 */

#pragma once

#include <torch/torch.h>
#include <iostream>
#include <functional>

#include "lenet.h"
#include "options.h"

class CENN
{
public:
    CENN(Options &options);
    ~CENN();

    void loadMNIST(const char* kDataRoot);

    // Define Optimizer (default: SGD optimizer)
    void SetOptimizer(unsigned long optimizer = OPTIMIZER_SGD);

    // Train ENN
    void trainENN();

    // Train CNN
    void trainCNN();

    // Train ENN with dataset
    template <typename DataLoader>
    void trainENN(size_t epoch, DataLoader& data_loader,
        torch::Tensor(*lossfunc)(const torch::Tensor&, const torch::Tensor&,
            size_t, size_t, size_t, torch::Device));

    // Train CNN with dataset
    template <typename DataLoader>
    void trainCNN(size_t epoch, DataLoader& data_loader,
        unsigned long lossfunc);

    // testing CNN
    void testCNN(unsigned long lossfunctype);

    // testing ENN
    void testENN();

    // testing CNN with dataset & loss function
    template <typename DataLoader>
    void testCNN(DataLoader& data_loader, unsigned long lossfunc);

    // testing ENN with dataset & loss function
    template <typename DataLoader>
    void testENN(DataLoader& data_loader,
        torch::Tensor(*lossfunc)(const torch::Tensor&, const torch::Tensor&,
            size_t, size_t, size_t, torch::Device));


    void printENNTrainingData(size_t& epoch, size_t batch_size,
        size_t train_dataset_size, size_t corrects, torch::Tensor& loss);

    int CreateLogFile();

private:
    // store all options
    Options m_options;

    torch::optim::Optimizer *m_pOptimizer;
	std::unique_ptr<torch::data::datasets::MNIST> m_train_dataset; // Using std::unique_ptr
    std::unique_ptr < torch::data::datasets::MNIST> m_test_dataset;

    void* m_test_loader;

    LeNet m_Model;
};


//class MNISTDataLoader {
//public:
//    MNISTDataLoader(const std::string& data_path, size_t batch_size)
//        : dataset_(data_path, torch::data::datasets::MNIST::Mode::kTrain),
//        //, data_loader_(dataset_, torch::data::DataLoaderOptions().batch_size(batch_size)) 
//        //data_loader_(torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
//        //    std::move(dataset_), batch_size))
//        data_loader_(torch::data::make_data_loader<torch::data::datasets::MNIST>(
//            dataset_,
//            batch_size))
//    {
//        //data_loader_ = torch::data::make_data_loader<torch::data::datasets::MNIST>(
//        //    std::move(dataset_), batch_size);
//    }
//
//    //torch::data::StatefulDataLoader<torch::data::datasets::MNIST>& getDataLoader() {
//    //    return data_loader_;
//    //}
//
//private:
//    torch::data::datasets::MNIST dataset_;
//    //std::shared_ptr<torch::data::StatefulDataLoader<torch::data::datasets::MNIST, torch::data::samplers::SequentialSampler>> data_loader_;
//
//    //torch::data::StatefulDataLoader<torch::data::datasets::MNIST> data_loader_;
//    std::shared_ptr<torch::data::StatefulDataLoader<torch::data::datasets::MNIST>> data_loader_;
//
//};
//
//
