/**
 * Implementation of LetNet class
 *
 *
 */

#pragma once
#include <torch/torch.h>

class LeNet : public torch::nn::Module {
public:
	LeNet(bool dropout = false) {
		use_dropout = dropout;

		// First 5x5 convolution layer
		//conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 20, /*kernel_size=*/5)));

		// Second 5x5 convolution layer
		//conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(20, 50, /*kernel_size=*/5)));

		// First fully-connected layer that maps to 500 features
		//fc1 = register_module("fc1", torch::nn::Linear(20000 /*20 x 20 x 50*/, 500));
		//fc2 = register_module("fc2", torch::nn::Linear(500, 10));

		conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 10, /*kernel_size=*/5)));
		conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(10, 20, /*kernel_size=*/5)));
		fc1 = register_module("fc1", torch::nn::Linear(8000 /*20 x 20 x 20*/, 50));   // number of channels ?height ?width
		fc2 = register_module("fc2", torch::nn::Linear(50, 10));
	}

	torch::Tensor forward(torch::Tensor x) {
		// Apply the first convolution layer followed by ReLU activation
		auto out = torch::relu(conv1->forward(x));

		// Perform max pooling after the first convolution layer
		out = torch::max_pool2d(out, /*kernel size*/1);

		// Apply the second convolution layer followed by ReLU activation
		out = torch::relu(conv2->forward(out));

		// Perform max pooling after the second convolution layer
		out = torch::max_pool2d(out, /*kernel size*/1);

		// Reshape the output tensor before passing it to the fully connected layers
		out = out.view({ out.size(0), -1 });

		// Apply ReLU activation to the output of the first fully connected layer
		out = torch::relu(fc1->forward(out));

		if (use_dropout) {
			out = torch::dropout(out, /*p=*/0.5, /*train=*/is_training());
		}

		// Pass the output through the second fully connected layer
		out = fc2->forward(out);

		return (out);

		//// Apply first convolution and max pooling.
		//x = torch::relu(torch::max_pool2d(conv1->forward(x), /*kernel size*/1)); std::cout << "34" << std::endl;

		//// Apply second convolution and max pooling.
		//x = torch::relu(torch::max_pool2d(conv2->forward(x), /*kernel size*/1)); std::cout << "37" << std::endl;

		//// flatten it to (batch_size, -1) to feed it into a fully connected layer
		//// Flatten the tensor to shape [batch_size, 50 * 4 * 4]
		//x = x.view({ x.size(0), -1 }); std::cout << "41" << std::endl;

		//x = torch::relu(fc1->forward(x)); std::cout << "43" << std::endl;
		//if (use_dropout) {
		//    x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
		//}
		//x = fc2->forward(x); std::cout << "47" << std::endl;
		//return x;
	}

private:
	bool use_dropout;
	torch::nn::Conv2d conv1{ nullptr }, conv2{ nullptr };
	torch::nn::Linear fc1{ nullptr }, fc2{ nullptr };
};

//class LeNet : public torch::nn::Module
//{
//public:
//    LeNet(): 
//		conv1(torch::nn::Conv2dOptions(1, 10, /*kernel_size=*/5)),
//		conv2(torch::nn::Conv2dOptions(10, 20, /*kernel_size=*/5)),
//		fc1(320, 50),
//		fc2(50, 10)
//    {
//		register_module("conv1", conv1);
//		register_module("conv2", conv2);
//		register_module("conv2_drop", conv2_drop);
//		register_module("fc1", fc1);
//		register_module("fc2", fc2);
//    }
//
//    torch::Tensor forward(torch::Tensor x)
//    {
//		x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
//		x = torch::relu(torch::max_pool2d(conv2_drop->forward(conv2->forward(x)), 2));
//		x = x.view({ -1, 320 });
//		x = torch::relu(fc1->forward(x));
//		x = torch::dropout(x, /*p=*/0.5, /*training=*/is_training());
//		x = fc2->forward(x);
//		return torch::log_softmax(x, /*dim=*/1);
//    }
//
//private:
//    torch::nn::Conv2d conv1;
//    torch::nn::Conv2d conv2;
//    torch::nn::Dropout2d conv2_drop;
//    torch::nn::Linear fc1;
//    torch::nn::Linear fc2;
//};