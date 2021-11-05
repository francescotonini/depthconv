#include <torch/extension.h>
#include <torch/types.h>
#include <vector>
#include <THC/THC.h>
#include <iostream>

torch::Tensor depthconv_forward(torch::Tensor input, torch::Tensor depth, torch::Tensor weights, torch::Tensor bias, int64_t stride_width, int64_t stride_height, int64_t padding_width, int64_t padding_height, int64_t dilation) {
    // TODO: check inputs
    
    int64_t batch_size = input.size(0);
    int64_t input_channels = input.size(1);
    int64_t input_height = input.size(2);
    int64_t input_width = input.size(3);
    int64_t kernel_width = weights.size(2);
    int64_t kernel_height = weights.size(3);
    int64_t output_channels = weights.size(0);
    int64_t output_height = (input_height + 2 * padding_height - kernel_height) / stride_height + 1;
    int64_t output_width = (input_width + 2 * padding_width - kernel_width) / stride_width + 1;

    torch::Tensor output = torch::zeros(torch::IntArrayRef({batch_size, output_channels, output_height, output_width})).cuda();
    torch::Tensor columns = torch::zeros(torch::IntArrayRef({input_channels * kernel_width * kernel_height, output_height * output_width})).cuda();
    torch::Tensor ones = torch::ones(torch::IntArrayRef({1, output_height * output_width})).cuda();

    weights = weights.reshape(torch::IntArrayRef({output_channels, input_channels * kernel_width * kernel_height})).cuda();
    bias = bias.reshape(torch::IntArrayRef({output_channels, 1})).cuda();

    for (int i = 0; i < batch_size; i++) {
        torch::Tensor this_input = input[i];
        
        // input, kernel_size, dilation, padding, stride
        columns = torch::im2col(
            this_input.clone(),
            torch::IntArrayRef({kernel_width, kernel_height}),
            torch::IntArrayRef({dilation, dilation}),
            torch::IntArrayRef({padding_width, padding_height}),
            torch::IntArrayRef({stride_width, stride_height})
        ).cuda();

        output[i].add_(bias.mm(ones).reshape(torch::IntArrayRef({output_channels, output_height, output_width})).cuda(), 1);
        output[i].add_(weights.mm(columns).reshape(torch::IntArrayRef({output_channels, output_height, output_width})).cuda(), 1);
    }

    return output;
}

torch::Tensor backward_grad_input(torch::Tensor input, torch::Tensor depth, torch::Tensor grad_output, torch::Tensor weights, int64_t stride_width, int64_t stride_height, int64_t padding_width, int64_t padding_height, int64_t dilation) {
    int64_t batch_size = input.size(0);
    int64_t input_channels = input.size(1);
    int64_t kernel_width = weights.size(2);
    int64_t kernel_height = weights.size(3);
    int64_t output_channels = grad_output.size(1);
    int64_t output_height = grad_output.size(2);
    int64_t output_width = grad_output.size(3);

    torch::Tensor grad_input = torch::zeros(torch::IntArrayRef({batch_size, output_channels, output_height, output_width})).cuda();
    torch::Tensor grad_columns = torch::zeros(torch::IntArrayRef({input_channels * kernel_width * kernel_height, output_height * output_width})).cuda();

    torch::Tensor weights_ = weights.clone();
    weights = weights.reshape(torch::IntArrayRef({output_channels, input_channels * kernel_width * kernel_height})).cuda();

    for (int i = 0; i < batch_size; i++) {
        torch::Tensor this_grad_output = grad_output[i].reshape(torch::IntArrayRef({output_channels, output_height * output_width})).cuda();
        torch::Tensor this_grad_columns = weights.mm(this_grad_output).cuda();

        grad_input[i].add_(torch::im2col(
            this_grad_columns.clone(),
            torch::IntArrayRef({kernel_width, kernel_height}),
            torch::IntArrayRef({dilation, dilation}),
            torch::IntArrayRef({padding_width, padding_height}),
            torch::IntArrayRef({stride_width, stride_height})
        ).cuda(), 1);
    }

    return grad_input;
}

std::vector<torch::Tensor> backward_grad_parameters(torch::Tensor input, torch::Tensor depth, torch::Tensor grad_output, torch::Tensor weights, int64_t stride_width, int64_t stride_height, int64_t padding_width, int64_t padding_height, int64_t dilation) {
    int64_t batch_size = input.size(0);
    int64_t input_channels = input.size(1);
    int64_t kernel_width = weights.size(2);
    int64_t kernel_height = weights.size(3);
    int64_t output_channels = grad_output.size(1);
    int64_t output_height = grad_output.size(2);
    int64_t output_width = grad_output.size(3);

    torch::Tensor grad_weights = torch::zeros(torch::IntArrayRef({weights.size(0), weights.size(1), weights.size(2), weights.size(3)})).cuda();
    torch::Tensor grad_bias = torch::zeros(torch::IntArrayRef({output_channels})).cuda();
    torch::Tensor columns = torch::zeros(torch::IntArrayRef({input_channels * kernel_width * kernel_height, output_height * output_width})).cuda();
    torch::Tensor ones = torch::ones(torch::IntArrayRef({output_height * output_width, 1})).cuda();

    torch::Tensor weights_ = weights.clone();
    weights = weights.reshape(torch::IntArrayRef({output_channels, input_channels * kernel_width * kernel_height})).cuda();

    for (int i = 0; i < batch_size; i++) {
        torch::Tensor this_input = input[i].clone();
        torch::Tensor this_grad_output = grad_output[i].reshape(torch::IntArrayRef({output_channels, output_height * output_width})).cuda();
        torch::Tensor this_grad_columns = weights.mm(this_grad_output).cuda();

        columns = torch::im2col(
            this_input,
            torch::IntArrayRef({kernel_width, kernel_height}),
            torch::IntArrayRef({dilation, dilation}),
            torch::IntArrayRef({padding_width, padding_height}),
            torch::IntArrayRef({stride_width, stride_height})
        ).cuda();

        grad_weights.add_(this_grad_output.mm(columns).reshape(torch::IntArrayRef({output_channels, input_channels, kernel_width, kernel_height})).cuda(), 1);
        grad_bias.add_(this_grad_output.mm(ones).reshape(torch::IntArrayRef({output_channels})), 1);
    }

    return {
        grad_weights,
        grad_bias
    };
}

std::vector<torch::Tensor> depthconv_backward(torch::Tensor input, torch::Tensor depth, torch::Tensor grad_output, torch::Tensor weights, int64_t stride_width, int64_t stride_height, int64_t padding_width, int64_t padding_height, int64_t dilation) {
    // TODO: check inputs!

    torch::Tensor grad_input = backward_grad_input(input, depth, grad_output, weights, stride_width, stride_height, padding_width, padding_height, dilation);
    std::vector<torch::Tensor> grad_params = backward_grad_parameters(input, depth, grad_output, weights, stride_width, stride_height, padding_width, padding_height, dilation);

    return {
        grad_input,
        grad_params[0], // Weights
        grad_params[1] // Bias
    };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &depthconv_forward, "depthconv forward (CUDA)");
    m.def("backward", &depthconv_backward, "depthconv backward (CUDA)");
}
