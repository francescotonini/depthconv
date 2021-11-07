#include <vector>
#include <torch/extension.h>
#include <torch/types.h>

// CUDA declarations
void depthconv_im2col(torch::Tensor data_im, torch::Tensor data_depth, int64_t channels, int64_t height, int64_t width, int64_t ksize_h, int64_t ksize_w, int64_t pad_h, int64_t pad_w, int64_t stride_h, int64_t stride_w, int64_t dilation_h, int64_t dilation_w, torch::Tensor data_col);
void depthconv_col2im(torch::Tensor data_col, torch::Tensor data_depth, int64_t channels, int64_t height, int64_t width, int64_t ksize_h, int64_t ksize_w, int64_t pad_h, int64_t pad_w, int64_t stride_h, int64_t stride_w, int64_t dilation_h, int64_t dilation_w, torch::Tensor grad_im);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor depthconv_forward(torch::Tensor input, torch::Tensor depth, torch::Tensor weight, torch::Tensor bias, int64_t padding_height, int64_t padding_width, int64_t stride_height, int64_t stride_width, int64_t dilation_height, int64_t dilation_width) {
    CHECK_INPUT(input);
    CHECK_INPUT(depth);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);

    int64_t batch_size = input.size(0);
    int64_t input_channels = input.size(1);
    int64_t output_channels = weight.size(0);

    int64_t input_height = input.size(2);
    int64_t input_width = input.size(3);
    int64_t kernel_height = weight.size(3);
    int64_t kernel_width = weight.size(2);

    int64_t output_height = (input_height + 2 * padding_height - (dilation_height * (kernel_height - 1) + 1)) / stride_height + 1;
    int64_t output_width = (input_width + 2 * padding_width - (dilation_width * (kernel_width - 1) + 1)) / dilation_width + 1;

    torch::Tensor output = torch::zeros(torch::IntArrayRef({batch_size, output_channels, output_height, output_width})).cuda();
    torch::Tensor columns = torch::zeros(torch::IntArrayRef({input_channels * kernel_width * kernel_height, output_height * output_width})).cuda();
    torch::Tensor ones = torch::ones(torch::IntArrayRef({1, output_height * output_width})).cuda();

    weight = weight.reshape(torch::IntArrayRef({output_channels, input_channels * kernel_width * kernel_height})).cuda();
    bias = bias.reshape(torch::IntArrayRef({output_channels, 1})).cuda();

    for (int i = 0; i < batch_size; i++) {
        torch::Tensor this_input = input[i];
        torch::Tensor this_depth = depth[i];
        torch::Tensor this_output = output[i];
        
        // Bias
        this_output.add_(bias.mm(ones).reshape(torch::IntArrayRef({output_channels, output_height, output_width})).cuda(), 1);

        // Kernel call
        depthconv_im2col(this_input, this_depth, input_channels, input_height, input_width, kernel_height, kernel_width, padding_height, padding_width, stride_height, stride_width, dilation_height, dilation_width, columns);

        // Add output
        this_output.add_(weight.mm(columns).reshape(torch::IntArrayRef({output_channels, output_height, output_width})).cuda(), 1);
    }

    return output;
}

torch::Tensor backward_grad_input(torch::Tensor input, torch::Tensor depth, torch::Tensor grad_output, torch::Tensor weight, int64_t padding_width, int64_t padding_height, int64_t stride_width, int64_t stride_height, int64_t dilation_height, int64_t dilation_width) {
    int64_t batch_size = input.size(0);
    int64_t input_channels = input.size(1);
    int64_t input_height = input.size(2);
    int64_t input_width = input.size(3);

    int64_t kernel_width = weight.size(2);
    int64_t kernel_height = weight.size(3);

    int64_t output_channels = grad_output.size(1);
    int64_t output_height = grad_output.size(2);
    int64_t output_width = grad_output.size(3);

    torch::Tensor grad_input = torch::zeros(torch::IntArrayRef({batch_size, output_channels, output_height, output_width})).cuda();
    torch::Tensor columns = torch::zeros(torch::IntArrayRef({input_channels * kernel_width * kernel_height, output_height * output_width})).cuda();

    // torch::Tensor weight_ = weight.clone();
    // weight = weight.reshape(torch::IntArrayRef({output_channels, input_channels * kernel_width * kernel_height})).cuda();

    for (int i = 0; i < batch_size; i++) {
        torch::Tensor this_grad_input = grad_input[i];
        torch::Tensor this_depth = depth[i];
        torch::Tensor this_grad_output = grad_output[i]; // .reshape(torch::IntArrayRef({output_channels, output_height * output_width})).cuda();

        this_grad_output.add_(weight.mm(columns).reshape(torch::IntArrayRef({output_channels, output_height, output_width})).cuda(), 1);

        // torch::Tensor this_grad_columns = weight.mm(this_grad_output).cuda();

        depthconv_col2im(columns, this_depth, input_channels, input_height, input_width, kernel_height, kernel_width, padding_height, padding_width, stride_height, stride_width, dilation_height, dilation_width, this_grad_input);
    }

    return grad_input;
}

std::vector<torch::Tensor> backward_grad_parameters(torch::Tensor input, torch::Tensor depth, torch::Tensor grad_output, torch::Tensor weight, int64_t padding_width, int64_t padding_height, int64_t stride_width, int64_t stride_height, int64_t dilation_height, int64_t dilation_width) {
    int64_t batch_size = input.size(0);
    int64_t input_channels = input.size(1);
    int64_t input_height = input.size(2);
    int64_t input_width = input.size(3);

    int64_t kernel_width = weight.size(2);
    int64_t kernel_height = weight.size(3);

    int64_t output_channels = grad_output.size(1);
    int64_t output_height = grad_output.size(2);
    int64_t output_width = grad_output.size(3);

    torch::Tensor grad_weight = torch::zeros(torch::IntArrayRef({weight.size(0), weight.size(1), weight.size(2), weight.size(3)})).cuda();
    torch::Tensor grad_bias = torch::zeros(torch::IntArrayRef({output_channels})).cuda();
    torch::Tensor columns = torch::zeros(torch::IntArrayRef({input_channels * kernel_width * kernel_height, output_height * output_width})).cuda();
    torch::Tensor ones = torch::ones(torch::IntArrayRef({output_height * output_width, 1})).cuda();

    torch::Tensor weight_ = weight.clone();
    weight = weight.reshape(torch::IntArrayRef({output_channels, input_channels * kernel_width * kernel_height})).cuda();

    for (int i = 0; i < batch_size; i++) {
        torch::Tensor this_input = input[i].clone();
        torch::Tensor this_depth = depth[i].clone();
        torch::Tensor this_grad_output = grad_output[i].reshape(torch::IntArrayRef({output_channels, output_height * output_width})).cuda();
        torch::Tensor this_grad_columns = weight.mm(this_grad_output).cuda();

        depthconv_col2im(this_input, this_depth, input_channels, input_height, input_width, kernel_height, kernel_width, padding_height, padding_width, stride_height, stride_width, dilation_height, dilation_width, this_grad_columns);

        grad_weight.add_(this_grad_output.mm(columns).reshape(torch::IntArrayRef({output_channels, input_channels, kernel_width, kernel_height})).cuda(), 1);
        grad_bias.add_(this_grad_output.mm(ones).reshape(torch::IntArrayRef({output_channels})), 1);
    }

    return {
        grad_weight,
        grad_bias
    };
}

std::vector<torch::Tensor> depthconv_backward(torch::Tensor input, torch::Tensor depth, torch::Tensor weight, torch::Tensor bias, torch::Tensor grad_output, int64_t padding_height, int64_t padding_width, int64_t stride_height, int64_t stride_width, int64_t dilation_height, int64_t dilation_width) {
    CHECK_INPUT(input);
    CHECK_INPUT(depth);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    CHECK_INPUT(grad_output);

    torch::Tensor grad_input = backward_grad_input(input, depth, grad_output, weight, padding_height, padding_width, stride_height, stride_width, dilation_height, dilation_width);
    // std::vector<torch::Tensor> grad_params = backward_grad_parameters(input, depth, grad_output, weight, padding_height, padding_width, stride_height, stride_width, dilation_height, dilation_width);

    return {
        grad_input//,
        // grad_params[0], // Weight
        // grad_params[1] // Bias
    };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &depthconv_forward, "depthconv forward (CUDA)");
    m.def("backward", &depthconv_backward, "depthconv backward (CUDA)");
}
