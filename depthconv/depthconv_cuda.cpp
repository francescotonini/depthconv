#include <vector>
#include <torch/extension.h>
#include <torch/types.h>

// CUDA declarations
void depthconv_im2col(torch::Tensor data_im, torch::Tensor data_depth, int channels, int height, int width, int ksize_h, int ksize_w, int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, torch::Tensor data_col);
void depthconv_col2im(torch::Tensor data_col, torch::Tensor data_depth, int channels, int height, int width, int ksize_h, int ksize_w, int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, torch::Tensor grad_im);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor depthconv_forward(torch::Tensor input, torch::Tensor depth, torch::Tensor weight, torch::Tensor bias, int stride_height, int stride_width, int padding_height, int padding_width, int dilation_height, int dilation_width, bool has_bias) {
    CHECK_INPUT(input);
    CHECK_INPUT(depth);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);

    int batch_size = input.size(0);
    int input_channels = input.size(1);

    int input_height = input.size(2);
    int input_width = input.size(3);

    int kernel_height = weight.size(3);
    int kernel_width = weight.size(2);

    int output_channels = weight.size(0);
    int output_height = (input_height + 2 * padding_height - (dilation_height * (kernel_height - 1) + 1)) / stride_height + 1;
    int output_width = (input_width + 2 * padding_width - (dilation_width * (kernel_width - 1) + 1)) / stride_width + 1;

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
        if (has_bias) {
            this_output.add_(bias.mm(ones).reshape(torch::IntArrayRef({output_channels, output_height, output_width})).cuda(), 1);
        }

        // Kernel call
        depthconv_im2col(this_input, this_depth, input_channels, input_height, input_width, kernel_height, kernel_width, stride_height, stride_width, padding_height, padding_width, dilation_height, dilation_width, columns);

        // Add output
        this_output.add_(weight.mm(columns).reshape(torch::IntArrayRef({output_channels, output_height, output_width})).cuda(), 1);
    }

    return output;
}

torch::Tensor backward_grad_input(torch::Tensor input, torch::Tensor depth, torch::Tensor grad_output, torch::Tensor weight, int stride_width, int stride_height, int padding_width, int padding_height, int dilation_height, int dilation_width) {
    int batch_size = input.size(0);
    int input_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);

    int kernel_width = weight.size(2);
    int kernel_height = weight.size(3);

    int output_channels = grad_output.size(1);
    int output_height = grad_output.size(2);
    int output_width = grad_output.size(3);

    torch::Tensor grad_input = torch::zeros(torch::IntArrayRef({batch_size, input_channels, input_height, input_width})).cuda();
    torch::Tensor grad_columns = torch::zeros(torch::IntArrayRef({input_channels * kernel_width * kernel_height, output_height * output_width})).cuda();
    weight = weight.reshape(torch::IntArrayRef({output_channels, input_channels * kernel_width * kernel_height})).t().cuda();

    for (int i = 0; i < batch_size; i++) {
        torch::Tensor this_depth = depth[i];
        torch::Tensor this_grad_input = grad_input[i];
        torch::Tensor this_grad_output = grad_output[i].reshape(torch::IntArrayRef({output_channels, output_height * output_width})).cuda();

        grad_columns = weight.mm(this_grad_output).cuda();

        depthconv_col2im(grad_columns, this_depth, input_channels, input_height, input_width, kernel_height, kernel_width, stride_height, stride_width, padding_height, padding_width, dilation_height, dilation_width, this_grad_input);
    }

    return grad_input;
}

std::vector<torch::Tensor> backward_grad_parameters(torch::Tensor input, torch::Tensor depth, torch::Tensor grad_output, torch::Tensor weight, int stride_width, int stride_height, int padding_width, int padding_height, int dilation_height, int dilation_width, bool has_bias) {
    int batch_size = input.size(0);
    int input_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);

    int kernel_width = weight.size(2);
    int kernel_height = weight.size(3);

    int output_channels = grad_output.size(1);
    int output_height = grad_output.size(2);
    int output_width = grad_output.size(3);

    torch::Tensor grad_weight = torch::zeros(torch::IntArrayRef({weight.size(0), weight.size(1), weight.size(2), weight.size(3)})).cuda();
    torch::Tensor grad_bias = torch::zeros(torch::IntArrayRef({output_channels})).cuda();
    torch::Tensor columns = torch::zeros(torch::IntArrayRef({input_channels * kernel_width * kernel_height, output_height * output_width})).cuda();
    torch::Tensor ones = torch::ones(torch::IntArrayRef({output_height * output_width, 1})).cuda();

    for (int i = 0; i < batch_size; i++) {
        torch::Tensor this_input = input[i];
        torch::Tensor this_depth = depth[i];
        torch::Tensor this_grad_output = grad_output[i].reshape(torch::IntArrayRef({output_channels, output_height * output_width})).cuda();

        depthconv_im2col(this_input, this_depth, input_channels, input_height, input_width, kernel_height, kernel_width, stride_height, stride_width, padding_height, padding_width, dilation_height, dilation_width, columns);

        grad_weight.add_(this_grad_output.mm(columns.t()).reshape(torch::IntArrayRef({output_channels, input_channels, kernel_width, kernel_height})).cuda(), 1);

        if (has_bias) {
            grad_bias.add_(this_grad_output.mm(ones).reshape(torch::IntArrayRef({output_channels})), 1);
        }
    }

    return {
        grad_weight,
        grad_bias
    };
}

std::vector<torch::Tensor> depthconv_backward(torch::Tensor input, torch::Tensor depth, torch::Tensor weight, torch::Tensor bias, torch::Tensor grad_output, int stride_height, int stride_width, int padding_width, int padding_height, int dilation_height, int dilation_width, bool has_bias) {
    CHECK_INPUT(input);
    CHECK_INPUT(depth);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    CHECK_INPUT(grad_output);

    torch::Tensor grad_input = backward_grad_input(input, depth, grad_output, weight, stride_height, stride_width, padding_height, padding_width, dilation_height, dilation_width);
    std::vector<torch::Tensor> grad_params = backward_grad_parameters(input, depth, grad_output, weight, stride_height, stride_width, padding_height, padding_width, dilation_height, dilation_width, has_bias);

    return {
        grad_input,
        grad_params[0], // Weight
        grad_params[1] // Bias
    };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &depthconv_forward, "depthconv forward (CUDA)");
    m.def("backward", &depthconv_backward, "depthconv backward (CUDA)");
}
