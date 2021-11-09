#include <torch/extension.h>
#include <torch/types.h>

// CUDA declarations
void avgpool_forward(int count, torch::Tensor input_data, torch::Tensor input_depth_data, int channels, int height, int width, int pooled_height, int pooled_width, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w, torch::Tensor top_data, torch::Tensor depth_weight_count);
void avgpool_backward(int count, torch::Tensor gradOutput, torch::Tensor input_depth, torch::Tensor depth_weight_count, int channels, int height, int width, int pooled_height, int pooled_width, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w, torch::Tensor bottom_diff);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor depthavgpooling_forward(torch::Tensor input, torch::Tensor depth, torch::Tensor depth_weight_count, int kernel_height, int kernel_width, int stride_height, int stride_width, int padding_height, int padding_width) {
    CHECK_INPUT(input);
    CHECK_INPUT(depth);
    CHECK_INPUT(depth_weight_count);

    int batch_size = input.size(0);
    int input_channels = input.size(1);
    int input_rows = input.size(2);
    int input_cols = input.size(3);

    int output_rows = floor(float(input_rows - kernel_height + 2 * padding_height) / float(stride_height)) + 1;
    int output_cols = floor(float(input_cols - kernel_width + 2 * padding_width) / float(stride_width)) + 1;

    if (padding_width || padding_height)
    {
        // ensure that the last pooling starts inside the image
        // needed to avoid problems in ceil mode
        if ((output_rows - 1) * stride_height >= input_rows + padding_height) {
            --output_rows;
        }

        if ((output_cols - 1) * stride_width >= input_cols + padding_width) {
            --output_cols;
        }
    }

    torch::Tensor output = torch::zeros(torch::IntArrayRef({batch_size, input_channels, output_rows, output_cols})).cuda();

    for (int i = 0; i < batch_size; i++) {
        torch::Tensor this_input = input[i];
        torch::Tensor this_depth = depth[i];
        torch::Tensor this_depth_weight_count = depth_weight_count[i];
        torch::Tensor this_output = output[i];
        
        int count = this_output.numel();

        avgpool_forward(count, this_input, this_depth, input_channels, input_rows, input_cols, output_rows, output_cols, kernel_height, kernel_width, stride_height, stride_width, padding_height, padding_width, this_output, this_depth_weight_count);
    }

    return output;
}

torch::Tensor depthavgpooling_backward(torch::Tensor input, torch::Tensor depth, torch::Tensor depth_weight_count, torch::Tensor grad_output, int kernel_height, int kernel_width, int stride_height, int stride_width, int padding_width, int padding_height) {
    CHECK_INPUT(input);
    CHECK_INPUT(depth);
    CHECK_INPUT(depth_weight_count);
    CHECK_INPUT(grad_output);

    int batch_size = input.size(0);
    int input_channels = input.size(1);
    int input_rows = input.size(2);
    int input_cols = input.size(3);

    int output_rows = floor(float(input_rows - kernel_height + 2 * padding_height) / float(stride_height)) + 1;
    int output_cols = floor(float(input_cols - kernel_width + 2 * padding_width) / float(stride_width)) + 1;

    if (padding_width || padding_height)
    {
        // ensure that the last pooling starts inside the image
        // needed to avoid problems in ceil mode
        if ((output_rows - 1) * stride_height >= input_rows + padding_height) {
            --output_rows;
        }

        if ((output_cols - 1) * stride_width >= input_cols + padding_width) {
            --output_cols;
        }
    }

    torch::Tensor grad_input = torch::zeros(torch::IntArrayRef({batch_size, input_channels, input_rows, input_cols})).cuda();

    for (int i = 0; i < batch_size; i++) {
        torch::Tensor this_grad_input = grad_input[i];
        torch::Tensor this_depth = depth[i];
        torch::Tensor this_depth_weight_count = depth_weight_count[i];
        torch::Tensor this_grad_output = grad_output[i];
        
        int count = this_grad_input.numel();

        avgpool_backward(count, this_grad_output, this_depth, this_depth_weight_count, input_channels, input_rows, input_cols, output_rows, output_cols, kernel_height, kernel_width, stride_height, stride_width, padding_height, padding_width, this_grad_output);
    }

    return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &depthavgpooling_forward, "depthavgpooling forward (CUDA)");
    m.def("backward", &depthavgpooling_backward, "depthavgpooling backward (CUDA)");
}
