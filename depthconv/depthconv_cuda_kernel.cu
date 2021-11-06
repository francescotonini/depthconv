#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// TODO: add explanation
// https://stackoverflow.com/questions/37566987/cuda-atomicadd-for-doubles-definition-error
// https://pytorch.org/docs/stable/cpp_extension.html
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double *address, double val)
{
    unsigned long long int *address_as_ull = (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

template <typename scalar_t>
__device__ scalar_t get_gradient_weight(scalar_t argmax_h, scalar_t argmax_w, int h, int w, int height, int width) {
    if (argmax_h < 0 || argmax_h > height || argmax_w < 0 || argmax_w > width) {
        // empty
        return 0;
    }

    argmax_h = max(argmax_h, (scalar_t)0.0f);
    argmax_w = max(argmax_w, (scalar_t)0.0f);

    int argmax_h_low = (int)argmax_h;
    int argmax_w_low = (int)argmax_w;
    int argmax_h_high;
    int argmax_w_high;

    if (argmax_h_low >= height - 1) {
        argmax_h_high = argmax_h_low = height - 1;
        argmax_h = (scalar_t)argmax_h_low;
    } else {
        argmax_h_high = argmax_h_low + 1;
    }

    if (argmax_w_low >= width - 1) {
        argmax_w_high = argmax_w_low = width - 1;
        argmax_w = (scalar_t)argmax_w_low;
    } else {
        argmax_w_high = argmax_w_low + 1;
    }

    scalar_t weight = 0;
    if (h == argmax_h_low) {
        if (w == argmax_w_low) {
            weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
        } else if (w == argmax_w_high) {
            weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
        }
    } else if (h == argmax_h_high) {
        if (w == argmax_w_low) {
            weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
        } else if (w == argmax_w_high) {
            weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
        }
    }

    return weight;
}

template <typename scalar_t>
__global__ void depthconv_im2col_gpu_kernel(int64_t n, scalar_t *data_im, scalar_t *data_depth, int64_t height, int64_t width, int64_t kernel_h, int64_t kernel_w, int64_t pad_h, int64_t pad_w, int64_t stride_h, int64_t stride_w, int64_t dilation_h, int64_t dilation_w, int64_t height_col, int64_t width_col, scalar_t *data_col) {
    // CxHxW --> (khxkw)x(CxHxW)
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < n; index += blockDim.x * gridDim.x) {
        int w_col = index % width_col;
        int h_col = (index / width_col) % height_col;
        int c_im = (index / width_col) / height_col;
        int c_col = c_im * kernel_h * kernel_w;
        int h_in = h_col * stride_h - pad_h;
        int w_in = w_col * stride_w - pad_w;

        scalar_t *data_col_ptr = data_col + (c_col * height_col + h_col) * width_col + w_col;
        scalar_t *data_im_ptr = data_im + (c_im * height + h_in) * width + w_in;
        scalar_t *data_depth_ptr = data_depth + h_in * width + w_in;
        
        scalar_t Di = 0.;
        bool valid = true;
        if ((h_in + dilation_h * (kernel_h - 1) / 2) >= 0 && w_in + dilation_w * (kernel_w - 1) / 2 >= 0 && (h_in + dilation_h * (kernel_h - 1) / 2) < height && w_in + dilation_w * (kernel_w - 1) / 2 < width) {
            Di = data_depth[(h_in + dilation_h * (kernel_h - 1) / 2) * width + w_in + dilation_w * (kernel_w - 1) / 2];
        }
        else {
            valid = false;
        }
        // scalar_t Di = data_depth[(h_in + (kernel_h - 1) / 2 + dilation_h - 1)
        // * width + (w_in + (kernel_w - 1) / 2 + dilation_w - 1)];

        for (int i = 0; i < kernel_h; ++i) {
            for (int j = 0; j < kernel_w; ++j) {
                scalar_t val = static_cast<scalar_t>(0);
                scalar_t Dval = static_cast<scalar_t>(0);
                int h_im = h_in + i * dilation_h;
                int w_im = w_in + j * dilation_w;

                if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
                    int map_h = i * dilation_h;
                    int map_w = j * dilation_w;

                    val = data_im_ptr[map_h * width + map_w];
                    if (valid) {
                        Dval = data_depth_ptr[map_h * width + map_w];
                    }
                    
                    // printf("%f,%d\n",Dval,h_in * width + w_in+map_h * width + map_w -
                    // ((h_in + (kernel_h - 1) / 2 + dilation_h - 1) * width + (w_in +
                    // (kernel_w - 1) / 2 + dilation_w - 1)));
                    // printf("Di-Dval: %f, %f\n", Di, Dval);
                    // if (exp(-abs(Di - Dval))<0.2)
                    //	printf("Di-Dval: %f\n", exp(-abs(Di - Dval)));
                    
                    val *= exp(-abs(Di - Dval));
                }

                *data_col_ptr = val;
                data_col_ptr += height_col * width_col;
            }
        }
    }
}

template <typename scalar_t>
__global__ void depthconv_col2im_gpu_kernel(int n, scalar_t *data_col, scalar_t *data_depth, int channels, int height, int width, int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w, int height_col, int width_col, scalar_t *grad_im) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < n; index += blockDim.x * gridDim.x) {
        for (int ii = 0; ii < kernel_h * kernel_w; ii++) {
            int ii_index = ii + index * kernel_h * kernel_w;
            int j = (ii_index / width_col / height_col) % kernel_w;
            int i = (ii_index / width_col / height_col / kernel_w) % kernel_h;
            int c = ii_index / width_col / height_col / kernel_w / kernel_h;
            // compute the start and end of the output
            int w_out = ii_index % width_col;
            int h_out = (ii_index / width_col) % height_col;
            int w_in = w_out * stride_w - pad_w;
            int h_in = h_out * stride_h - pad_h;

            // scalar_t cur_inv_h_data = h_in + i * dilation_h;
            // scalar_t cur_inv_w_data = w_in + j * dilation_w;

            scalar_t cur_top_grad = data_col[ii_index];
            int cur_h = h_in + i * dilation_h; //(int)cur_inv_h_data;
            int cur_w = w_in + j * dilation_w; //(int)cur_inv_w_data;

            scalar_t Di = 0.;
            bool valid = true;
            if ((h_in + dilation_h * (kernel_h - 1) / 2) >= 0 && w_in + dilation_w * (kernel_w - 1) / 2 >= 0 && (h_in + dilation_h * (kernel_h - 1) / 2) < height && w_in + dilation_w * (kernel_w - 1) / 2 < width) {
                Di = data_depth[(h_in + dilation_h * (kernel_h - 1) / 2) * width + w_in + dilation_w * (kernel_w - 1) / 2];
            }
            else {
                valid = false;
            }

            //      scalar_t Di = data_depth[(h_in + dilation_h * (kernel_h - 1) /
            //      2) * width + w_in  + dilation_w * (kernel_w - 1) / 2];
            // scalar_t Di = data_depth[(h_in + (kernel_h - 1) / 2 + dilation_h -
            // 1) * width + w_in  + (kernel_w - 1) / 2 + dilation_w - 1];
            // printf("%d\n",(h_in + dilation_h * (kernel_h - 1) / 2) * width + w_in
            // + dilation_w * (kernel_w - 1) / 2); data_depth[cur_h * width + cur_w];
            // data_depth[(h_in + (kernel_h - 1) / 2 + dilation_h - 1) * width + w_in
            // + (kernel_w - 1) / 2 + dilation_w - 1];

            int cur_bottom_grad_pos = (c * height + cur_h) * width + cur_w;
            int cur_bottom_depth_pos = (cur_h)*width + cur_w;

            // printf("%d,%d,%d,%d\n",i,j,((h_in + dilation_h * (kernel_h - 1) / 2) *
            // width + w_in  + dilation_w * (kernel_w - 1) /
            // 2-cur_bottom_depth_pos),dilation_h); printf("%d\n",((h_in + dilation_h *
            // (kernel_h - 1) / 2) * width + w_in  + dilation_w * (kernel_w - 1) /
            // 2-cur_bottom_depth_pos));

            scalar_t Dval = 0.;
            if (valid) {
                Dval = data_depth[cur_bottom_depth_pos];
            }

            // TODO: fix this!
            if (cur_h >= 0 && cur_h < height && cur_w >= 0 && cur_w < width) {
                atomicAdd(grad_im + cur_bottom_grad_pos, cur_top_grad * exp(-abs(Di - Dval)));
            }
        }
    }
}

void depthconv_im2col(torch::Tensor data_im, torch::Tensor data_depth, int64_t channels, int64_t height, int64_t width, int64_t ksize_h, int64_t ksize_w, int64_t pad_h, int64_t pad_w, int64_t stride_h, int64_t stride_w, int64_t dilation_h, int64_t dilation_w, torch::Tensor data_col) {
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int64_t height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
    int64_t width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
    int64_t num_kernels = channels * height_col * width_col;
    int64_t num_blocks = (num_kernels + 1024 - 1) / 1024;

    AT_DISPATCH_FLOATING_TYPES(data_im.type(), "depthconv_im2col", ([&] {
        depthconv_im2col_gpu_kernel<scalar_t><<<num_blocks, num_kernels>>>(
            num_kernels,
            data_im.data<scalar_t>(),
            data_depth.data<scalar_t>(),
            height,
            width,
            ksize_h,
            ksize_w,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            height_col,
            width_col,
            data_col.data<scalar_t>());
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in depthconv_im2col: %s\n", cudaGetErrorString(err));
        // TODO(BZ) panic
    }
}

void depthconv_col2im(torch::Tensor data_col, torch::Tensor data_depth, int64_t channels, int64_t height, int64_t width, int64_t ksize_h, int64_t ksize_w, int64_t pad_h, int64_t pad_w, int64_t stride_h, int64_t stride_w, int64_t dilation_h, int64_t dilation_w, torch::Tensor grad_im) {
    int64_t height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
    int64_t width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
    // To avoid involving atomic operations, we will launch one kernel per
    // bottom dimension, and then in the kernel add up the top dimensions.
    int64_t num_kernels = channels * height_col * width_col;
    int64_t num_blocks = (num_kernels + 1024 - 1) / 1024;

    AT_DISPATCH_FLOATING_TYPES(data_col.type(), "depthconv_col2im", ([&] {
        depthconv_col2im_gpu_kernel<scalar_t><<<num_blocks, num_kernels>>>(
            num_kernels,
            data_col.data<scalar_t>(),
            data_depth.data<scalar_t>(),
            channels,
            height,
            width,
            ksize_h,
            ksize_w,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            height_col,
            width_col,
            grad_im.data<scalar_t>());
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in depthconv_col2im: %s\n", cudaGetErrorString(err));
        // TODO(BZ) panic
    }
}
