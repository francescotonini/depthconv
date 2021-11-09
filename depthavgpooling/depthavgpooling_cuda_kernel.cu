#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_NUM_THREADS 1024

template <typename Dtype, typename Acctype>
__global__ void avgpool_forward_kernel(const int nthreads, const Dtype* const bottom_data, const Dtype* const bottom_data_depth, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, const int pad_h, const int pad_w, Dtype* const top_data, Dtype* const depth_weight_count) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; index += blockDim.x * gridDim.x) {
        const int pw = index % pooled_width;
        const int ph = (index / pooled_width) % pooled_height;
        const int c = (index / pooled_width / pooled_height) % channels;
        const int n = index / pooled_width / pooled_height / channels;
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, height + pad_h);
        int wend = min(wstart + kernel_w, width + pad_w);
        Dtype pool_size = (hend - hstart) * (wend - wstart);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        hend = min(hend, height);
        wend = min(wend, width);
        Acctype aveval = Acctype(0);
        const Dtype* const bottom_slice = bottom_data + (n * channels + c) * height * width;

        int ih = (hstart + hend) / 2;
        int iw = (wstart + wend) / 2;
        Acctype Di = bottom_data_depth[ih * width + iw];
        Acctype divcount = 0.;

        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                Acctype Dval = bottom_data_depth[h * width + w];
                Acctype weight_val = exp(-abs(Di - Dval));

                pool_size -= (1. - weight_val);
                aveval += (bottom_slice[h * width + w] * weight_val);
            }
        }
        depth_weight_count[ih * width + iw] = pool_size;
        top_data[index] = Dtype(aveval / pool_size);
    }
}

template <typename Dtype, typename Acctype>
__global__ void avgpool_backward_kernel(const int nthreads, const Dtype* const top_diff, const Dtype* const bottom_data_depth, const Dtype* const depth_weight_count, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, const int pad_h, const int pad_w, Dtype* const bottom_diff) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; index += blockDim.x * gridDim.x) {
        // find out the local index
        // find out the local offset
        const int w = index % width + pad_w;
        const int h = (index / width) % height + pad_h;
        const int c = (index / width / height) % channels;
        const int n = index / width / height / channels;
        const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
        const int phend = min(h / stride_h + 1, pooled_height);
        const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
        const int pwend = min(w / stride_w + 1, pooled_width);
        Acctype gradient = Acctype(0);
        const Dtype* const top_diff_slice = top_diff + (n * channels + c) * pooled_height * pooled_width;

        bool valid = true;
        Acctype Dval = 0.;
        if (h < 0 || h > height || w < 0 || w > width) {
            valid = false;
        }
        else {
            Dval = bottom_data_depth[h * width + w];
        }

        for (int ph = phstart; ph < phend; ++ph) {
            for (int pw = pwstart; pw < pwend; ++pw) {
                // figure out the pooling size
                int hstart = ph * stride_h - pad_h;
                int wstart = pw * stride_w - pad_w;
                int hend = min(hstart + kernel_h, height + pad_h);
                int wend = min(wstart + kernel_w, width + pad_w);
                Dtype weight_count = (hend - hstart) * (wend - wstart);
                hstart = max(hstart, 0);
                wstart = max(wstart, 0);
                hend = min(hend, height);
                wend = min(wend, width);

                int ih = (hstart + hend) / 2;
                int iw = (wstart + wend) / 2;
                Acctype Di = bottom_data_depth[ih * width + iw];
                Acctype weight_val = 1.;//
                if(valid && depth_weight_count[ih * width + iw]==0){
                    weight_val = exp(-abs(Di - Dval));
                    weight_count = depth_weight_count[ih * width + iw];
                }

                gradient += top_diff_slice[ph * pooled_width + pw] * weight_val / weight_count;
            }
        }

        bottom_diff[index] = Dtype(gradient);
    }
}

void avgpool_forward(int count, torch::Tensor input_data, torch::Tensor input_depth_data, int channels, int height, int width, int pooled_height, int pooled_width, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w, torch::Tensor top_data, torch::Tensor depth_weight_count) {
    int num_blocks = (count + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
    
    AT_DISPATCH_FLOATING_TYPES(input_data.type(), "avgpool_forward", ([&] {
        avgpool_forward_kernel<scalar_t, scalar_t><<<num_blocks, CUDA_NUM_THREADS>>>(
            count,
            input_data.data<scalar_t>(),
            input_depth_data.data<scalar_t>(),
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            top_data.data<scalar_t>(),
            depth_weight_count.data<scalar_t>()
        );
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in avgpool_forward: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
}

void avgpool_backward(int count, torch::Tensor grad_output, torch::Tensor input_depth, torch::Tensor depth_weight_count, int channels, int height, int width, int pooled_height, int pooled_width, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w, torch::Tensor bottom_diff) {
    int num_blocks = (count + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;

    AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "avgpool_backward", ([&] {
        avgpool_backward_kernel<scalar_t, scalar_t><<<num_blocks, CUDA_NUM_THREADS>>>(
            count,
            grad_output.data<scalar_t>(),
            input_depth.data<scalar_t>(),
            depth_weight_count.data<scalar_t>(),
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            bottom_diff.data<scalar_t>()
        );
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in avgpool_backward: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
}
