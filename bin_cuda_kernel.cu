#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

template <typename scalar_t>
__global__ void bin_cuda_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,size_t> avg_pools,
    const int B, 
    const int sub_desc_dim, 
    const int num_bins,
    const int num_patches0,
    const int num_patches1,
    const int hierarchy,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> bin_x) {

    //
    //batch index
    int batch = threadIdx.y;
    // int batch = 0;
    int y = blockIdx.x;
    int x = blockIdx.y;

    int thread_index = threadIdx.x;
    
    while (y < num_patches0)
    {
        while (x < num_patches1)
        {
            int part_idx = 0;
            int kernel_size = 1;
            for (int k = 0; k < hierarchy; k++)
            {
                if (k > 0)
                {
                    kernel_size *= 3;
                }
                
                for (int i = (y - kernel_size); i < (y + kernel_size + 1); i += kernel_size)
                {
                    for (int j = (x - kernel_size); j < (x + kernel_size + 1); j += kernel_size)
                    {
                        if (i == y && j == x && k != 0)
                        {
                            continue;
                        }
                        if (0 <= i && i < num_patches0 && 0 <= j && j < num_patches1)
                        {
                            batch = threadIdx.y;

                            while (batch < B)
                            {
                                thread_index = threadIdx.x;
                                while (thread_index < sub_desc_dim)
                                {
                                    bin_x[batch][part_idx * sub_desc_dim + thread_index][y][x] = avg_pools[k][batch][thread_index][i][j];
                                    thread_index += blockDim.x;
                                }
                                batch += blockDim.y;

                            }
                        }
                        else
                        {
                            batch = threadIdx.y;

                            thread_index = threadIdx.x;
                            int temp_i = max(0, min(i, num_patches0 - 1));
                            int temp_j = max(0, min(j, num_patches1 - 1));
                            while (batch < B)
                            {
                                thread_index = threadIdx.x;
                                while (thread_index < sub_desc_dim)
                                {
                                    bin_x[batch][part_idx * sub_desc_dim + thread_index][y][x] = avg_pools[k][batch][thread_index][temp_i][temp_j];
                                    thread_index += blockDim.x;
                                }
                                batch += blockDim.y;
                            }
                        }
                        part_idx ++;
                    }
                }
            }
            x += gridDim.y;
        }
        y += gridDim.x;
    }
}

} // namespace

std::vector<torch::Tensor> bin_cuda_forward(
    torch::Tensor avg_pools,
    int B,
    int sub_desc_dim,
    int num_bins,
    int num_patches0,
    int num_patches1,
    int hierarchy) {
    auto bin_x = torch::zeros({B, sub_desc_dim * num_bins, num_patches0, num_patches1}).to(avg_pools.device());

//   const int threads = 64;
    // const dim3 threads(sub_desc_dim, B);
    const dim3 threads(sub_desc_dim, 1);
    const dim3 blocks(num_patches0, num_patches1);

    AT_DISPATCH_FLOATING_TYPES(avg_pools.type(), "bin_forward_cuda", ([&] {
    bin_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        avg_pools.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,size_t>(),
        B,
        sub_desc_dim,
        num_bins,
        num_patches0,
        num_patches1,
        hierarchy,
        bin_x.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>());
    }));

    return {bin_x};
}

std::vector<torch::Tensor> bin_cuda_backward(
    torch::Tensor grad_bin_x) {
  return {grad_bin_x};
}
