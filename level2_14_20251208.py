import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

// Tile sizes for WMMA (Tensor Cores)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Block tile sizes
#define BLOCK_TILE_M 128
#define BLOCK_TILE_N 128
#define BLOCK_TILE_K 32

// Warp configuration
#define WARPS_M 4
#define WARPS_N 4
#define NUM_THREADS (WARPS_M * WARPS_N * 32)

// Shared memory padding to avoid bank conflicts
#define SKEW_HALF 8

__global__ void fused_gemm_reduce_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int input_size,
    int hidden_size,
    float scaling_factor
) {
    // Shared memory for input and weight tiles
    __shared__ half A_smem[BLOCK_TILE_M][BLOCK_TILE_K + SKEW_HALF];
    __shared__ half B_smem[BLOCK_TILE_K][BLOCK_TILE_N + SKEW_HALF];
    
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int warp_row = warp_id / WARPS_N;
    const int warp_col = warp_id % WARPS_N;
    
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;
    
    // Each warp computes a 64x64 tile using 4x4 WMMA tiles
    const int num_wmma_tiles_m = BLOCK_TILE_M / WMMA_M / WARPS_M;
    const int num_wmma_tiles_n = BLOCK_TILE_N / WMMA_N / WARPS_N;
    
    // Accumulator fragments for this warp
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frags[num_wmma_tiles_m][num_wmma_tiles_n];
    
    // Initialize accumulators to zero
    #pragma unroll
    for (int i = 0; i < num_wmma_tiles_m; i++) {
        #pragma unroll
        for (int j = 0; j < num_wmma_tiles_n; j++) {
            wmma::fill_fragment(acc_frags[i][j], 0.0f);
        }
    }
    
    // Loop over K dimension in tiles
    const int num_k_tiles = (input_size + BLOCK_TILE_K - 1) / BLOCK_TILE_K;
    
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        // Load input tile to shared memory (A_smem)
        // Each thread loads multiple elements
        const int global_row_start = block_row * BLOCK_TILE_M;
        const int global_k_start = k_tile * BLOCK_TILE_K;
        
        #pragma unroll
        for (int i = threadIdx.x; i < BLOCK_TILE_M * BLOCK_TILE_K; i += NUM_THREADS) {
            int local_row = i / BLOCK_TILE_K;
            int local_k = i % BLOCK_TILE_K;
            int global_row = global_row_start + local_row;
            int global_k = global_k_start + local_k;
            
            float val = 0.0f;
            if (global_row < batch_size && global_k < input_size) {
                val = input[global_row * input_size + global_k];
            }
            A_smem[local_row][local_k] = __float2half(val);
        }
        
        // Load weight tile to shared memory (B_smem)
        // Weight is [hidden_size, input_size], we need transpose
        const int global_col_start = block_col * BLOCK_TILE_N;
        
        #pragma unroll
        for (int i = threadIdx.x; i < BLOCK_TILE_K * BLOCK_TILE_N; i += NUM_THREADS) {
            int local_k = i / BLOCK_TILE_N;
            int local_col = i % BLOCK_TILE_N;
            int global_k = global_k_start + local_k;
            int global_col = global_col_start + local_col;
            
            float val = 0.0f;
            if (global_col < hidden_size && global_k < input_size) {
                val = weight[global_col * input_size + global_k];
            }
            B_smem[local_k][local_col] = __float2half(val);
        }
        
        __syncthreads();
        
        // Perform WMMA operations
        #pragma unroll
        for (int k_step = 0; k_step < BLOCK_TILE_K / WMMA_K; k_step++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
            
            #pragma unroll
            for (int i = 0; i < num_wmma_tiles_m; i++) {
                #pragma unroll
                for (int j = 0; j < num_wmma_tiles_n; j++) {
                    int a_row = warp_row * num_wmma_tiles_m * WMMA_M + i * WMMA_M;
                    int b_col = warp_col * num_wmma_tiles_n * WMMA_N + j * WMMA_N;
                    int k_offset = k_step * WMMA_K;
                    
                    wmma::load_matrix_sync(a_frag, &A_smem[a_row][k_offset], BLOCK_TILE_K + SKEW_HALF);
                    wmma::load_matrix_sync(b_frag, &B_smem[k_offset][b_col], BLOCK_TILE_N + SKEW_HALF);
                    wmma::mma_sync(acc_frags[i][j], a_frag, b_frag, acc_frags[i][j]);
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store results and perform reduction + scaling
    // Each warp writes its tiles and performs partial reduction
    __shared__ float reduction_smem[BLOCK_TILE_M];
    
    if (threadIdx.x < BLOCK_TILE_M) {
        reduction_smem[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    
    #pragma unroll
    for (int i = 0; i < num_wmma_tiles_m; i++) {
        #pragma unroll
        for (int j = 0; j < num_wmma_tiles_n; j++) {
            int global_row = block_row * BLOCK_TILE_M + warp_row * num_wmma_tiles_m * WMMA_M + i * WMMA_M;
            int global_col = block_col * BLOCK_TILE_N + warp_col * num_wmma_tiles_n * WMMA_N + j * WMMA_N;
            
            if (global_row < batch_size && global_col < hidden_size) {
                // Apply division by 2 and accumulate for reduction
                float frag_sum = 0.0f;
                #pragma unroll
                for (int elem = 0; elem < acc_frags[i][j].num_elements; elem++) {
                    frag_sum += acc_frags[i][j].x[elem] * 0.5f;
                }
                
                // Atomic add to shared memory for reduction
                atomicAdd(&reduction_smem[warp_row * num_wmma_tiles_m * WMMA_M + i * WMMA_M + lane_id / 4], frag_sum);
            }
        }
    }
    
    __syncthreads();
    
    // Final reduction and write output
    if (threadIdx.x < BLOCK_TILE_M) {
        int global_row = block_row * BLOCK_TILE_M + threadIdx.x;
        if (global_row < batch_size) {
            atomicAdd(&output[global_row], reduction_smem[threadIdx.x] * scaling_factor);
        }
    }
}

torch::Tensor fused_gemm_reduce_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    float scaling_factor
) {
    const int batch_size = input.size(0);
    const int input_size = input.size(1);
    const int hidden_size = weight.size(0);
    
    auto output = torch::zeros({batch_size, 1}, input.options());
    
    dim3 threads(NUM_THREADS);
    dim3 blocks(
        (hidden_size + BLOCK_TILE_N - 1) / BLOCK_TILE_N,
        (batch_size + BLOCK_TILE_M - 1) / BLOCK_TILE_M
    );
    
    fused_gemm_reduce_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size,
        scaling_factor
    );
    
    return output;
}
"""

cpp_source = """
torch::Tensor fused_gemm_reduce_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    float scaling_factor
);
"""

fused_module = load_inline(
    name="fused_gemm_reduce",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_gemm_reduce_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-arch=sm_80",
        "--ptxas-options=-v"
    ],
)

class ModelNew(nn.Module):
    """
    Optimized model with fully fused GEMM+reduce kernel using Tensor Cores.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        return fused_module.fused_gemm_reduce_cuda(x, self.weight, self.scaling_factor)


batch_size = 1024
input_size = 8192
hidden_size = 8192
scaling_factor = 1.5

def get_inputs():
    return [torch.rand(batch_size, input_size).cuda()]

def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]
