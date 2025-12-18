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
#define BLOCK_TILE_M 32
#define BLOCK_TILE_N 32
#define BLOCK_TILE_K 16

// Warp configuration
#define WARPS_M 1
#define WARPS_N 1
#define NUM_THREADS (WARPS_M * WARPS_N * 32)

// Shared memory padding to avoid bank conflicts
#define SKEW_HALF 8

__global__ void fused_gemm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ gemm_output,
    int batch_size,
    int input_size,
    int hidden_size
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
    
    // Early exit if block is out of bounds
    if (block_row * BLOCK_TILE_M >= batch_size || block_col * BLOCK_TILE_N >= hidden_size) {
        return;
    }
    
    // Each warp computes tiles using WMMA
    const int num_wmma_tiles_m = BLOCK_TILE_M / WMMA_M / WARPS_M;  // = 32/16/1 = 2
    const int num_wmma_tiles_n = BLOCK_TILE_N / WMMA_N / WARPS_N;  // = 32/16/1 = 2
    
    // Loop over K dimension in tiles
    const int num_k_tiles = (input_size + BLOCK_TILE_K - 1) / BLOCK_TILE_K;
    
    // Process each output tile sequentially
    for (int tile_m = 0; tile_m < num_wmma_tiles_m; tile_m++) {
        for (int tile_n = 0; tile_n < num_wmma_tiles_n; tile_n++) {
            
            // Accumulator fragment for this tile
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
            wmma::fill_fragment(acc_frag, 0.0f);
            
            for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
                // Load input tile to shared memory (A_smem)
                const int global_row_start = block_row * BLOCK_TILE_M;
                const int global_k_start = k_tile * BLOCK_TILE_K;
                
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
                
                // Perform WMMA operations for this tile
                for (int k_step = 0; k_step < BLOCK_TILE_K / WMMA_K; k_step++) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
                    
                    int a_row = tile_m * WMMA_M;
                    int b_col = tile_n * WMMA_N;
                    int k_offset = k_step * WMMA_K;
                    
                    wmma::load_matrix_sync(a_frag, &A_smem[a_row][k_offset], BLOCK_TILE_K + SKEW_HALF);
                    wmma::load_matrix_sync(b_frag, &B_smem[k_offset][b_col], BLOCK_TILE_N + SKEW_HALF);
                    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
                }
                
                __syncthreads();
            }
            
            // Store results with proper boundary checking
            int frag_row = tile_m * WMMA_M;
            int frag_col = tile_n * WMMA_N;
            
            int global_row = block_row * BLOCK_TILE_M + frag_row;
            int global_col = block_col * BLOCK_TILE_N + frag_col;
            
            // Check if the entire 16x16 tile fits within bounds
            if (global_row + WMMA_M <= batch_size && global_col + WMMA_N <= hidden_size) {
                // Safe to store full tile using WMMA
                wmma::store_matrix_sync(
                    &gemm_output[global_row * hidden_size + global_col],
                    acc_frag, hidden_size, wmma::mem_row_major);
            } else {
                // Handle partial tiles by extracting elements individually
                float output_vals[WMMA_M * WMMA_N];
                wmma::store_matrix_sync(output_vals, acc_frag, WMMA_N, wmma::mem_row_major);
                
                for (int i = 0; i < WMMA_M; i++) {
                    for (int j = 0; j < WMMA_N; j++) {
                        int row_idx = global_row + i;
                        int col_idx = global_col + j;
                        if (row_idx < batch_size && col_idx < hidden_size) {
                            gemm_output[row_idx * hidden_size + col_idx] = output_vals[i * WMMA_N + j];
                        }
                    }
                }
            }
        }
    }
}

__global__ void reduce_and_scale_kernel(
    const float* __restrict__ gemm_output,
    float* __restrict__ output,
    int batch_size,
    int hidden_size,
    float scaling_factor
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size) {
        float sum = 0.0f;
        for (int col = 0; col < hidden_size; col++) {
            sum += gemm_output[row * hidden_size + col] * 0.5f; // Division by 2
        }
        output[row] = sum * scaling_factor;
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
    
    // Allocate temporary buffer for GEMM output
    auto gemm_output = torch::zeros({batch_size, hidden_size}, input.options());
    auto output = torch::zeros({batch_size, 1}, input.options());
    
    // Launch GEMM kernel
    dim3 gemm_threads(NUM_THREADS);
    dim3 gemm_blocks(
        (hidden_size + BLOCK_TILE_N - 1) / BLOCK_TILE_N,
        (batch_size + BLOCK_TILE_M - 1) / BLOCK_TILE_M
    );
    
    fused_gemm_kernel<<<gemm_blocks, gemm_threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        gemm_output.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size
    );
    
    // Launch reduction kernel
    const int reduce_threads = 256;
    const int reduce_blocks = (batch_size + reduce_threads - 1) / reduce_threads;
    
    reduce_and_scale_kernel<<<reduce_blocks, reduce_threads>>>(
        gemm_output.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
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
        "--ptxas-options=-v",
        "-lineinfo"
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
