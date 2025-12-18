import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Fuse matmul + div operations
fused_matmul_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_div_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int input_size,
    int hidden_size
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < hidden_size) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; i++) {
            sum += input[row * input_size + i] * weight[col * input_size + i];
        }
        output[row * hidden_size + col] = sum * 0.5f;
    }
}

torch::Tensor matmul_div_cuda(torch::Tensor input, torch::Tensor weight) {
    int batch_size = input.size(0);
    int input_size = input.size(1);
    int hidden_size = weight.size(0);
    
    auto output = torch::zeros({batch_size, hidden_size}, input.options());
    
    dim3 threads(16, 16);
    dim3 blocks((hidden_size + threads.x - 1) / threads.x,
                (batch_size + threads.y - 1) / threads.y);
    
    matmul_div_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size
    );
    
    return output;
}
"""

fused_matmul_div_cpp_source = """
torch::Tensor matmul_div_cuda(torch::Tensor input, torch::Tensor weight);
"""

fused_matmul_div = load_inline(
    name="fused_matmul_div",
    cpp_sources=fused_matmul_div_cpp_source,
    cuda_sources=fused_matmul_div_source,
    functions=["matmul_div_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused matmul+div kernel.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor
        self.fused_matmul_div = fused_matmul_div

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        # Fused matmul + div
        x = self.fused_matmul_div.matmul_div_cuda(x, self.weight)
        # Sum along hidden dimension
        x = torch.sum(x, dim=1, keepdim=True)
        # Scale
        x = x * self.scaling_factor
        return x


batch_size = 1024
input_size = 8192
hidden_size = 8192
scaling_factor = 1.5

def get_inputs():
    return [torch.rand(batch_size, input_size).cuda()]

def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]