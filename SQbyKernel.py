import os
import sys
import time
import torch
import numpy as np
import json
import torch.nn as nn

from utils.timingtools import now
from exp_vgg import validate_vgg19_cifar10


class ScalarQuantizer:
    def quantize_tensor(self, tensor, n_bits, symmetric=True):
        """Quantize tensor to specified bit precision"""
        if symmetric:
            max_val = torch.max(torch.abs(tensor))
            scale = max_val / (2**(n_bits-1) - 1) if max_val > 0 else 1.0
            quantized = torch.clamp(torch.round(tensor / scale), 
                                  -2**(n_bits-1), 2**(n_bits-1)-1)
            return quantized * scale
        else:
            min_val = torch.min(tensor)
            max_val = torch.max(tensor)
            scale = (max_val - min_val) / (2**n_bits - 1) if (max_val - min_val) > 0 else 1.0
            quantized = torch.clamp(torch.round((tensor - min_val) / scale), 
                                  0, 2**n_bits-1)
            return quantized * scale + min_val

    def quantize_kernels(self, kernels, n_bits=4):
        """Quantize all kernels to specified bit precision"""
        quantized_kernels = torch.zeros_like(kernels)
        n = kernels.shape[0]
        
        with torch.no_grad():
            for i in range(n):
                quantized_kernels[i] = self.quantize_tensor(kernels[i], n_bits)
                
        # Calculate MSE
        mse = torch.mean((kernels - quantized_kernels) **2).item()
        return quantized_kernels, mse


def compress_torch_weights(model_path, compressed_model_path, n_bits=4):
    """
    Apply scalar quantization to VGG model weights
    Save compressed weights to disk
    """
    # Load model weights
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    weights = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # Initialize quantizer
    quantizer = ScalarQuantizer()
    
    # Concatenate all weight kernels
    from weights import concat_weights  # Reuse weight concatenation function
    K = concat_weights(weights)
    N = K.shape[0]
    print(f"Total kernels: {N}, Quantization bits: {n_bits}")

    # Apply quantization
    quantized_K, mse = quantizer.quantize_kernels(K, n_bits)
    print(f"Quantization MSE: {mse:.6f}")

    # Reconstruct compressed weights dictionary
    from weights import reassign_weights  # Reuse weight reassignment function
    compressed_dict = reassign_weights(quantized_K, weights)
    compressed_weights = dict(
        state_dict=compressed_dict
    )
    
    # Save compressed model
    os.makedirs(os.path.dirname(compressed_model_path), exist_ok=True)
    torch.save(compressed_weights, compressed_model_path)
    
    # Validate quantized model
    top1_avg = validate_vgg19_cifar10(compressed_model_path)

    # Save results
    result_dict = dict(
        timestamp=now(),
        precision=top1_avg,
        quantization_bits=n_bits,
        quantization_mse=mse,
        model_name='vgg19',
        task='classification',
        dataset='cifar10',
        compression_type_base='scalar_quantization',
        compressed_model_name=compressed_model_path
    )
    
    save_dir = './results'
    os.makedirs(save_dir, exist_ok=True)
    json_file_path = os.path.join(save_dir, f'result_{n_bits}bits_scalar.json')
    with open(json_file_path, 'w') as f:
        json.dump(result_dict, f, indent=4)
    
    return top1_avg, mse


def simulate_activation_quantization(model, n_bits=4):
    """Add hooks to simulate activation quantization in VGG model"""
    quantizer = ScalarQuantizer()
    hooks = []
    
    def quant_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            return quantizer.quantize_tensor(output, n_bits)
        return output
    
    # Add hooks to convolutional and linear layers
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(quant_hook))
    
    return model, hooks


def main():
    model_path = 'vgg11.pth'
    
    if len(sys.argv) > 1:
        n_bits = int(sys.argv[1])
    else:
        n_bits = 4  # Default to 4-bit quantization
    
    compressed_model_path = f'./compressed_models/vgg11_scalar_{n_bits}bit.pth'

    t0 = time.time()
    
    acc, mse = compress_torch_weights(
        model_path, compressed_model_path, 
        n_bits=n_bits  
    )
    
    t1 = time.time()
    print(f'Processing time: {t1 - t0:.2f} seconds')
    print(f'Quantization bits: {n_bits}')
    print(f'Top-1 Accuracy: {acc:.4f}')
    print(f'Quantization MSE: {mse:.6f}')


if __name__ == '__main__':
    main()