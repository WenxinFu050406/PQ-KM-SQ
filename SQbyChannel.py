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

    def quantize_per_input_channel(self, kernels, n_bits=4):
        """明确按输入通道（cin）量化（卷积核维度为[cout, cin, k_h, k_w]）"""
        quantized_kernels = torch.zeros_like(kernels)
        cin = kernels.shape[1]  # 卷积核第1维为输入通道数cin
        
        with torch.no_grad():
            # 循环输入通道维度（cin），每个输入通道单独量化
            for c_in in range(cin):
                # 提取所有输出通道中属于当前输入通道的核参数
                # 形状为 [cout, 1, k_h, k_w]
                channel_data = kernels[:, c_in:c_in+1, ...]
                quantized_channel = self.quantize_tensor(channel_data, n_bits)
                quantized_kernels[:, c_in:c_in+1, ...] = quantized_channel
                
        # 计算量化误差
        mse = torch.mean((kernels - quantized_kernels) **2).item()
        return quantized_kernels, mse


def compress_torch_weights(model_path, compressed_model_path, n_bits=4):
    """应用按输入通道（cin）的量化逻辑（不依赖concat_weights）"""
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    weights = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # 初始化量化器
    quantizer = ScalarQuantizer()
    
    # 直接遍历原始权重字典，筛选卷积层
    quantized_weights = {}  # 存储量化后的权重（按层名组织）
    total_mse = 0.0
    conv_layer_count = 0

    for layer_name, weight_tensor in weights.items():
        # 仅处理卷积层权重（形状为 [cout, cin, k_h, k_w]）
        if 'features' in layer_name and len(weight_tensor.shape) == 4:
            print(f" {layer_name}, shape: {weight_tensor.shape}")
            # 按输入通道（cin）量化当前层的卷积核
            quantized_kernel, mse = quantizer.quantize_per_input_channel(weight_tensor, n_bits)
            quantized_weights[layer_name] = quantized_kernel
            total_mse += mse
            conv_layer_count += 1
            print(f" {layer_name} 's MSE: {mse:.6f}")
        else:
            # 非卷积层权重不量化，直接保留
            quantized_weights[layer_name] = weight_tensor

    # 计算平均MSE
    avg_mse = total_mse / conv_layer_count if conv_layer_count > 0 else 0.0
    print(f"Avg MSE: {avg_mse:.6f}")

    # 保存压缩模型
    compressed_weights = {'state_dict': quantized_weights}
    os.makedirs(os.path.dirname(compressed_model_path), exist_ok=True)
    torch.save(compressed_weights, compressed_model_path)
    
    # 验证量化模型
    top1_avg = validate_vgg19_cifar10(compressed_model_path)

    # 保存结果
    result_dict = dict(
        timestamp=now(),
        precision=top1_avg,
        quantization_bits=n_bits,
        average_quantization_mse=avg_mse,
        model_name='vgg19',
        task='classification',
        dataset='cifar10',
        compression_type_base='per_input_channel_quantization',
        compressed_model_name=compressed_model_path
    )
    
    save_dir = './results'
    os.makedirs(save_dir, exist_ok=True)
    json_file_path = os.path.join(save_dir, f'result_{n_bits}bits_per_cin.json')
    with open(json_file_path, 'w') as f:
        json.dump(result_dict, f, indent=4)
    
    return top1_avg, avg_mse


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
    model_path = 'vgg19.pth'  # 与SQbyKernel保持一致的模型路径
    
    if len(sys.argv) > 1:
        n_bits = int(sys.argv[1])
    else:
        n_bits = 4  # Default to 4-bit quantization
    
    compressed_model_path = f'./compressed_models/vgg19_per_input_channel_{n_bits}bit.pth'

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