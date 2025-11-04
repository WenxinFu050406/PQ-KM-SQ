import os
import sys
import time
import torch
import numpy as np
import json
import faiss  

import weights as wgt
from utils.timingtools import now
from exp_vgg import validate_vgg19_cifar10


def plot_vectors_centroid(index, original_kernels, pq_centroids, pq_codes, subvec_len, nsubq):
    import seaborn as sns
    import matplotlib.pyplot as plt

    # 随机选取部分样本可视化
    sample_indexes = np.random.choice(len(pq_codes), min(20, len(pq_codes)), replace=False)
    print(f'质心组合 {index} 相关样本数量: {len(sample_indexes)}')

    plt.clf()
    for k in sample_indexes:
        sns.lineplot(
            x=list(range(original_kernels.shape[1])),
            y=original_kernels[k].cpu().numpy(),
            color='blue',
            alpha=0.3
        )

    # 重构质心向量
    centroid_vec = []
    temp_index = index  # 使用临时变量避免修改原index
    for i in range(nsubq):
        cent_idx = temp_index % pq_centroids[i].shape[0]
        centroid_vec.extend(pq_centroids[i][cent_idx])
        temp_index = temp_index // pq_centroids[i].shape[0]
    # 截断到原始维度
    centroid_vec = np.array(centroid_vec[:original_kernels.shape[1]])
    
    sns.lineplot(
        x=list(range(original_kernels.shape[1])),
        y=centroid_vec,
        color='red',
        linewidth=2
    )
    plt.title(f'PQ 质心组合可视化 (子空间数: {nsubq})')
    os.makedirs('pq_centroid_plots', exist_ok=True)
    plt.savefig(f'pq_centroid_plots/centroid_combination_{index}.jpg')
    plt.close()


def apply_pq_to_kernels(kernels, pq, pq_codes):
    """应用PQ解码重构核权重（适配旧版本FAISS）"""
    # 直接调用decode，仅传递编码，接收返回的重构结果
    reconstructed = pq.decode(pq_codes)  # 旧版本FAISS的正确用法
    # 转换为与原始kernels一致的设备和数据类型
    reconstructed = torch.from_numpy(reconstructed).to(kernels.device).type(kernels.dtype)
    return reconstructed


def compress_torch_weights(model_path, compressed_model_path, nsubq=3, ncentroids=8):
    """
    使用FAISS的Product Quantization压缩模型权重
    nsubq: 子空间数量
    ncentroids: 每个子空间的期望聚类数（会被调整为不小于该值的最小2的整数次幂）
    """
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    weights = checkpoint

    # 拼接权重核
    K = wgt.concat_weights(weights)
    N = K.shape[0]
    original_dim = K.shape[1]
    print(f"总核数量: {N}, 原始维度: {original_dim}, 子空间数: {nsubq}, 期望聚类数: {ncentroids}")

    # 校验子空间数量合法性
    if original_dim % nsubq != 0:
        raise ValueError(f"子空间数量 {nsubq} 不能整除原始维度 {original_dim}")
    subvec_len = original_dim // nsubq

    # 调整聚类数为不小于输入值的最小2的整数次幂，且不超过2^24
    if ncentroids <= 1:
        nbits = 1
    else:
        # 计算满足2^nbits >= ncentroids的最小nbits
        nbits = int(np.ceil(np.log2(ncentroids)))
    nbits = min(nbits, 24)  # 限制最大为2^24
    adjusted_centroids = 2 ** nbits
    print(f"调整后聚类数: 2^{nbits} = {adjusted_centroids}")

    # 转换为FAISS要求的格式
    all_kernels_np = K.cpu().numpy().astype(np.float32)

    # 训练PQ模型（使用调整后的聚类数）
    pq = faiss.ProductQuantizer(original_dim, nsubq, adjusted_centroids)
    pq.train(all_kernels_np)

    # 生成编码并重构
    pq_codes = pq.compute_codes(all_kernels_np)
    compressed_K = apply_pq_to_kernels(K, pq, pq_codes)

    # 重建权重字典并保存
    compressed_dict = wgt.reassign_weights(compressed_K, weights)
    compressed_weights = dict(
        state_dict=compressed_dict
    )
    # 确保保存目录存在
    save_dir = os.path.dirname(compressed_model_path)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(compressed_weights, compressed_model_path)

    # 验证压缩模型
    top1_avg = validate_vgg19_cifar10(compressed_model_path)

    # 保存结果信息
    result_dict = dict(
        timestamp=now(),
        precision=top1_avg,
        nsubq=nsubq,
        expected_centroids=ncentroids,  # 记录原始期望的聚类数
        adjusted_centroids=adjusted_centroids,  # 记录调整后的聚类数
        nbits=nbits,
        original_dim=original_dim,
        total_kernels=N,
        model_name='vgg19',
        task='classification',
        dataset='cifar10',
        compression_type_base='faiss_pq',
        compression_type='faiss_pq_standard',
        compressed_model_name=compressed_model_path
    )
    save_dir = './results'
    os.makedirs(save_dir, exist_ok=True)
    json_file_path = os.path.join(save_dir, f'result_{nsubq}subq_{adjusted_centroids}cent.json')
    with open(json_file_path, 'w') as f:
        json.dump(result_dict, f, indent=4)


def demo_visualize_clusters(nsubq=3, ncentroids=8):
    """可视化PQ聚类结果"""
    os.makedirs('pq_centroid_plots', exist_ok=True)
    # 加载模型
    try:
        checkpoint = torch.load('model_best.pth.tar', map_location='cpu')
    except:
        checkpoint = torch.load('vgg19.pth', map_location='cpu')
    
    weights = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    K = wgt.concat_weights(weights)[:1000]  # 取部分样本
    original_dim = K.shape[1]

    # 校验子空间数量
    if original_dim % nsubq != 0:
        raise ValueError(f"子空间数量 {nsubq} 不能整除原始维度 {original_dim}")
    subvec_len = original_dim // nsubq

    # 调整聚类数为不小于输入值的最小2的整数次幂
    if ncentroids <= 1:
        nbits = 1
    else:
        nbits = int(np.ceil(np.log2(ncentroids)))
    nbits = min(nbits, 24)
    adjusted_centroids = 2 ** nbits
    print(f"可视化参数: 子空间={nsubq}, 期望聚类数={ncentroids}, 调整后聚类数={adjusted_centroids} (2^{nbits})")

    # 训练PQ（使用调整后的聚类数）
    kernel_np = K.cpu().numpy().astype(np.float32)
    pq = faiss.ProductQuantizer(original_dim, nsubq, adjusted_centroids)
    pq.train(kernel_np)
    pq_codes = pq.compute_codes(kernel_np)

    # 提取质心
    centroids = []
    for i in range(nsubq):
        cent_arr = faiss.vector_to_array(pq.centroids[i])
        centroids.append(cent_arr.reshape(adjusted_centroids, subvec_len))

    # 可视化前5种质心组合
    for i in range(min(5, adjusted_centroids ** nsubq)):
        plot_vectors_centroid(i, K, centroids, pq_codes, subvec_len, nsubq)
    print("可视化完成，结果保存至 pq_centroid_plots 目录")


def main():
   
    try:
        nsubq = 1
        ncentroids = 24
    except ValueError:
        print("错误：子空间数量和聚类数必须为整数")
        sys.exit(1)

    # 定义保存路径（使用调整后的聚类数命名文件）
    # 先计算调整后的聚类数用于文件名
    if ncentroids <= 1:
        nbits = 1
    else:
        nbits = int(np.ceil(np.log2(ncentroids)))
    nbits = min(nbits, 24)
    adjusted_centroids = 2 ** nbits
    
    model_path = 'vgg19.pth'
    compressed_model_path = os.path.join(
        './compressed_models',
        f'vgg19_faiss_pq_{nsubq}subq_{adjusted_centroids}cent.pth'
    )

    # 执行压缩
    t0 = time.time()
    compress_torch_weights(
        model_path,
        compressed_model_path,
        nsubq=nsubq,
        ncentroids=ncentroids
    )
    t1 = time.time()

    print(f'处理时间: {t1 - t0:.2f} 秒')
    print(f'子空间数量: {nsubq}, 期望聚类数: {ncentroids}, 调整后聚类数: {adjusted_centroids}')


if __name__ == '__main__':
    # 如需可视化，取消下面一行注释
    # demo_visualize_clusters(nsubq=3, ncentroids=256)
    main()