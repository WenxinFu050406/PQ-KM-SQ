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

    centroid_vec = []
    temp_index = index
    for i in range(nsubq):
        cent_idx = temp_index % pq_centroids[i].shape[0]
        centroid_vec.extend(pq_centroids[i][cent_idx])
        temp_index = temp_index // pq_centroids[i].shape[0]
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
    reconstructed = pq.decode(pq_codes)  # 直接接收解码结果
    reconstructed = torch.from_numpy(reconstructed).to(kernels.device).type(kernels.dtype)
    return reconstructed


def compress_torch_weights(model_path, compressed_model_path, nsubq=3, nbits=3):
    """
    使用FAISS的Product Quantization压缩模型权重
    nsubq: 子空间数量
    nbits: 每个子空间的编码位数（聚类数 = 2^nbits，需满足 nbits ≤24）
    """
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    weights = checkpoint

    # 拼接权重核
    K = wgt.concat_weights(weights)
    N = K.shape[0]
    original_dim = K.shape[1]
    # 计算实际聚类数（2^nbits）
    ncentroids = 2 ** nbits
    print(f"总核数量: {N}, 原始维度: {original_dim}, 子空间数: {nsubq}, "
          f"编码位数: {nbits}, 聚类数: {ncentroids}")

    # 校验子空间数量合法性
    if original_dim % nsubq != 0:
        raise ValueError(f"子空间数量 {nsubq} 不能整除原始维度 {original_dim}")
    subvec_len = original_dim // nsubq

    # 强制限制nbits不超过24（FAISS的硬性要求）
    if nbits > 24:
        nbits = 24
        ncentroids = 2 **24
        print(f"警告：编码位数超过24，自动限制为24，聚类数调整为 {ncentroids}")

    # 转换为FAISS要求的格式
    all_kernels_np = K.cpu().numpy().astype(np.float32)

    # 训练PQ模型（第三个参数是nbits，而非聚类数！）
    pq = faiss.ProductQuantizer(original_dim, nsubq, nbits)  # 核心修正
    pq.train(all_kernels_np)

    # 生成编码并重构
    pq_codes = pq.compute_codes(all_kernels_np)
    compressed_K = apply_pq_to_kernels(K, pq, pq_codes)

    # 重建权重字典并保存
    compressed_dict = wgt.reassign_weights(compressed_K, weights)
    compressed_weights = dict(
        state_dict=compressed_dict
    )
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
        nbits=nbits,
        ncentroids=ncentroids,
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
    json_file_path = os.path.join(save_dir, f'result_{nsubq}subq_{nbits}bits.json')
    with open(json_file_path, 'w') as f:
        json.dump(result_dict, f, indent=4)


def demo_visualize_clusters(nsubq=3, nbits=3):
    """可视化PQ聚类结果"""
    os.makedirs('pq_centroid_plots', exist_ok=True)
    try:
        checkpoint = torch.load('model_best.pth.tar', map_location='cpu')
    except:
        checkpoint = torch.load('vgg19.pth', map_location='cpu')
    
    weights = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    K = wgt.concat_weights(weights)[:1000]
    original_dim = K.shape[1]

    if original_dim % nsubq != 0:
        raise ValueError(f"子空间数量 {nsubq} 不能整除原始维度 {original_dim}")
    subvec_len = original_dim // nsubq
    ncentroids = 2** nbits
    print(f"可视化参数: 子空间={nsubq}, 编码位数={nbits}, 聚类数={ncentroids} (2^{nbits})")

    # 训练PQ（第三个参数是nbits）
    kernel_np = K.cpu().numpy().astype(np.float32)
    pq = faiss.ProductQuantizer(original_dim, nsubq, nbits)
    pq.train(kernel_np)
    pq_codes = pq.compute_codes(kernel_np)

    # 提取质心
    centroids = []
    for i in range(nsubq):
        cent_arr = faiss.vector_to_array(pq.centroids[i])
        centroids.append(cent_arr.reshape(ncentroids, subvec_len))

    # 可视化前5种质心组合
    for i in range(min(5, ncentroids ** nsubq)):
        plot_vectors_centroid(i, K, centroids, pq_codes, subvec_len, nsubq)
    print("可视化完成，结果保存至 pq_centroid_plots 目录")


def main():
    if len(sys.argv) != 3:
        print("用法: python faissPQ.py <子空间数量> <每个子空间编码位数>")
        print("示例: python faissPQ.py 3 5 （3个子空间，每个子空间5位编码，聚类数=2^5=32）")
        sys.exit(1)

    # 解析参数（第二个参数是nbits，而非聚类数）
    try:
        nsubq = int(sys.argv[1])
        nbits = int(sys.argv[2])  # 编码位数
    except ValueError:
        print("错误：子空间数量和编码位数必须为整数")
        sys.exit(1)

    # 计算聚类数（用于文件名）
    ncentroids = 2 ** nbits
    model_path = 'vgg19.pth'
    compressed_model_path = os.path.join(
        './compressed_models',
        f'vgg19_faiss_pq_{nsubq}subq_{nbits}bits_{ncentroids}cent.pth'
    )

    # 执行压缩
    t0 = time.time()
    compress_torch_weights(
        model_path,
        compressed_model_path,
        nsubq=nsubq,
        nbits=nbits  # 传入编码位数
    )
    t1 = time.time()

    print(f'处理时间: {t1 - t0:.2f} 秒')
    print(f'子空间数量: {nsubq}, 编码位数: {nbits}, 聚类数: {ncentroids}')


if __name__ == '__main__':
    # 如需可视化，取消下面一行注释
    # demo_visualize_clusters(nsubq=3, nbits=5)
    main()