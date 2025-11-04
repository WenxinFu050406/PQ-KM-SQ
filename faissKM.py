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


def plot_vectors_centroid(index, original_kernels, kmeans_centroids, kmeans_indexes):
    import seaborn as sns
    import matplotlib.pyplot as plt

    kernel_indexes = np.where(kmeans_indexes == index)[0]
    print(f'found {len(kernel_indexes)} kernels to centroid of index {index}')

    kernels = []
    for i in kernel_indexes:
        kernels.append(original_kernels[i].cpu().numpy())
    

    plt.clf()
    for k in kernel_indexes:
        ax = sns.lineplot(x=list(range(9)), y=original_kernels[k].cpu().numpy(), color='blue')


    centroid = kmeans_centroids[index]
    ax = sns.lineplot(x=list(range(9)), y=centroid, color='red')
    ax.set_title(f'{len(kernel_indexes)} kernels to centroid of index {index}')
    plt.savefig(f'centroid_plots/centroid_{index}.jpg')


def apply_kmeans_to_kernels(kernels, kmeans_centroids, kmeans_indexes):
    """
    Apply kmeans to kernels using the results from faiss kmeans
    """
    
    if isinstance(kmeans_centroids, np.ndarray):
        kmeans_centroids = torch.from_numpy(kmeans_centroids).to(kernels.device)
    
    compressed_kernels = torch.zeros_like(kernels)
    n = kernels.shape[0]
    for i in range(n):
        compressed_kernels[i] = kmeans_centroids[kmeans_indexes[i]]

    return compressed_kernels


def compress_torch_weights(model_path, compressed_model_path, num_clusters=200):
    """
    打开pytorch模型权重并对所有核权重一次性应用faiss kmeans
    将压缩后的权重保存到磁盘
    """
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    weights = checkpoint

    # 拼接所有权重核
    K = wgt.concat_weights(weights)
    N = K.shape[0]
    print(f"总核数量: {N}, 聚类数量: {num_clusters}")

    # 转换为numpy数组并调整维度（faiss需要float32类型的二维数组）
    all_kernels_np = K.cpu().numpy().astype(np.float32)
    d = all_kernels_np.shape[1]  # 特征维度

    # 初始化faiss kmeans（一次性处理所有数据）
    kmeans = faiss.Kmeans(d, num_clusters, niter=20, verbose=True)
    # 训练聚类
    kmeans.train(all_kernels_np)
    # 获取聚类结果
    kmeans_indexes = kmeans.assign(all_kernels_np)[1]  # 聚类索引
    kmeans_centroids = kmeans.centroids  # 聚类中心

    # 应用聚类结果
    compressed_K = apply_kmeans_to_kernels(K, kmeans_centroids, kmeans_indexes)

    # 重建压缩后的权重字典并保存
    compressed_dict = wgt.reassign_weights(compressed_K, weights)
    compressed_weights = dict(
        state_dict=compressed_dict
    )
    torch.save(compressed_weights, compressed_model_path)
    top1_avg = validate_vgg19_cifar10(compressed_model_path)


    result_dict = dict(
        timestamp=now(),
        precision=top1_avg,
        num_clusters=num_clusters,
        model_name='vgg16',
        task='classification',
        dataset='cifar10',
        compression_type_base='faiss_kmeans',
        compression_type='faiss_kmeans_no_batch',  # 标识无分批处理
        compressed_model_name=compressed_model_path
    )
    save_dir = './results'
    os.makedirs(save_dir, exist_ok=True)
    json_file_path = os.path.join(save_dir, f'result_{num_clusters}_no_batch.json')
    with open(json_file_path, 'w') as f:
        json.dump(result_dict, f, indent=4)


def demo_visualize_clusters():
    os.makedirs('centroid_plots', exist_ok=True)
    checkpoint = torch.load('model_best.pth.tar')
    weights = checkpoint['state_dict']
    K = wgt.concat_weights(weights)[:10000]
    N = K.shape[0]
    num_clusters = 200
    print(f'{N}, {num_clusters}, {N/num_clusters:.4f}')

    # 转换为numpy数组（faiss需要float32类型）
    kernel_as_vector = K.cpu().numpy().astype(np.float32)
    d = kernel_as_vector.shape[1] 

    
    kmeans = faiss.Kmeans(d, num_clusters, niter=20, verbose=True)
    kmeans.train(kernel_as_vector)
    
    cluster_ids_x = kmeans.assign(kernel_as_vector)[1]
    cluster_centers = kmeans.centroids

    print('finished faiss k-means clustering')

    for i in range(num_clusters):
        print(i)
        plot_vectors_centroid(i, K, cluster_centers, cluster_ids_x)


def main():
    model_path = 'vgg19.pth'
    
    num_clusters = int(sys.argv[1])
    compressed_model_path = f'./compressed_models/vgg16_faiss_kmeans_{num_clusters}.pth'

    t0 = time.time()
    
    compress_torch_weights(
        model_path, compressed_model_path, 
        num_clusters=num_clusters  
    )
    t1 = time.time()
    print(f'处理时间: {t1 - t0:.2f} 秒')
    print(f'聚类数量: {num_clusters}')


if __name__ == '__main__':
    main()