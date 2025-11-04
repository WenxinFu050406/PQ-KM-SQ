import os
import sys
import time
import torch
import torch.nn as nn
import pprint
import numpy as np
import pretty_errors
import tqdm
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


def compress_torch_weights(model_path, compressed_model_path, kmeans_batch_size=10000, num_clusters=200):
    """
    opens the pytorch model weights and applies faiss kmeans to the kernels
    saves the compressed weights to disk
    """
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    weights = checkpoint

    K = wgt.concat_weights(weights)
    compressed_K = torch.zeros_like(K)

    N = K.shape[0]
    batch_size = kmeans_batch_size


    i = 0
    while i * batch_size < N:
        print(i * batch_size, N)
        i0 = i * batch_size
        i1 = i0 + batch_size
        batch = K[i0: i1]
        print(f'[{i0}/{N}] ({i1/N * 100:.2f}/100)')

        # 转换为numpy数组并调整维度（faiss需要float32类型的二维数组）
        batch_np = batch.cpu().numpy().astype(np.float32)
        d = batch_np.shape[1]  # 特征维度

        # 初始化faiss kmeans
        kmeans = faiss.Kmeans(d, num_clusters, niter=20, verbose=False)
        # 训练聚类
        kmeans.train(batch_np)
        # 获取聚类结果
        kmeans_indexes = kmeans.assign(batch_np)[1]  # 第二个元素是聚类索引
        kmeans_centroids = kmeans.centroids  # 聚类中心

        # 应用聚类结果
        compressed_batch = apply_kmeans_to_kernels(batch, kmeans_centroids, kmeans_indexes)
        compressed_K[i0: i1] = compressed_batch

        i += 1


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
        kmeans_batch_size=batch_size,
        model_name='vgg19',
        task='classification',
        dataset='cifar10',
        compression_type_base='faiss_kmeans',  # 更新压缩类型标识
        compression_type='faiss_kmeans_no_centroid_rescale',
        compressed_model_name=compressed_model_path
    )
    save_dir = './results'
    os.makedirs(save_dir, exist_ok=True)
    json_file_path = os.path.join(save_dir, f'result_{num_clusters}_{kmeans_batch_size}.json')
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
    d = kernel_as_vector.shape[1]  # 特征维度

    # 初始化并训练faiss kmeans
    kmeans = faiss.Kmeans(d, num_clusters, niter=20, verbose=True)
    kmeans.train(kernel_as_vector)
    # 获取聚类索引和中心
    cluster_ids_x = kmeans.assign(kernel_as_vector)[1]
    cluster_centers = kmeans.centroids

    print('finished faiss k-means clustering')

    for i in range(num_clusters):
        print(i)
        plot_vectors_centroid(i, K, cluster_centers, cluster_ids_x)


def main():
    model_path = 'vgg19.pth'
    
    num_clusters = int(sys.argv[1])
    N = int(sys.argv[2])
    assert num_clusters < N
    compressed_model_path = f'./compressed_models/vgg19_faiss_kmeans_centroid_cifar10_{num_clusters}_{N}.pth'  # 更新文件名

    t0 = time.time()
    
    compress_torch_weights(
        model_path, compressed_model_path, 
        kmeans_batch_size=N, num_clusters=num_clusters
    )
    t1 = time.time()
    print(f'time: {t1 - t0:.2f} seconds')
    print(f'num clusters: {num_clusters}/{N}')


if __name__ == '__main__':
    main()