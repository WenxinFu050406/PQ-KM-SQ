"""
weight tensor transformation

"""

import os
from os.path import join
from pathlib import Path
import sys
import time
import torch
import torch.nn as nn
from torch import norm
import pprint
import numpy as np
import faiss  # 导入faiss库

from exp_vgg import validate_vgg19_cifar10
from utils import ORJSONIO, now


def split_parent_name(path):
    p = Path(path)
    return p.parent, p.name


def contains_keywords(string, keywords=None):
    """
    returns True if and only if all the keywrds are a substring of string
    """
    if keywords is None:
        return True
    
    for kw in keywords:
        if kw not in string:
            return False
    
    return True


def concat_weights(weights, keywords=['weight', 'features.']):
    """
    Concatenates the weights of the model from shape (a, b, c, d) to (a*b, c*d)
    :param weights:
    keywords: 
        vgg19_keywords = ['features.', 'weight']
        rcnn_keywords = ['.conv', '.weight']
    :return:
    """
    keys = list(weights.keys())
    values = list(weights.values())

    tensor_shapes = []
    flat_tensors = []
    for k, v in zip(keys, values):

        if len(v.shape) == 4:
            c0, c1, k0, k1 = v.shape
        if contains_keywords(k, keywords=keywords):
            a, b, c, d = v.shape
            tensor_shapes.append((a, b, c, d))
            reshaped = v.reshape(a * b, c * d)
            flat_tensors.append(reshaped)

    concat = torch.concatenate(flat_tensors, axis=0)
    return concat


def reassign_weights(concat, weights, keywords=['weight', 'features.'], kernel_size=3):
    """
    Reassigns the weights of the model from the concatenated weights concat from shape (a*b, c*d) to (a, b, c, d)
    :param concat:
    :return:
    """
    concat = concat.reshape(-1, kernel_size, kernel_size)
    offset = 0
    result = dict()
    for k, v in weights.items():
        result[k] = v

        if len(v.shape) == 4:
            c0, c1, k0, k1 = v.shape
        if contains_keywords(k, keywords=keywords):
            print('reassign', k, v.shape)
            a, b, c, d = v.shape
            current_size = a * b
            reshaped = concat[offset: offset + current_size].reshape(a, b, c, d)
            offset += current_size
            result[k] = reshaped

    return result


def compare_weights(w1, w2, verbose=0):
    """
    Compares the weights of two models
    verbose: 0 for silent, 1 for detailed
    :param w1:
    :param w2:
    :return:
    """
    print('compare conv weights:')
    error_dict = dict()

    c = 0
    is_same_tensor = True
    acc = 0
    cnt = 0
    for (k1, v1), (k2, v2) in zip(w1.items(), w2.items()):
        error_dict[k1] = torch.mean((v1 - v2)**2).cpu().numpy()
        acc += error_dict[k1]
        cnt += 1
        if k1.endswith('weight') and k1.startswith('features.'):
            assert k1 == k2, f'{k1} != {k2}'
            cond = torch.allclose(v1, v2)
            if not cond:
                print(f'[{c:03}] {k1}: tensor mismatch')
                is_same_tensor = False
            c += 1
    error = acc / cnt
    return is_same_tensor, error, error_dict


def apply_kmeans_to_kernels(kernels, kmeans_centroids, kmeans_indexes):
    """
    Apply kmeans to kernels using the results from kmeans
    """
    compressed_kernels = torch.zeros_like(kernels)
    n = kernels.shape[0]
    for i in range(n):
        compressed_kernels[i] = kmeans_centroids[kmeans_indexes[i]]

    return compressed_kernels


def get_scalar(t1, t2):
    S = torch.sum(t2 * t1, dim=1)
    C2 = torch.sum(t2 ** 2, dim=1)
    beta = S/C2
    return beta


def compress_torch_weights(model_path, compressed_model_path, kmeans_batch_size=10000, num_clusters=200):
    """
    simple custering based method to achieve vector quantization
    opens the pytorch model weights and applies faiss kmeans to the kernels
    saves the compressed weights to disk
    """
    checkpoint = torch.load(model_path)
    weights = checkpoint

    K = concat_weights(weights)
    compressed_K = torch.zeros_like(K)

    N = K.shape[0]
    batch_size = kmeans_batch_size
    kmeans_repeat = 1

    # 获取特征维度
    d = K.shape[1]
    # 使用GPU初始化faiss聚类器（如果有GPU）
    if torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False  # 使用32位浮点数提高精度
        index = faiss.GpuIndexFlatL2(res, d, cfg)  # L2距离
    else:
        index = faiss.IndexFlatL2(d)  # CPU版本

    i = 0
    while i * batch_size < N:
        print(f'Processing batch {i}: {i*batch_size}/{N}')
        i0 = i * batch_size
        i1 = min(i0 + batch_size, N)  # 处理最后一个批次
        batch = K[i0: i1]
        
        # 转换为numpy数组并确保是32位浮点数（faiss要求）
        batch_np = batch.cpu().numpy().astype(np.float32)
        
        kmeans_results = []
        errors = []
        for index_repeat in range(kmeans_repeat):
            print(f'repeating {index_repeat}')
            
            # 使用faiss进行kmeans聚类
            kmeans = faiss.Kmeans(
                d=d, 
                k=num_clusters, 
                niter=20,  # 迭代次数
                nredo=5,   # 重复次数，取最好结果
                gpu=torch.cuda.is_available()  # 是否使用GPU
            )
            kmeans.train(batch_np)
            
            # 获取聚类中心和分配结果
            _, kmeans_indexes = kmeans.index.search(batch_np, 1)  # 每个样本找最近的中心
            kmeans_indexes = kmeans_indexes.flatten()  # 展平为一维数组
            kmeans_centroids = torch.from_numpy(kmeans.centroids).to(batch.device)  # 转换为tensor并移到相同设备
            
            # 应用聚类结果
            result = apply_kmeans_to_kernels(batch, kmeans_centroids, kmeans_indexes)
            
            beta = 1
            kmeans_results.append(result)
            errors.append(norm(batch - beta * result))
            print(f'error: {errors[-1]:.4}')

        min_error_vector_index = torch.argmin(torch.tensor(errors))
        compressed_K[i0: i1] = kmeans_results[min_error_vector_index]
        i += 1

    compressed_dict = reassign_weights(compressed_K, weights)
    torch.save(compressed_dict, compressed_model_path)
    top1_avg = validate_vgg19_cifar10(compressed_model_path)

    baseline = 90.9
    diff = baseline - top1_avg
    result_dict = dict(
        timestamp=now(),
        baseline=baseline,
        precision=top1_avg,
        precision_diff=diff,
        num_clusters=num_clusters,
        kmeans_batch_size=batch_size,
        N=N,
        compression_rate=num_clusters/batch_size,
        model_name='vgg19',
        task='classification',
        dataset='cifar10',
        compression_type_base='faiss_kmeans',  # 更新压缩类型标识
        compression_type='faiss_kmeans',
        compressed_model_name=compressed_model_path
    )
    print(result_dict)

    fname = 'C:/Users/Wenxin.Fu23/Desktop/vgg_vq/results/result_vqid.json'
    if not os.path.exists(fname):
        content = []
    else:
        content = ORJSONIO(fname).read()
    content.append(result_dict)
    ORJSONIO(fname).write(content)


def do():
    pass


def main():
    do()


if __name__ == '__main__':
    main()