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


import weights as wgt
from algorithms.kmeans import kmeans
from utils.timingtools import now
from exp_vgg import validate_vgg19_cifar10





def plot_vectors_centroid(index, original_kernels, kmeans_centroids, kmeans_indexes):

    import seaborn as sns
    import matplotlib.pyplot as plt

    kernel_indexes = np.where(kmeans_indexes.cpu().numpy() == index)[0]
    print(f'found {len(kernel_indexes)} kernels to centroid of index {index}')

    kernels = []
    for i in kernel_indexes:
        kernels.append(original_kernels[i].cpu().numpy())
    

    plt.clf()
    for k in kernel_indexes:
        ax = sns.lineplot(x=list(range(9)), y=original_kernels[k].cpu().numpy(), color='blue')


    centroid = kmeans_centroids[index].cpu().numpy()
    ax = sns.lineplot(x=list(range(9)), y=centroid, color='red')
    ax.set_title(f'{len(kernel_indexes)} kernels to centroid of index {index}')
    plt.savefig(f'centroid_plots/centroid_{index}.jpg')
    # print()


def apply_kmeans_to_kernels(kernels, kmeans_centroids, kmeans_indexes):
    """
    Apply kmeans to kernels using the results from kmeans
    """
    compressed_kernels = torch.zeros_like(kernels)
    n = kernels.shape[0]
    for i in range(n):
        compressed_kernels[i] = kmeans_centroids[kmeans_indexes[i]]

    return compressed_kernels


def compress_torch_weights(model_path, compressed_model_path, kmeans_batch_size=10000, num_clusters=200):
    """
    opens the pytorch model weights and applies kmeans to the kernels
    saves the compressed weights to disk
    """
    checkpoint = torch.load(model_path)
    weights = checkpoint['state_dict']

    K = wgt.concat_weights(weights)
    compressed_K = torch.zeros_like(K)

    N = K.shape[0]
    batch_size = kmeans_batch_size


    i = 0
    while i*batch_size < N:
        print(i*batch_size, N)
        i0 = i*batch_size
        i1 = i0 + batch_size
        batch = K[i0: i1]
        print(f'[{i0}/{N}] ({i1/N * 100:.2f}/100)')

        kmeans_indexes, kmeans_centroids = kmeans(
            X=batch, num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0')
        )
        compressed_K[i0: i1] = apply_kmeans_to_kernels(batch, kmeans_centroids, kmeans_indexes)

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
        compression_type_base='kmeans',
        compression_type='kmeans_no_centroid_rescale',
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

    kernel_as_vector = torch.tensor(K)  # K_1.reshape(n, -1)
    cluster_ids_x, cluster_centers = kmeans(
        X=kernel_as_vector, num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0')
    )
    print('finished k-means clustering')

    for i in range(num_clusters):
        print(i)
        plot_vectors_centroid(i, kernel_as_vector, cluster_centers, cluster_ids_x)







def main():
    model_path = 'vgg19.pth'
    
    num_clusters = int(sys.argv[1])
    N = int(sys.argv[2])
    assert num_clusters < N
    compressed_model_path = f'./compressed_models/vgg19_kmeans_centroid_cifar10_{num_clusters}_{N}.pth'

    t0 = time.time()
    wgt.compress_torch_weights(
        model_path, compressed_model_path, 
        kmeans_batch_size=N, num_clusters=num_clusters
    )
    t1 = time.time()
    print(f'time: {t1 - t0:.2f} seconds')
    print(f'num clusters: {num_clusters}/{N}')


if __name__ == '__main__':
    main()