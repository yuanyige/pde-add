# import os
# import json
# import torch

# from core.models import load_model
# from core.testfn import run_final_test_select_cifarc, run_final_test_cifarc, run_final_test_selfmade
# from core.parse import parser_test
from core.data import load_cifar_c


# args_test = parser_test()

# with open(os.path.join(args_test.ckpt_path,'train/args.txt'), 'r') as f:
#     old = json.load(f)
#     args_test.__dict__ = dict(vars(args_test), **old)


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = load_model(args_test.backbone, args_test.protocol)
# model = model.to(device)
# checkpoint = torch.load(os.path.join(args_test.ckpt_path,'train',args_test.load_ckpt+'.pt'))
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()
# del checkpoint

cname_list = ['gaussian_blur','gaussian_noise','brightness','saturate','contrast']
    


# def get_embedding(dataloader_test):
#     with torch.no_grad():
#         for x, y in dataloader_test:

#             x, y = x.to(device), y.to(device)

#             out = model(x, use_diffusion = False)

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torchvision.datasets import CIFAR10
from torchvision import transforms
from sklearn.manifold import TSNE

# 定义数据预处理方法
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 加载CIFAR-10数据集
# trainset = CIFAR10(root='./datasets/cifar10', train=False,
#                     download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
#                                           shuffle=True, num_workers=2)

# 将图像数据转化为一维向量
data = []
for ci, cname in enumerate(cname_list):
    # Load data
    loader_c = load_cifar_c('cifar10', './datasets', 1, cname, severity=5)
    for i, (inputs, labels) in enumerate(loader_c):
        #print(labels)
        inputs = inputs.view(inputs.size(0), -1)
        data.append(inputs.numpy())
data = np.concatenate(data, axis=0)
print(data.shape)

# 数据预处理
scaler = StandardScaler()
data = scaler.fit_transform(data)

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(data)

# 绘制散点图
plt.figure(figsize=(10, 10))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], cmap='jet')
plt.colorbar()
plt.title('t-SNE Visualization of MNIST (0-9)')
plt.savefig('vis.png')


# # 使用K-means聚类算法
# kmeans = KMeans(n_clusters=10, random_state=0)
# kmeans.fit(data)

# # 可视化聚类结果
# pca = PCA(n_components=2)
# pca.fit(data)
# transformed_data = pca.transform(data)
# transformed_centers = pca.transform(kmeans.cluster_centers_)
# plt.figure(figsize=(10, 6))
# plt.scatter(transformed_data[:,0], transformed_data[:,1], c=kmeans.labels_, cmap='rainbow', s=5)
# plt.scatter(transformed_centers[:,0], transformed_centers[:,1], marker='x', color='black', s=100)
# plt.title('CIFAR-10 K-means Clustering')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.savefig('vis.png')