import torch
from torch import nn
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from . import ViT

''' position embedding code from 
https://github.com/ZeyuGaoAi/Instance_based_Vision_Transformer/blob/0ebe034f9b6faf171870a3964bf49346477c2a48/I-ViT/vit/
models/PositionalEncoding.py '''

class LearnedClusterEncoding(nn.Module):
    def __init__(self, max_cluster_embeddings, embedding_dim):
        super(LearnedClusterEncoding2, self).__init__()
        # max_cluster_embeddings should be n_clusters + 1
        self.pe = nn.Embedding(max_cluster_embeddings, embedding_dim)
        self.register_buffer(
            "cluster_labels",
            torch.arange(max_cluster_embeddings).expand((1, -1)),
        )

    def forward(self, x, cluster_labels):
        cluster_labels = cluster_labels.to(torch.int64)
        a = nn.Parameter(torch.zeros(cluster_labels.shape[0], 1)).cuda()
        cluster_labels = torch.cat((a, cluster_labels), dim=1).to(torch.int64)
        cluster_embeddings = self.pe(cluster_labels)
        return x + cluster_embeddings


class ClusterViT(nn.Module):
    def __init__(self, num_classes, dim, depth, heads, mlp_dim, patch_dim, max_num_patches,
                 n_clusters=4, pool='cls', dim_head=64, dropout=0., emb_dropout=0.):

        super().__init__()

        self.dim = dim
        self.n_clusters = n_clusters
        self.max_num_patches = max_num_patches

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.scale_down = nn.Linear(patch_dim, dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, max_num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.cluster_token = LearnedClusterEncoding(n_clusters + 1, dim)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = ViT.Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        if num_classes == 1:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, num_classes),
                nn.Sigmoid()
            )
        elif num_classes > 1:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, num_classes)
            )

    def forward(self, img, clusterings):
        x = self.scale_down(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        clusterings = torch.tensor(clusterings).unsqueeze(0).cuda()
        x = self.cluster_token(x, clusterings)
        x += self.pos_embedding[:, :(n + 1)]
        
        x = self.dropout(x)

        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
