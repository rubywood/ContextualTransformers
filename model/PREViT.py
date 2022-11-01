import torch
import torch.nn as nn
import numpy as np

from einops import rearrange, repeat
from . import ViT
from . import ClusterViT

'''PRE module is build on code from TransMIL paper: https://github.com/szc19990412/TransMIL'''

class PRE(nn.Module):

    def __init__(self, tile_size, dim):
        super().__init__()
        self.tile_size = tile_size
        self.dim = dim
        self.proj = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=7, stride=1, padding=7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, stride=1, padding=5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=3 // 2, groups=dim)

    def forward(self, x, patch_paths):
        # input x is shape [1, n+1, dim] 
        # n = number of patches per WSI
        # dim = dimension of ViT
        B, _, C = x.shape #B = 1, C = 512
        cls_token, feat_token = x[:, 0], x[:, 1:]

        # ----> restore patch layout on WSIs
        all_paths = [patch_path_coordinates(path) for path in patch_paths]
        scaled_paths = np.divide(all_paths, self.tile_size).astype(int)
        x_scaled_paths = scaled_paths[:, 0]
        y_scaled_paths = scaled_paths[:, 1]

        Xfrom = np.min(x_scaled_paths)
        Yfrom = np.min(y_scaled_paths)

        rescaled_paths = scaled_paths - [Xfrom, Yfrom]
        x_rescaled_paths = rescaled_paths[:, 0]
        y_rescaled_paths = rescaled_paths[:, 1]

        reXto = np.max(x_rescaled_paths)
        reYto = np.max(y_rescaled_paths)
        
        reXfrom = np.min(x_rescaled_paths)
        reYfrom = np.min(y_rescaled_paths)

        assert rescaled_paths.shape[0] == feat_token.shape[1], \
            f"rescaled paths shape {rescaled_paths.shape} but feat token shape {feat_token.shape}"

        canvas_tensor = torch.zeros(size=(self.dim, reYto - reYfrom + 1, reXto - reXfrom + 1), dtype=torch.float32).cuda()
        canvas_tensor[:, rescaled_paths[:, 1], rescaled_paths[:, 0]] = feat_token[0, :, :].transpose(0, 1)
        canvas_tensor = canvas_tensor.unsqueeze(0)

        # ----> convolve as 2D
        x = self.proj(canvas_tensor) + canvas_tensor + self.proj1(canvas_tensor) + self.proj2(canvas_tensor)

        # ----> restore vector shape
        feat_token[0, :] = x[0, :, rescaled_paths[:, 1], rescaled_paths[:, 0]].transpose(0, 1)

        # ----> add cls token
        x = torch.cat((cls_token.unsqueeze(1), feat_token), dim=1)

        return x
    

def patch_path_coordinates(patch_path):
    path_list = patch_path.split('/')[-1].split('.')[0].split('_')
    try:
        x_index = path_list.index('x')
        y_index = path_list.index('y')
    except IndexError:
        raise ValueError("x or y is missing")
    x_coord = int(path_list[x_index + 1])
    y_coord = int(path_list[y_index + 1])
    return x_coord, y_coord


class PREViT(nn.Module):
    def __init__(self, *, num_classes, dim, depth, heads, mlp_dim, patch_dim, pool='cls', dim_head=64, dropout=0.,
                 emb_dropout=0.):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.scale_down = nn.Linear(patch_dim, dim)
        self.dim = dim
        self.patch_dim = patch_dim

        self.pos_embedding_pre = PRE(tile_size=256, dim=dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
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

    def forward(self, img, patch_paths):
        if not self.dim == self.patch_dim:
            x = self.scale_down(img)
        else:
            x = img
        del img
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b) 
        x = torch.cat((cls_tokens, x), dim=1) 
        del cls_tokens
        x = self.pos_embedding_pre(x, patch_paths)

        x = self.dropout(x)
        
        x = self.transformer(x) 

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
    
    
class ClusterPREViT(nn.Module):
    def __init__(self, num_classes, dim, depth, heads, mlp_dim, patch_dim,
                 n_clusters=4, pool='cls', dim_head=64, dropout=0., emb_dropout=0.):

        super().__init__()

        self.dim = dim
        self.patch_dim = patch_dim

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.scale_down = nn.Linear(patch_dim, dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.cluster_token = ClusterViT.LearnedClusterEncoding2(n_clusters+1, dim)
        self.dropout = nn.Dropout(emb_dropout)

        self.pos_embedding_pre = PRE(tile_size=256, dim=dim)


        self.transformer = ViT.Transformer(dim, depth, heads, dim_head, mlp_dim*2, dropout)

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

    def forward(self, features, clusterings, patch_paths):
        if not self.dim == self.patch_dim:
            x = self.scale_down(features)
        else:
            x = features
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b) 
        x = torch.cat((cls_tokens, x), dim=1)

        clusterings = torch.tensor(clusterings).unsqueeze(0).cuda()
        x = self.cluster_token(x, clusterings)

        x = self.pos_embedding_pre(x, patch_paths)

        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
