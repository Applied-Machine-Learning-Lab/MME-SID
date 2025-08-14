import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
#import wandb
import random
import collections
from .layers import MLPLayers
from .rq_mm import ResidualVectorQuantizer
from info_nce import InfoNCE, info_nce
from tllib.alignment.dan import MultipleKernelMaximumMeanDiscrepancy
from tllib.modules.kernels import GaussianKernel

class RQVAE(nn.Module):
    def __init__(self,
                 in_dim=768,
                 num_emb_list=None,
                 e_dim=64,
                 pic_dim=1280,
                 text_dim=1280,
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 kmeans_init=False,
                 kmeans_iters=100,
                 sk_epsilons= None,
                 sk_iters=100,
                 n_clusters = 10,
                 sample_strategy = 'all',
                 cf_embedding = 0,
                 align=0.01,
                 recon=1  
        ):
        super(RQVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight=quant_loss_weight
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.cf_embedding = cf_embedding
        self.n_clusters = n_clusters
        self.sample_strategy = sample_strategy
        self.pic_dim = pic_dim
        self.text_dim = text_dim
        self.align=align
        self.recon=recon
        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)
        self.pic_encode_layer_dims = [self.pic_dim] + self.layers + [self.e_dim]
        self.pic_encoder = MLPLayers(layers=self.pic_encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)
        self.text_encode_layer_dims = [self.text_dim] + self.layers + [self.e_dim]
        self.text_encoder = MLPLayers(layers=self.text_encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)
        self.rq = ResidualVectorQuantizer(num_emb_list, e_dim, align=self.align,
                                          kmeans_init = self.kmeans_init,
                                          kmeans_iters = self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters,)
        self.pic_rq = ResidualVectorQuantizer(num_emb_list, e_dim, align=self.align,
                                          kmeans_init = self.kmeans_init,
                                          kmeans_iters = self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters,)
        self.text_rq = ResidualVectorQuantizer(num_emb_list, e_dim, align=self.align,
                                          kmeans_init = self.kmeans_init,
                                          kmeans_iters = self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters,)    
        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                       dropout=self.dropout_prob,bn=self.bn)
        self.pic_decode_layer_dims = self.pic_encode_layer_dims[::-1]
        self.pic_decoder = MLPLayers(layers=self.pic_decode_layer_dims,
                                       dropout=self.dropout_prob,bn=self.bn)
        self.text_decode_layer_dims = self.text_encode_layer_dims[::-1]
        self.text_decoder = MLPLayers(layers=self.text_decode_layer_dims,
                                       dropout=self.dropout_prob,bn=self.bn) 
        self.mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(kernels=[GaussianKernel(alpha=2 ** -1)])  
    def forward(self, x, y,z,labels,labels_2,labels_3, use_sk=True):
        x = self.encoder(x)
        y = self.pic_encoder(y)
        z = self.text_encoder(z)
        x_q,  rq_loss, indices = self.rq(x,labels, use_sk=use_sk)
        y_q,  rq_loss_2, indices_2 = self.pic_rq(y,labels_2, use_sk=use_sk)
        z_q,  rq_loss_3, indices_3 = self.text_rq(z,labels_3, use_sk=use_sk)
        out = self.decoder(x_q)
        pic_out = self.pic_decoder(y_q)
        text_out = self.text_decoder(z_q)
        return out, pic_out, text_out, rq_loss,rq_loss_2,rq_loss_3, indices, indices_2,indices_3,x_q, y_q, z_q
    
    def vq_initialization(self,x,y,z, use_sk=True):
        self.rq.vq_ini(self.encoder(x))
        self.pic_rq.vq_ini(self.pic_encoder(y))
        self.text_rq.vq_ini(self.text_encoder(z))
    @torch.no_grad()
    def get_indices(self, xs,ys,zs, labels,labels_2,labels_3, use_sk=False):
        x_e = self.encoder(xs)
        y_e = self.pic_encoder(ys)
        z_e = self.text_encoder(zs)
        _, _, indices_1 = self.rq(x_e, labels, use_sk=use_sk)
        _, _, indices_2 = self.pic_rq(y_e, labels_2, use_sk=use_sk)
        _, _, indices_3 = self.text_rq(z_e, labels_3, use_sk=use_sk)
        return indices_1,indices_2,indices_3

    def compute_loss(self, out,pic_out,text_out, quant_loss,quant_loss_2,quant_loss_3, emb_idx, dense_out,dense_out_2,dense_out_3, xs,ys,zs):

        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')+F.mse_loss(pic_out, ys, reduction='mean')+F.mse_loss(text_out, zs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        elif self.loss_type == 'mmd':
            loss_recon = self.mkmmd_loss(out, xs)+self.mkmmd_loss(pic_out, ys)+self.mkmmd_loss(text_out, zs)
        else:
            raise ValueError('incompatible loss type')
        total_loss = self.recon*loss_recon + self.quant_loss_weight * (quant_loss+quant_loss_2+quant_loss_3)+self.align*(self.mkmmd_loss(dense_out,dense_out_2)+self.mkmmd_loss(dense_out,dense_out_3))

        return total_loss, None, loss_recon, quant_loss
