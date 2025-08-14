import collections
import json
import logging

import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm

from torch.utils.data import DataLoader

from datasets_mm import EmbDataset
from models.rqvae_mm import RQVAE
import argparse
import os

def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item==tot_indice

def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count

def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []

    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups

def parse_args():
    parser = argparse.ArgumentParser(description="MM-RQ-VAE")
    parser.add_argument("--dataset", type=str,default="Beauty", help='dataset')
    parser.add_argument("--root_path", type=str,default="../checkpoint/", help='root path')
    parser.add_argument('--alpha', type=str, default='1e-1', help='cf loss weight')
    parser.add_argument('--epoch', type=int, default='10000', help='epoch')
    parser.add_argument('--checkpoint', type=str, default='epoch_9999_collision_0.0012_model.pth', help='checkpoint name')
    parser.add_argument('--beta', type=str, default='1e-4', help='div loss weight')
    parser.add_argument("--data_path_1", type=str, default="../data", help="CF emb.")
    parser.add_argument("--data_path_2", type=str, default="../data", help="Text emb.")
    parser.add_argument("--data_path_3", type=str, default="../data", help="Vision emb.")
    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256,256,256,256], help='emb num of every vq')
    parser.add_argument('--e_dim', type=int, default=32, help='rq codebook embedding size')
    parser.add_argument('--layers', type=int, nargs='+', default=[2048,1024,512,256,128,64], help='hidden sizes of every layer')
    return parser.parse_args()

args_setting = parse_args()

dataset = args_setting.dataset
ckpt_path = args_setting.root_path + args_setting.checkpoint

device = torch.device("cuda:0")

ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
args = ckpt["args"]
state_dict = ckpt["state_dict"]


data = EmbDataset(args.data_path_1,args.data_path_2,args.data_path_3)

model = RQVAE(in_dim=data.dim,
                  num_emb_list=args.num_emb_list,
                  e_dim=args.e_dim,
                  layers=args.layers,
                  dropout_prob=args.dropout_prob,
                  bn=args.bn,
                  loss_type=args.loss_type,
                  quant_loss_weight=args.quant_loss_weight,
                  kmeans_init=args.kmeans_init,
                  kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons,
                  sk_iters=args.sk_iters,
                  )

model.load_state_dict(state_dict,strict=False)
model = model.to(device)
model.eval()
print(model)

data_loader = DataLoader(data,num_workers=args.num_workers,
                             batch_size=64, shuffle=False,
                             pin_memory=True)

all_indices = []
all_indices_str = []
all_indices_2 = []
all_indices_str_2 = []
all_indices_3 = []
all_indices_str_3 = []

def constrained_km(data, n_clusters=10):
    from k_means_constrained import KMeansConstrained 
    x = data
    size_min = min(len(data) // (n_clusters * 2), 10)
    clf = KMeansConstrained(n_clusters=n_clusters, size_min=size_min, size_max=n_clusters * 6, max_iter=10, n_init=10,
                            n_jobs=10, verbose=False)
    clf.fit(x)
    t_centers = torch.from_numpy(clf.cluster_centers_)
    t_labels = torch.from_numpy(clf.labels_).tolist()
    return t_centers, t_labels

labels = {"0":[],"1":[],"2":[], "3":[]}
embs  = [layer.embedding.weight.cpu().detach().numpy() for layer in model.rq.vq_layers]

for idx, emb in enumerate(embs):
    centers, label = constrained_km(emb)
    labels[str(idx)] = label

labels_2 = {"0":[],"1":[],"2":[], "3":[]}
embs  = [layer.embedding.weight.cpu().detach().numpy() for layer in model.pic_rq.vq_layers]

for idx, emb in enumerate(embs):
    centers, label = constrained_km(emb)
    labels_2[str(idx)] = label

labels_3 = {"0":[],"1":[],"2":[], "3":[]}
embs  = [layer.embedding.weight.cpu().detach().numpy() for layer in model.text_rq.vq_layers]

for idx, emb in enumerate(embs):
    centers, label = constrained_km(emb)
    labels_3[str(idx)] = label

for d in tqdm(data_loader):
    d, pic, text,emb_idx = d[0], d[1], d[2], d[3]
    d, pic, text = d.to(device), pic.to(device), text.to(device)
    
    indices,indices_2,indices_3 = model.get_indices(d,pic,text, labels,labels_2,labels_3,use_sk=False)

    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
    indices_2 = indices_2.view(-1, indices.shape[-1]).cpu().numpy()
    indices_3 = indices_3.view(-1, indices.shape[-1]).cpu().numpy()
    for index in indices:
        code = []
        for i, ind in enumerate(index):
            code.append(int(ind))
        all_indices.append(code)
        all_indices_str.append(str(code))
    for index in indices_2:
        code = []
        for i, ind in enumerate(index):
            code.append(int(ind))
        all_indices_2.append(code)
        all_indices_str_2.append(str(code))
    for index in indices_3:
        code = []
        for i, ind in enumerate(index):
            code.append(int(ind))
        all_indices_3.append(code)
        all_indices_str_3.append(str(code))

all_indices = np.array(all_indices)
all_indices_str = np.array(all_indices_str)
all_indices_2 = np.array(all_indices_2)
all_indices_str_2 = np.array(all_indices_str_2)
all_indices_3 = np.array(all_indices_3)
all_indices_str_3 = np.array(all_indices_str_3)

# save SID
torch.save(all_indices,f'dataset/mm_ID_SID_4096_num{args_setting.num_emb_list[0]}_0.pt')
torch.save(all_indices_2,f'dataset/mm_pic_SID_4096_num{args_setting.num_emb_list[0]}_0.pt')
torch.save(all_indices_3,f'dataset/mm_text_SID_4096_num{args_setting.num_emb_list[0]}_0.pt')

# save codebook embedding
emb = model.rq.get_code()
torch.save(emb,f'/dataset/mm_ID_SID_code_4096_num{args_setting.num_emb_list[0]}_0.pt')
emb = model.pic_rq.get_code()
torch.save(emb,f'/datasets/mm_pic_SID_code_4096_num{args_setting.num_emb_list[0]}_0.pt')
emb = model.text_rq.get_code()
torch.save(emb,f'/datasets/mm_text_SID_code_4096_num{args_setting.num_emb_list[0]}_0.pt')

