import argparse
import random
import torch
import numpy as np
from time import time
import logging
from torch.utils.data import DataLoader
import pickle
from datasets_MM import EmbDataset
from models.rqvae_mm import RQVAE
from trainer_new_mm import Trainer
import os

def parse_args():
    parser = argparse.ArgumentParser(description="MM-RQ-VAE")
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=20000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, )
    parser.add_argument('--eval_step', type=int, default=2000, help='eval step')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')
    parser.add_argument("--data_path_1", type=str, default="../data", help="CF emb.")
    parser.add_argument("--data_path_2", type=str, default="../data", help="Text emb.")
    parser.add_argument("--data_path_3", type=str, default="../data", help="Vision emb.")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='l2 regularization weight')
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    parser.add_argument("--bn", type=bool, default=False, help="use bn or not")
    parser.add_argument("--loss_type", type=str, default="mmd", help="loss_type")
    parser.add_argument("--kmeans_init", type=bool, default=True, help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.0, 0.003], help="sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")
    parser.add_argument("--device", type=str, default="cuda:0", help="gpu or cpu")
    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256,256,256,256], help='emb num of every layer')
    parser.add_argument('--e_dim', type=int, default=32, help='vq codebook embedding size')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='vq quantion loss weight')
    parser.add_argument('--n_clusters', type=int, default=10, help='n_clusters')
    parser.add_argument('--sample_strategy', type=str, default="all", help='sample_strategy')
    parser.add_argument('--layers', type=int, nargs='+', default=[2048,1024,512,256,128,64], help='hidden sizes of every layer')
    parser.add_argument("--maxe", type=int, default=2000, help="earlystop")
    parser.add_argument("--ckpt_dir", type=str, default="../checkpoint", help="output directory for model")
    parser.add_argument('--align', type=float, default=0.01, help='alignment loss coefficient')
    parser.add_argument('--recon', type=float, default=3, help='reconstruction loss coefficient')

    return parser.parse_args()


if __name__ == '__main__':
    """fix the random seed"""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()

    print(args)
    logging.basicConfig(level=logging.DEBUG)
    """build dataset"""
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
                  n_clusters= args.n_clusters,
                  sample_strategy =args.sample_strategy,
                  align=align,
                  recon=recon
                  )
    print(model)
    data_loader = DataLoader(data,num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=True,
                             pin_memory=True)

    trainer = Trainer(args,model)
    best_loss, best_collision_rate = trainer.fit(data_loader)
    #best_loss = trainer.fit(data_loader)
    print("Best Loss",best_loss)
    #print("Best Collision Rate", best_collision_rate)




