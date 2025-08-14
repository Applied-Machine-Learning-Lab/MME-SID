import numpy as np
import torch
import torch.utils.data as data
import pickle

class EmbDataset(data.Dataset):

    def __init__(self,data_path_1,data_path_2,data_path_3):

        self.embeddings = pickle.load(open(data_path_1,'rb')).to('cpu').squeeze().detach().numpy()
        self.pic = torch.load(data_path_2).to('cpu').detach().numpy()
        self.text = torch.load(data_path_3).to('cpu').detach().numpy()
        self.dim = self.embeddings.shape[-1]
        self.pic_dim = self.pic.shape[-1]
        self.text_dim = self.text.shape[-1]

    def __getitem__(self, index):
        emb = self.embeddings[index]
        pic = self.pic[index]
        text = self.text[index]
        return torch.FloatTensor(emb), torch.FloatTensor(pic), torch.FloatTensor(text), index

    def __len__(self):
        return len(self.embeddings)
