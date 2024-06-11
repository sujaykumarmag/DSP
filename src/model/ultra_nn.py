

import math
import torch
import torch.nn as nn
import pandas as pd
import numpy as np 
import src.model.utils as mu

from torch import distributed as dist
from torch.utils import data as torch_data

from tqdm import tqdm
from transformers import AutoModel

from collections import UserDict
from typing import Dict, Iterable, Mapping, Union

from chemicalx.data import DrugCombDB, BatchGenerator, DrugComb, OncoPolyPharmacology
from torchdrug.data import Molecule,PackedGraph,Graph

from torch_geometric.data import Data
from torch_geometric.data import Data
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip

from src.model.molecule import MoleculesData
from ultra.tasks import build_relation_graph
from ultra import tasks, util
from ultra.datasets import HM









class UltraNet(nn.Module):

    def __init__(self,input_dims,padding):
        super().__init__()
        self.input_dim = input_dims
        self.ultra_model =  AutoModel.from_pretrained("mgalkin/ultra_3g", trust_remote_code=True)
        self.padding = padding

        self.lin1 = nn.Linear(in_features=self.input_dim,out_features=32)
        self.lin2 = nn.Linear(in_features=32, out_features=32)
        self.lin3 = nn.Linear(in_features=32, out_features=32)
        self.lin4 = nn.Linear(in_features=32, out_features=1)


        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)


    def all_negative(self, data, batch):
        pos_h_index, pos_t_index, pos_r_index = batch
        r_index = pos_r_index.unsqueeze(-1).expand(-1, data.num_nodes)
        # generate all negative tails for this batch
        all_index = torch.arange(data.num_nodes)
        h_index, t_index = torch.meshgrid(pos_h_index, all_index, indexing="ij")  # indexing "xy" would return transposed
        t_batch = torch.stack([h_index, t_index, r_index], dim=-1)
        # generate all negative heads for this batch
        all_index = torch.arange(data.num_nodes)
        t_index, h_index = torch.meshgrid(pos_t_index, all_index, indexing="ij")
        h_batch = torch.stack([h_index, t_index, r_index], dim=-1)
        return t_batch, h_batch



    def save_edges(self,edge_lists_save):
        with open(f"src/model/cache/test.txt", 'w') as file:
            for edge in edge_lists_save:
                file.write('\t'.join(map(str, edge)) + '\n')



    def get_embeddings(self,molecules):
        drug_embedding_batch = []
        drugs = []
        mrr = []
        hits10 = []
        for ind, i in enumerate(molecules):
            edge_lists = [j  for j in i.edge_list.cpu().numpy().tolist()]
            self.save_edges(edge_lists)
            
            embeddings  = []
  
            md = MoleculesData()
            data = md.process()
            device = util.get_devices(None)
            world_size = util.get_world_size()
            rank = util.get_rank()
            sampler = torch_data.DistributedSampler(edge_lists, world_size, rank)
            test_loader = torch_data.DataLoader(edge_lists, len(edge_lists), sampler=sampler)    
            self.ultra_model.eval()
            
            for batch in test_loader:
                t_batch, h_batch = self.all_negative(data, batch)
                try:
                    t_pred,_ = self.ultra_model(data, t_batch)
                    h_pred,_ = self.ultra_model(data, h_batch)
                    emb = h_pred.reshape(-1).detach().numpy()
                except:
                    # print(f"Some Error in model - Check once file no {ind} and drug {i}")
                    t_pred = np.zeros((2,3))
                    h_pred = np.zeros((2,3))
                    drug_df = pd.concat([pd.DataFrame(t_pred),pd.DataFrame(h_pred)],axis=1)
                    emb = np.array(drug_df).reshape(-1)
                embeddings.append(emb)
            embeddings = np.array(embeddings).reshape(-1)
            padded_embedding = np.pad(embeddings, (0, self.padding - len(embeddings)), 'constant')
            drug_embedding_batch.append(padded_embedding)
        return torch.tensor(drug_embedding_batch,dtype=torch.float32)
        


    def forward(self,  context_features: torch.FloatTensor, molecules_left: PackedGraph, molecules_right: PackedGraph, plot=False,
    ) -> torch.FloatTensor:
            
        drug_embedding_left = self.get_embeddings(molecules_left)
        drug_embedding_right = self.get_embeddings(molecules_right)

        out = self.lin1(torch.cat([drug_embedding_left,drug_embedding_right],dim=1))
        out = self.relu(out)

        out = self.lin2(out)
        out = self.relu(out)

        out = self.lin3(out)
        out = self.relu(out)

        out = self.lin4(out)
        out = self.relu(out)

        out = self.sigmoid(out)

        return out

    

