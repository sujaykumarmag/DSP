import torch
import torchdrug.data
from torch.types import Device
from src.utils import plot_embeddings

__all__ = [
    "PackedGraph",
    "Graph",
]


class PackedGraph(torchdrug.data.PackedGraph):
    def to(self, device: Device):
        if isinstance(device, str):
            if device == "cpu":
                return self.cpu()
            elif device == "cuda":
                return self.cuda()
            else:
                raise NotImplementedError(f"{self.__class__.__name__}.to() is not implemented for string: {device}")
        elif isinstance(device, torch.device):
            if device.type == "cpu":
                return self.cpu()
            elif device.type == "cuda":
                return self.cuda()
            else:
                raise NotImplementedError
        else:
            raise TypeError


class Graph(torchdrug.data.Graph):
    packed_type = PackedGraph



from typing import List, Optional

import torch
from torch import nn
from torch.nn.functional import normalize
from torchdrug.layers import MaxReadout
from torchdrug.layers import MultiLayerPerceptron as MLP
from torchdrug.models import GraphConvolutionalNetwork, GraphAttentionNetwork, GraphIsomorphismNetwork

# from compat import PackedGraph
from chemicalx.constants import TORCHDRUG_NODE_FEATURES
from chemicalx.data import DrugPairBatch
from chemicalx.models import Model

__all__ = [
    "DeepDDS",
]





class MultiDeepDDS(Model):


    def __init__(
        self,
        *,
        context_channels: int,
        context_hidden_dims: Optional[List[int]] = None,
        drug_channels: int = TORCHDRUG_NODE_FEATURES,
        drug_gcn_hidden_dims: Optional[List[int]] = None,
        drug_mlp_hidden_dims: Optional[List[int]] = None,
        context_output_size: int = 32,
        fc_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.5,  # Different from rate used in paper
    ):
       
        super().__init__()

        if context_hidden_dims is None:
            context_hidden_dims = [32, 32]
        if drug_gcn_hidden_dims is None:
            drug_gcn_hidden_dims = [drug_channels, drug_channels * 2, drug_channels * 4]
        if drug_mlp_hidden_dims is None:
            drug_mlp_hidden_dims = [drug_channels * 2]
        if fc_hidden_dims is None:
            fc_hidden_dims = [32, 32]

        # Cell feature extraction with MLP
        self.cell_mlp = MLP(
            input_dim=context_channels,
            hidden_dims=[*context_hidden_dims, context_output_size],
        )

        # GIN
        self.drug_conv2 = GraphIsomorphismNetwork(
            input_dim=drug_channels,
            hidden_dims=drug_gcn_hidden_dims,
            learn_eps=True,
            activation="relu",
        )

        # GAT
        self.drug_conv1 = GraphAttentionNetwork(
            input_dim=drug_channels,
            hidden_dims=drug_gcn_hidden_dims,
            activation="relu",
        )

        self.drug_readout = MaxReadout()

        self.drug_mlp = MLP(
            input_dim=drug_gcn_hidden_dims[-1],
            hidden_dims=[*drug_mlp_hidden_dims, context_output_size],
            dropout=dropout,
            activation="relu",
        )

        # Final layers
        self.final = nn.Sequential(
            MLP(
                input_dim=context_output_size * 5,
                hidden_dims=[*fc_hidden_dims, 1],
                dropout=dropout,
            ),
            torch.nn.Sigmoid(),
        )

    def unpack(self, batch: DrugPairBatch):
        """Return the context features, left drug features and right drug features."""
        return batch.context_features, batch.drug_molecules_left, batch.drug_molecules_right

    def _forward_molecules_gcn(self, molecules: PackedGraph) -> torch.FloatTensor:
        features = self.drug_conv1(molecules, molecules.data_dict["node_feature"])["node_feature"]
        features = self.drug_readout(molecules, features)
        return self.drug_mlp(features)
    
    def _forward_molecules_gin(self, molecules: PackedGraph) -> torch.FloatTensor:
        features = self.drug_conv2(molecules, molecules.data_dict["node_feature"])["node_feature"]
        features = self.drug_readout(molecules, features)
        return self.drug_mlp(features)
    

    def forward(
        self, context_features: torch.FloatTensor, molecules_left: PackedGraph, molecules_right: PackedGraph, plot=False,
    ) -> torch.FloatTensor:
        

        mlp_out = self.cell_mlp(normalize(context_features, p=2, dim=1))

        features_gcn_left = self._forward_molecules_gcn(molecules_left)
        features_gcn_right = self._forward_molecules_gcn(molecules_right)

        features_gin_left = self._forward_molecules_gin(molecules_left)
        features_gin_right = self._forward_molecules_gin(molecules_right)

        if plot is True:
            plot_embeddings(features_gcn_left,molecules_left,"GCN LEFT DRUG")
            plot_embeddings(features_gcn_right,molecules_right,"GCN RIGHT DRUG")

            plot_embeddings(features_gin_left,molecules_left,"GIN LEFT DRUG")
            plot_embeddings(features_gin_right,molecules_right,"GCN RIGHT DRUG")

        concat_in = torch.cat([mlp_out, features_gcn_left, features_gcn_right,features_gin_left,features_gin_right], dim=1)

        return self.final(concat_in)
