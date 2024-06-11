

import argparse
import os
from src.dataloader import DataLoader
from src.training.normal import NormalTrain
from src.training.stratify import StratifyTrain
from src.model.ultra_nn import UltraNet

parser = argparse.ArgumentParser(description="command line Args based enabled training experiment pipeline for Drug Synergy Prediction")

parser.add_argument("model",metavar="model",type=str,default="gin-gat",help="Choose a model for the Experiment")
parser.add_argument("dataset",metavar="dataset",type=str,default="drugcomb",help="Datasets for Synergy Prediction (all small) \n 1. drugcomb, 2. drugcombdb 3. oncolypharm \n")
parser.add_argument("--validation",metavar="--validation",type=bool,default=False,help="Stratified Cross Fold Validation")
parser.add_argument("--results_dir",metavar="--results_dir",type=str,default="./runs/",help="All results in the runs directory by default")
parser.add_argument("--batch_size",metavar="--batch_size",type=int,default=32,help="Batching parameter for both models")
parser.add_argument("--epochs",metavar="--epochs",type=int,default=1, help="No of epochs for both models")
parser.add_argument("--learningrate",metavar='learningrate',type=float,default=0.01, help='Learning Rate Hyperparameter')
parser.add_argument("--device",metavar="device",type=str,default="cpu",help="Device for Training (CPU/MPS)")


args = parser.parse_args()

data_loader = DataLoader(args)

if args.validation:
    StratifyTrain(args,data_loader).stratify_train()
else:
    NormalTrain(args,data_loader).train()


