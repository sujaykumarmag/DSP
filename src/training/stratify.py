
import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.model.deep_dds import DeepDDS
from src.model.multi_deep_dds import MultiDeepDDS
from src.model.ultra_nn import UltraNet
from src.training.normal import NormalTrain

from sklearn.metrics import classification_report
from src.utils import save_results 




class StratifyTrain(NormalTrain):

    def __init__(self,args,loader):
        self.args = args
        self.epochs = args.epochs
        self.lr = args.learningrate
        self.loader = loader.loader
        self.dataset = args.dataset
        self.model = args.model
        self.device = args.device
        self.resultsdir = args.results_dir
        os.makedirs(f'{self.resultsdir}',exist_ok=True)


    def get_models(self):
        if self.model == "gin-gat":
            return MultiDeepDDS(context_channels=len(self.loader.context_set))
        elif self.model == "gat-gat" or self.model == "deepdds":
            return DeepDDS(context_channels=len(self.loader.context_set))
        else:
            """
                1. Dataset 1 (DrugcombDB) == Padded == 48980
                2. Dataset 2 (DrugComb)   == Padded == 
            """
            if self.dataset == "drugcombdb":
                return UltraNet(input_dims=2*48980,padding=48980)
            elif self.dataset == "drugcomb":
                x = 48980
                return UltraNet(input_dims=2*x,padding=x) 
            



    def train(self, train_loader, model):
        lossfn = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=self.lr)
        model.to(self.device)
        for i in range(self.epochs):
            r_loss = 0.0
            for batch in train_loader:
                logits = model(batch.context_features, batch.drug_molecules_left, batch.drug_molecules_right)
                loss = lossfn(logits,batch.labels)
                r_loss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return model
    


    def get_training_metrics(self,report):
        if isinstance(report, dict):
            metrics = report
        else:
            raise ValueError("Report should be a dictionary")
        df = pd.DataFrame(metrics).transpose()
        df_flat = df.unstack().to_frame().T
        df_flat.columns = [f'{col[0]}_{col[1]}' for col in df_flat.columns]
        return df_flat



    

    def evaluate(self,test_loader,model):
        model.eval()
        model.to(torch.device("cpu"))
        with torch.no_grad():
            all_preds = []
            all_test = []
            for batch in test_loader:
                logits = model(batch.context_features, batch.drug_molecules_left, batch.drug_molecules_right)
                y_pred = (logits > 0.5).int()
                all_preds.extend(y_pred.cpu().numpy())
                all_test.extend(batch.labels.cpu().numpy())
        return all_preds, all_test
    
    def final_evaluate(self,model):
        model.eval()
        model.to(torch.device("cpu"))
        with torch.no_grad():
            all_preds = []
            all_test = []
            for batch in self.test_loader:
                logits = model(batch.context_features, batch.drug_molecules_left, batch.drug_molecules_right)
                y_pred = (logits > 0.5).int()
                all_preds.extend(y_pred.cpu().numpy())
                all_test.extend(batch.labels.cpu().numpy())
        return all_preds, all_test




    def stratify_train(self):
        model = self.get_models()
        print(model)
        model_dict = {}
        model_dict["model"] = model
        k_splits = self.loader.get_dataset_split()
        results_folded = []
        for fold_idx, (train_idx, test_idx) in tqdm(enumerate(k_splits, start=1)):
            train_loader, test_loader = self.loader.get_stratify_dataloader(train_idx,test_idx)
            # model_dict["model"] = self.train(train_loader,model_dict["model"])
            y_pred, y_test = self.evaluate(test_loader,model_dict["model"])
            cls = classification_report(y_pred=y_pred,y_true=y_test,output_dict=True)
            results_folded.append(self.get_training_metrics(cls))
        self.test_loader = self.loader.get_final_test_loader()
        y_pred, y_test = self.final_evaluate(model_dict["model"])
        self.save_result(results_folded,model_dict["model"],y_pred, y_test)



        
