
import torch
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch.nn as nn

from tqdm import tqdm

from sklearn.metrics import classification_report
from sklearn.manifold import TSNE
import pandas as pd

from src.model.ultra_nn import UltraNet
from src.model.deep_dds import DeepDDS
from src.model.multi_deep_dds import MultiDeepDDS
from src.utils import save_results 


class NormalTrain():
    
    def __init__(self,args, loader):
        
        self.args = args
        self.lr = args.learningrate
        self.dataset = args.dataset
        self.epochs = args.epochs
        self.train_loader = loader.train_loader
        self.test_loader =  loader.test_loader
        

        print("Train Dataloader -- length : ",len(self.train_loader))
        print("Test Dataloader -- length : ",len(self.test_loader))

        self.model = args.model
        self.device = args.device
        
        self.resultsdir = args.results_dir
        os.makedirs(f'{self.resultsdir}',exist_ok=True)


    def get_models(self):
        if self.model == "gin-gat":
            return MultiDeepDDS(context_channels=len(self.train_loader.context_feature_set))
        elif self.model == "gat-gat" or self.model == "deepdds":
            return DeepDDS(context_channels=len(self.train_loader.context_feature_set))
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
            

    def train_model(self, model, optimizer):
        loss = 0.0
        lossfn = nn.BCELoss()

        model.to(self.device)
        for batch in self.train_loader:
            optimizer.zero_grad() 
            logits = model(batch.context_features, batch.drug_molecules_left, batch.drug_molecules_right)
            r_loss = lossfn(input=logits,target=batch.labels)           
            r_loss.backward()
            optimizer.step()
            loss = loss + r_loss
        return loss, model
    




    def get_training_metrics(self,loss,report):
    
        if isinstance(report, dict):
            metrics = report
        else:
            raise ValueError("Report should be a dictionary")
        df = pd.DataFrame(metrics).transpose()
        df['loss'] = loss
        df_flat = df.unstack().to_frame().T
        df_flat.columns = [f'{col[0]}_{col[1]}' for col in df_flat.columns]
        return df_flat




    def save_result(self,training_metrics,model,y_pred, y_test):
        if self.model == "ultra":
            self.plot_embeddings_ultra(model)
            save_results(self,training_metrics,model,y_pred, y_test)
        else:
            self.plot_embeddings_dds(model)
            save_results(self,training_metrics,model,y_pred, y_test)

    def plot_embeddings_ultra(self,model):


        """
            1. Plot the Embeddings after the model.gat and model.gin 
            2. Combine 2 plots into 1 pot
            3. try for one sample / one batch
        """
        for i in self.test_loader:
            sample = i
        z_latent1 = model.get_embeddings(sample.drug_molecules_left)
        z_latent2 = model.get_embeddings(sample.drug_molecules_right)
        z1 = z_latent1[0].detach().numpy()
        z2 = z_latent2[0].detach().numpy()
    
        label = sample.labels[0]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(z2,z1, c='blue', label=f'{label}')
        ax.set_title('Z2 and Z1')
        ax.legend()
        os.makedirs(f'{self.resultsdir}',exist_ok=True)
        os.makedirs(f'{self.resultsdir}/train {self.get_train_num()}',exist_ok=True)
        plt.savefig(f'{self.resultsdir}/train {self.get_train_num()}/embeddings_ultra.png')


    def get_train_num(self):
        arr = os.listdir(self.resultsdir)
        try:
            arr.remove(".DS_Store")
        except:
            print("No .DS_Store")
        runs = [run for run in arr if run.startswith("train") and run[6:].isdigit()]
        if not runs:
            return 1
        num = sorted(int(run[6:]) for run in runs)[-1]
        return num+1


        

    def plot_embeddings_dds(self,model):

        """
            1. Plot the Embeddings after the model.gat and model.gin 
            2. Combine 2 plots into 1 pot
            3. try for one sample / one batch
        """
        for i in self.test_loader:
            sample = i
        z_latent1 = model._forward_molecules_gcn(sample.drug_molecules_left)
        z_latent2 = model._forward_molecules_gin(sample.drug_molecules_left)
        z1 = z_latent1[0].detach().numpy()
        z2 = z_latent2[0].detach().numpy()
        z3 = model._forward_molecules_gcn(sample.drug_molecules_right).detach().numpy()
        z4 = model._forward_molecules_gcn(sample.drug_molecules_right).detach().numpy()
        label = sample.labels[0]
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].scatter(z2,z1, c='blue', label=f'{label}')
        ax[0].set_title('Z2 and Z1')
        ax[0].legend()
        ax[1].scatter(z3,z4, c='blue', label=f'{label}')
        ax[1].set_title('Z3 and Z4')
        ax[1].legend()
        self.num = self.get_train_num()
        os.makedirs(f'{self.resultsdir}/train {self.num}',exist_ok=True)
        plt.savefig(f'{self.resultsdir}/train {self.num}/embeddings_dds.png')




        
    def train(self):
        model = self.get_models()
        print(model)
        model_dict = {}
        training_metrics = []
        model_dict["model"] = model
        optimizer = torch.optim.Adam(model.parameters(),lr=self.lr)
        for i in tqdm(range(self.epochs)):
            # loss, model_dict["model"] = self.train_model(model_dict["model"],optimizer)
            loss = 0.0
            y_pred, y_test = self.evaluate(model_dict["model"])
            cls = classification_report(y_pred=y_pred,y_true=y_test,output_dict=True)
            training_metrics.append(self.get_training_metrics(loss,cls))
            tqdm.write(f"Epochs : [{i+1}/{self.epochs}], The loss is  {loss}")
        y_pred, y_test = self.evaluate(model_dict["model"])
        self.save_result(training_metrics,model_dict["model"],y_pred, y_test)



    def evaluate(self,model):
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