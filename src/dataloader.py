

import math
import torch

from typing import Iterable, Iterator, Optional, Sequence

from  chemicalx.data import BatchGenerator, DrugComb, DrugCombDB, OncoPolyPharmacology
from chemicalx.data import DrugPairBatch
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd



from  chemicalx.data.labeledtriples import LabeledTriples




    



'''
    Stratified KFold should be subclassed from `chemicalx` package

    As of now :
        1. Write Everything in terms of pytorch
'''
class StratifyLoader():
    

    def __init__(self,args):
        self.model = args.model
        self.batch_size = args.batch_size
        self.dataset = args.dataset
        self.drug_set, self.context_set, self.train, self.final_test = self.get_datasets()


    def get_final_test_loader(self):
        test_loader = BatchGenerator(batch_size=self.batch_size,context_features=True, drug_features=True,drug_molecules=True,
                                      context_feature_set=self.context_set, drug_feature_set=self.drug_set, labeled_triples=self.final_test)
        return test_loader



    def get_dataset_split(self):
        self.X  = np.array(self.train.drop("label",axis=1))
        self.y  = np.array(self.train["label"])
        skf = StratifiedKFold(n_splits=10,shuffle=True) # need to add this as user hyperparameter
        return skf.split(self.X,self.y)
    
    def get_stratify_dataloader(self, train_idx, test_idx):
        X_train, X_test = self.X[train_idx],self.X[test_idx]
        y_train, y_test = self.y[train_idx],self.y[test_idx]
        X_train = pd.DataFrame(X_train,columns=["drug_1", "drug_2", "context"])
        X_test = pd.DataFrame(X_test,columns=["drug_1", "drug_2", "context"])

        X_train["label"] = y_train
        train = LabeledTriples(X_train)

        X_test["label"] = y_test
        test = LabeledTriples(X_test)

        train_loader = BatchGenerator(batch_size=self.batch_size,context_features=True, drug_features=True,drug_molecules=True,
                                      context_feature_set=self.context_set, drug_feature_set=self.drug_set, labeled_triples=train)
        
        test_loader = BatchGenerator(batch_size=self.batch_size,context_features=True, drug_features=True,drug_molecules=True,
                                      context_feature_set=self.context_set, drug_feature_set=self.drug_set, labeled_triples=test)
        
        return train_loader,test_loader
    


    def get_datasets(self):

        if self.dataset == "drugcomb":
            loader = DrugComb()
        elif self.dataset == "drugcombdb":
            loader = DrugCombDB()
        # Oncology Pharma Dataset not working
        # elif self.dataset == "oncolypharma":
        #     loader = OncoPolyPharmacology()
        else:
            print(f"The given dataset {self.dataset} is not a part of chemicalx, provide the 3 synergy prediction tasks")
            exit()

        drug_set = loader.get_drug_features()
        context_set = loader.get_context_features()
        triplets = loader.get_labeled_triples()
        train, test = triplets.train_test_split(train_size=0.9)
        return drug_set, context_set, train.data, test













class DataLoader():
    
    def __init__(self,args):
        self.model = args.model
        self.dataset = args.dataset
        self.validation = args.validation
        drug_set, context_set, train, self.test = self.get_datasets()

        if self.validation:
            self.loader = StratifyLoader(args)
        else:
            self.train_loader = BatchGenerator(batch_size=args.batch_size,context_features=True, drug_features=True,drug_molecules=True,
                                      context_feature_set=context_set, drug_feature_set=drug_set, labeled_triples=train)
            self.test_loader =  BatchGenerator(batch_size=args.batch_size,context_features=True, drug_features=True,drug_molecules=True,
                                      context_feature_set=context_set, drug_feature_set=drug_set, labeled_triples=self.test)
       
    
    def get_datasets(self):
        if self.dataset == "drugcomb":
            loader = DrugComb()
        elif self.dataset == "drugcombdb":
            loader = DrugCombDB()
        # Oncology Pharma Dataset not working
        # elif self.dataset == "oncolypharma":
        #     loader = OncoPolyPharmacology()
        else:
            print(f"The given dataset {self.dataset} is not a part of chemicalx, provide the 3 synergy prediction tasks")
            exit()

        drug_set = loader.get_drug_features()
        context_set = loader.get_context_features()
        triplets = loader.get_labeled_triples()
        train, test = triplets.train_test_split(train_size=0.8)
        return drug_set, context_set, train, test
    
        






        








