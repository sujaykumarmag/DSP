
from src.model.multi_deep_dds import MultiDeepDDS
import torch
from chemicalx.data import DrugCombDB

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,classification_report
from chemicalx.data import DrugCombDB, BatchGenerator


"""Example with DeepDDs."""

from chemicalx import pipeline
from chemicalx.data import DrugCombDB
from chemicalx.models import DeepDDS


def main():
    """Train and evaluate the DeepDDs model."""
    dataset = DrugCombDB()
    model = DeepDDS(
        context_channels=dataset.context_channels,
    )
    results = pipeline(
        dataset=dataset,
        model=model,
        batch_size=32,
        epochs=10,
        context_features=True,
        drug_features=True,
        drug_molecules=True,
    )
    results.summarize()


if __name__ == "__main__":
    main()



loader = DrugCombDB()


NUM_EPOCHS = 100
BATCH_SIZE = 32

context_set = loader.get_context_features()
drug_set = loader.get_drug_features()
triplets = loader.get_labeled_triples()

train, test = triplets.train_test_split(train_size=0.8)

generator = BatchGenerator(batch_size=32,context_features=True, drug_features=True,drug_molecules=True,
                           context_feature_set=context_set, drug_feature_set=drug_set, labeled_triples=train)
model = MultiDeepDDS(context_channels=loader.context_channels)

print(model)



optimizer = torch.optim.Adam(model.parameters())
model.train()
loss_fn = torch.nn.BCELoss()
for i in range(NUM_EPOCHS):
    tot_loss = 0
    for batch in generator:
        optimizer.zero_grad()
        logits = model(batch.context_features,batch.drug_molecules_left, batch.drug_molecules_right)
        loss = loss_fn(logits,batch.labels)
        tot_loss += loss
        loss.backward()
        optimizer.step()
    print(f"The loss for the Epoch [{i+1}/{NUM_EPOCHS}] is {tot_loss}")


model.eval()
generator.labeled_triples = test
preds = []
for batch in generator:
    logits = model(batch.context_features,batch.drug_molecules_left,batch.drug_molecules_right).detach().cpu().numpy()
    pred = np.round(logits)
    prediction = batch.identifiers
    prediction["Predictions"] = logits
    prediction["Real prediction"] = pred
    preds.append(prediction)
df = pd.concat(preds)


y_pred = np.array(df["Real prediction"])
y_real  = np.array(df["label"])
print(classification_report(y_pred,y_real))



torch.save(model,"model.t5")
torch.save(model.state_dict(), "model.pt")
