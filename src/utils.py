
import torch
import matplotlib
matplotlib.use('TkAgg')

import os
import yaml
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score, confusion_matrix,precision_recall_curve, recall_score,precision_score, roc_auc_score, roc_curve, classification_report



def plot_embeddings(z_hidden):
    xs, ys = zip(*TSNE().fit_transform(z_hidden.detach().numpy()))
    plt.scatter(xs, ys)
    plt.show()


    




def save_results(self, training_metrics, model, y_pred, y_test):
    # name = model["name"]
    weights_dir = self.resultsdir+'train '+str(self.num)+"/weights/"
    results_dir = self.resultsdir+'train ' +str(self.num)+"/"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    with open(f"{results_dir}args.yaml", 'w') as f:
        yaml.dump(vars(self.args), f)

    torch.save(model.state_dict(), weights_dir+'best.pt')
    
    # Classification report
    # report = classification_report(y_pred, y_test)
    # print(report)

    metrics_df = pd.concat(training_metrics, ignore_index=True)
    metrics_df.to_csv(f'{results_dir}training_metrics.csv', index=False)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.savefig(os.path.join(results_dir, 'roc_curve.png'))
    plt.close()

    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.xticks([0, 1], ['0', '1'])  
    plt.yticks([0, 1], ['0', '1']) 
    # Add text annotations for TP and FP
    plt.text(0, 0, f'TN={cm[0, 0]}', ha='center', va='center', color='white' if cm[0, 0] > cm.max() / 2 else 'black')
    plt.text(1, 0, f'FP={cm[0, 1]}', ha='center', va='center', color='white' if cm[0, 1] > cm.max() / 2 else 'black')
    plt.text(0, 1, f'FN={cm[1, 0]}', ha='center', va='center', color='white' if cm[1, 0] > cm.max() / 2 else 'black')
    plt.text(1, 1, f'TP={cm[1, 1]}', ha='center', va='center', color='white' if cm[1, 1] > cm.max() / 2 else 'black')
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    plt.close()


    
    # Precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(os.path.join(results_dir, 'precision_recall_curve.png'))
    plt.close()



    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    f1score = f1_score(y_test,y_pred)

    
    # Save metrics to a CSV file
    metrics_dict = {
        'Accuracy': [accuracy],
        'Sensitivity': [sensitivity],
        'Precision': [precision],
        'Recall': [recall],
        'ROC AUC': [roc_auc],
        'F1 Score':[f1score]
    }
    
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_csv_path = os.path.join(results_dir, 'metrics.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)



