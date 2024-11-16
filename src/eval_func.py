import torch
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F 
from torchmetrics.classification import MulticlassConfusionMatrix

def get_predictions(model, dl, device):
    model.eval()
    predictions,targets = [],[]
    with torch.no_grad():
        for data, labels in tqdm(dl, desc="Predicting"):
            input_ids = data['input_ids'].to(device)
            attn_masks = data['attention_mask'].to(device)
            labels = labels.to(device)

            logits = model(input_ids, attn_masks)
            outputs = F.softmax(logits, dim=-1)
            pred = torch.argmax(outputs, dim=-1)

            predictions.extend(labels.cpu().detach())
            targets.extend(pred.cpu().detach())
    return predictions, targets

def plt_cmf(preds, targs, nmc, save:str=None):
    mcm = MulticlassConfusionMatrix(num_classes=3)
    preds = torch.tensor(preds)
    targs = torch.tensor(targs)
    cmf = mcm(preds, targs)
    sns.heatmap(cmf, annot=True, fmt='g', cmap='coolwarm', linewidths=0.5)
    if save is not None:plt.savefig(save)
    plt.show()

def predict_sentiments(model, dl, n, label_encoders, tokenizer):
    texts, real_sentiments, predicted_sentiments = [], [], []

    for data, labels in dl:
        random_indices = torch.randint(0, data['input_ids'].size(0), (n,))
        random_input_ids = data['input_ids'][random_indices]
        random_attention_mask = data['attention_mask'][random_indices]
        random_labels = labels[random_indices]

        logits = model(random_input_ids, random_attention_mask)
        preds = torch.argmax(F.softmax(logits, dim=-1), dim=-1)

        texts = [tokenizer.decode(ids.cpu(), skip_special_tokens=True) for ids in random_input_ids]
        real_sentiments = [label_encoders[label.item()] for label in random_labels]
        predicted_sentiments = [label_encoders[pred.item()] for pred in preds]
        break  

    for i, (text, real, pred) in enumerate(zip(texts, real_sentiments, predicted_sentiments), 1):
        print(f"Text {i}: {text}")
        print(f"  Real Sentiment: {real}")
        print(f"  Predicted Sentiment: {pred}")
        print("-" * 50)

    return texts, real_sentiments, predicted_sentiments

if __name__ == '__main__':
    pass