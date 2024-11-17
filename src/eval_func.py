import torch
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassConfusionMatrix

def get_predictions(model, dl, device):
    """
    Generate predictions from the model on a given dataloader.

    Args:
        model (torch.nn.Module): The trained model for inference.
        dl (torch.utils.data.DataLoader): DataLoader for the test dataset.
        device (torch.device): The device ('cpu' or 'cuda') to run the model on.

    Returns:
        tuple: Predictions (list of true labels) and targets (list of predicted labels).
    """
    model.eval()  # Set the model to evaluation mode
    predictions, targets = [], []  # Initialize lists to store results
    with torch.no_grad():  # Disable gradient calculations for inference
        for data, labels in tqdm(dl, desc="Predicting"):
            # Move input data and labels to the device
            input_ids = data['input_ids'].to(device)
            attn_masks = data['attention_mask'].to(device)
            labels = labels.to(device)

            # Forward pass to get logits
            logits = model(input_ids, attn_masks)
            outputs = F.softmax(logits, dim=-1)  # Apply softmax to get probabilities
            pred = torch.argmax(outputs, dim=-1)  # Get predicted class by finding max probability

            # Store results in lists
            predictions.extend(labels.cpu().detach())
            targets.extend(pred.cpu().detach())
    return predictions, targets


def plt_cmf(preds, targs, nmc, save:str=None):
    """
    Plot and optionally save the confusion matrix.

    Args:
        preds (list): Predicted labels.
        targs (list): True labels.
        nmc (int): Number of classes.
        save (str, optional): Filepath to save the confusion matrix plot. Defaults to None.

    Returns:
        None
    """
    # Initialize multiclass confusion matrix computation
    mcm = MulticlassConfusionMatrix(num_classes=nmc)
    preds = torch.tensor(preds)
    targs = torch.tensor(targs)
    cmf = mcm(preds, targs)  # Compute confusion matrix

    # Plot confusion matrix as a heatmap
    sns.heatmap(cmf, annot=True, fmt='g', cmap='coolwarm', linewidths=0.5)
    if save is not None:
        plt.savefig(save)  # Save the plot if a path is provided
    plt.show()


def predict_sentiments(model, dl, n, label_encoders, tokenizer):
    """
    Randomly sample and display text sentiment predictions along with their real labels.

    Args:
        model (torch.nn.Module): The trained model for inference.
        dl (torch.utils.data.DataLoader): DataLoader for the dataset.
        n (int): Number of random samples to predict.
        label_encoders (dict): Mapping of sentiment classes to their human-readable labels.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to decode input text.

    Returns:
        tuple: Lists of texts, real sentiments, and predicted sentiments.
    """
    texts, real_sentiments, predicted_sentiments = [], [], []

    for data, labels in dl:
        # Randomly sample indices from the current batch
        random_indices = torch.randint(0, data['input_ids'].size(0), (n,))
        random_input_ids = data['input_ids'][random_indices]
        random_attention_mask = data['attention_mask'][random_indices]
        random_labels = labels[random_indices]

        # Forward pass to get logits for sampled data
        logits = model(random_input_ids, random_attention_mask)
        preds = torch.argmax(F.softmax(logits, dim=-1), dim=-1)  # Get predicted class

        # Decode input text and map labels to readable sentiments
        texts = [tokenizer.decode(ids.cpu(), skip_special_tokens=True) for ids in random_input_ids]
        real_sentiments = [label_encoders[label.item()] for label in random_labels]
        predicted_sentiments = [label_encoders[pred.item()] for pred in preds]
        break  # Exit loop after processing one batch

    # Print predictions for the sampled texts
    for i, (text, real, pred) in enumerate(zip(texts, real_sentiments, predicted_sentiments), 1):
        print(f"Text {i}: {text}")
        print(f"  Real Sentiment: {real}")
        print(f"  Predicted Sentiment: {pred}")
        print("-" * 50)

    return texts, real_sentiments, predicted_sentiments


if __name__ == '__main__':
    # Entry point for the script; can be used to test functions in isolation.
    pass
