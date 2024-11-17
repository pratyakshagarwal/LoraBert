import torch
import torch.nn as nn
import pandas as pd
import logging
import argparse

from params import LoraBertParams
from src.data import GET_DLS
from src.model import FinBERTSentimentClassifier
from src.loRA import apply_lora, freeze_params, check_frozen_parameters
from src.train_func import train
from src.eval_func import get_predictions, plt_cmf, predict_sentiments

# Set up logging to track and display progress and errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def data_pipeline(params:LoraBertParams, tokenizer):
    """
    Loads and processes the data, then returns training and testing data loaders.
    
    Parameters:
        params (LoraBertParams): The parameters containing paths and configuration.
        tokenizer: Tokenizer to process text input into token IDs.

    Returns:
        tuple: Training and testing data loaders.
    """
    logger.info("Running data pipeline")
    # Load the dataset from CSV
    data = pd.read_csv(params.DATA_PATH)
    # Map sentiment labels using encoder mappings
    data['sentiment'] = data['sentiment'].map(params.ENCODERS)

    # Generate the data loaders using the helper function
    train_dl, test_dl = GET_DLS(data, tokenizer=tokenizer, maxlen=params.MAXLEN, tsz=params.TSZ, random_state=params.RANDOM_STATE).get_dls(params.BATCH_SIZE)
    return train_dl, test_dl

def training_pipeline(bert, train_dl, test_dl, params:LoraBertParams, device):
    """
    Performs model training using the provided data and model configuration.

    Parameters:
        bert: The pre-trained BERT model to fine-tune.
        train_dl: The training data loader.
        test_dl: The testing data loader.
        params (LoraBertParams): The parameters for training (epochs, batch size, etc.).
        device: The device to run the model (e.g., "cuda" or "cpu").

    Returns:
        model: The trained sentiment classification model.
    """
    logger.info("Running training pipeline")
    # Initialize the FinBERT sentiment classifier model
    model = FinBERTSentimentClassifier(finbert=bert)
    
    # Load a pre-trained model if provided
    if params.TRND_BERT_PTH:
        model.load_state_dict(torch.load(params.TRND_BERT_PTH))
        logger.info(f"Loaded pretrained model from {params.TRND_BERT_PTH}")

    # Apply LoRA (Low-Rank Adaptation) and freeze model parameters to reduce trainable parameters
    apply_lora(model=model, layers=(nn.Linear, nn.Embedding), rank=params.RANK, device=device)
    freeze_params(model=model)
    check_frozen_parameters(model=model)

    # Train the model
    train(train_loader=train_dl, test_dl=test_dl, net=model, epochs=params.EPOCHS, device=device, lr=params.LR)

    # Save the trained model to a file
    torch.save(model.state_dict(), params.TRND_BERTLORA_PTH)
    logger.info(f"Model trained and saved at: {params.TRND_BERTLORA_PTH}")
    
    return model

def evaluation_pipeline(model, dl, params:LoraBertParams, tokenizer, device):
    """
    Evaluates the trained model using the provided data loader.

    Parameters:
        model: The trained sentiment analysis model.
        dl: The data loader for evaluation.
        params (LoraBertParams): The parameters for evaluation.
        tokenizer: Tokenizer used to encode the text.
        device: The device on which the model runs.

    Returns:
        tuple: Lists of sample texts, true sentiment labels, and predicted sentiment labels.
    """
    logger.info("Running evaluation pipeline")
    # Generate predictions and targets from the model
    predictions, targets = get_predictions(model, dl, device)
    
    # Generate and save the confusion matrix for evaluation
    plt_cmf(preds=predictions, targs=targets, nmc=params.NUM_CLASSES, save=params.CNM_PATH)
    logger.info(f"Confusion matrix saved at {params.CNM_PATH}")

    # Print a few sample predictions for inspection
    texts, true_labels, pred_labels = predict_sentiments(model=model, dl=dl, n=5, label_encoders=params.LABEL_ENCODERS, tokenizer=tokenizer)
    return texts, true_labels, pred_labels

if __name__ == '__main__':
    # No executable code in the main block, used as an entry point if needed
    pass