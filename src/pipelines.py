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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def data_pipeline(params:LoraBertParams, tokenizer):
    logger.info("Running data pipeline")
    # Load and preprocess data
    data = pd.read_csv(params.DATA_PATH)
    data['sentiment'] = data['sentiment'].map(params.ENCODERS)

    # Get data loaders
    train_dl, test_dl = GET_DLS(data, tokenizer=tokenizer, maxlen=params.MAXLEN, tsz=params.TSZ, random_state=params.RANDOM_STATE).get_dls(params.BATCH_SIZE)
    return train_dl, test_dl

def training_pipeline(bert, train_dl, test_dl, params:LoraBertParams, device):
    logger.info("Running training pipeline")
    # Initialize model
    model = FinBERTSentimentClassifier(finbert=bert)
    if params.TRND_BERT_PTH:
        model.load_state_dict(torch.load(params.TRND_BERT_PTH))
        logger.info(f"Loaded pretrained model from {params.TRND_BERT_PTH}")

    # Apply LoRA and freeze parameters
    apply_lora(model=model, layers=(nn.Linear, nn.Embedding), rank=params.RANK, device=device)
    freeze_params(model=model)
    check_frozen_parameters(model=model)

    # Train the model
    train(train_loader=train_dl, test_dl=test_dl, net=model, epochs=params.EPOCHS, device=device, lr=params.LR)

    # Save trained model
    torch.save(model.state_dict(), params.TRND_BERTLORA_PTH)
    logger.info(f"Model trained and saved at: {params.TRND_BERTLORA_PTH}")
    return model

def evaluation_pipeline(model, dl, params:LoraBertParams, tokenizer, device):
    logger.info("Running evaluation pipeline")
    # Make predictions and evaluate
    predictions, targets = get_predictions(model, dl, device)
    plt_cmf(preds=predictions, targs=targets, nmc=params.NUM_CLASSES, save=params.CNM_PATH)
    logger.info(f"Confusion matrix saved at {params.CNM_PATH}")

    # Print sample predictions
    texts, true_labels, pred_labels = predict_sentiments(model=model, dl=dl, n=5, label_encoders=params.LABEL_ENCODERS, tokenizer=tokenizer)
    return texts, true_labels, pred_labels

if __name__ == '__main__':pass
