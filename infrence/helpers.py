import torch
import torch.nn as nn
import torch.nn.functional as F
from params import LoraBertParams
from src.model import FinBERTSentimentClassifier
from src.loRA import apply_lora, freeze_params
from transformers import AutoTokenizer, AutoModel

def get_trnd_model(params:LoraBertParams, device:str):
    # load the bert-mini model from hugging face
    bert = AutoModel.from_pretrained(params.MODEL_NAME)

    # encapsule it to predict for three class classification and load the trained encapsulation model
    finbert = FinBERTSentimentClassifier(finbert=bert) 
    finbert.load_state_dict(torch.load(params.TRND_BERT_PTH))

    # apply lora with RANK to Linear and Embeddings layer and freeze params
    apply_lora(model=finbert, layers=(nn.Linear, nn.Embedding), rank=params.RANK, device=device)
    freeze_params(model=finbert)
    
    # load the trained lora bert model and return it
    finbert.load_state_dict(torch.load(params.TRND_BERTLORA_PTH))
    return finbert

def tokenize_text(sentence, tokenizer, params:LoraBertParams, device):
    # tokenizer the sentence with torch tensor and max length - 256 
    inputs = tokenizer(sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=params.MAXLEN)
    # config to device and return the ids and mask
    return inputs['input_ids'].to(device), inputs['attention_mask'].to(device)


def get_sentiment(model, ids:torch.Tensor, mask:torch.Tensor, params:LoraBertParams) -> str:
    # get the logit from model and apply softmax to it and get the index
    logits = model(ids, mask)
    outputs = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
    # label encode the index to get the string
    return params.LABEL_ENCODERS[int(outputs.item())]
