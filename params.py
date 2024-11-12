from dataclasses import dataclass

RANK = 4  # Lora rank
MODEL_NAME = "huawei-noah/TinyBERT_General_4L_312D" # the model name from hugging face
DATA_PATH = r"data\finnews.csv"  # dataset path
TRND_BERT_PTH = "models\whtout-lora-trainedbert.pth"  # thr hugging face model trained with wrapper path
TRND_BERTLORA_PTH =  "models\wht-lora-trainedbert.pth"  # the final model saving path
CNM_PATH = "figures\confusion-mtx.png" # confusion matrix path
TSZ = 0.33       # testing size split
RANDOM_STATE = 42  # random state
BATCH_SIZE = 8   # batch size
EPOCHS = 4       # no of epochs to train  
LR = 0.001       # learning rate 
MAXLEN = 256     # the maximum sequence length
NUM_CLASSES = 3  # number of classes to predict
ENCODERS = {'neutral':0, 'positive':1, 'negative':2}  # the encoder for int to str converstion
LABEL_ENCODERS = {k:i for i,k in ENCODERS.items()}   # label encoders for str to int converstion

class LoraBertParams:
    RANK: int=RANK
    MODEL_NAME: str= MODEL_NAME
    DATA_PATH: str=DATA_PATH
    TRND_BERT_PTH: str=TRND_BERT_PTH
    TRND_BERTLORA_PTH: str= TRND_BERTLORA_PTH
    CNM_PATH: str=CNM_PATH
    TSZ: float=TSZ
    RANDOM_STATE: int = RANDOM_STATE
    BATCH_SIZE: int=BATCH_SIZE
    EPOCHS: int=EPOCHS
    LR: float=LR
    MAXLEN: int=MAXLEN
    NUM_CLASSES: int=NUM_CLASSES
    ENCODERS: dict=ENCODERS
    LABEL_ENCODERS: dict=LABEL_ENCODERS
