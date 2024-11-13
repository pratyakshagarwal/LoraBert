import torch
import pandas as pd
import torch.nn as nn
import logging
import argparse
from transformers import AutoModel, AutoTokenizer

from params import LoraBertParams
from src.pipelines import data_pipeline, training_pipeline, evaluation_pipeline

# filtering warnings, placing seed, Set up logging
import warnings; warnings.filterwarnings("ignore")
torch.set_printoptions(precision=3, linewidth=125, sci_mode=False)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
_=torch.manual_seed(0)

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description="LoRA-enhanced FinBERT sentiment analysis pipeline")
    parser.add_argument('--device', type=str, default="cuda:0", 
                        help="Device to use, 'cuda:0' for GPU or 'cpu' for CPU")
    args = parser.parse_args()

    # Configuration setup
    params = LoraBertParams()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load model and tokenizer
    bert = AutoModel.from_pretrained(params.MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(params.MODEL_NAME)

    # Run data pipeline
    train_dl, test_dl = data_pipeline(params=params, tokenizer=tokenizer)

    # Run training pipeline
    model = training_pipeline(bert=bert, train_dl=train_dl, test_dl=test_dl,
                               params=params, device=device)

    # Run evaluation pipeline
    _, _, _ = evaluation_pipeline(model=model, dl=test_dl, params=params,
                                   tokenizer=tokenizer, device=device)