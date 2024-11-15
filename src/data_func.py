import os
import torch
import zipfile
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def extract_zipfile(zip_file_path, extract_dir):
    """
    Extracts the contents of a ZIP file to a specified directory.

    Args:
        zip_file_path (str): Path to the ZIP file.
        extract_dir (str): Directory where the files will be extracted.

    Prints:
        - Names of the files inside the ZIP.
        - Confirmation message after extraction.
    """
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_content = zip_ref.namelist()
        print(f"Files inside the ZIP file: {zip_content}")
        zip_ref.extractall(extract_dir)
    print(f"Files extracted at path: {extract_dir}")

# -------------------------------------------------------------------------------------------------------

def handle_data(file_paths: list, to_save: bool = True):
    """
    Processes a list of text files containing news and sentiment data, 
    and saves it as a CSV file or returns a dictionary.

    Args:
        file_paths (list): List of file paths to process.
        to_save (str or bool): Path to save the resulting DataFrame as a CSV file. 
                               If False, returns a dictionary instead.

    Returns:
        pd.DataFrame or dict: Processed data in DataFrame or dictionary format.

    Prints:
        - Success message for each file processed.
        - Error message if a file cannot be processed.
        - Message confirming data is saved or processed.
    """
    content = {'news': [], 'sentiment': []}
    for file_path in file_paths:
        print(f"Handling file: {file_path}")
        try:
            with open(file_path, 'r') as file:
                data = file.read()
            data_mid = [line.split("@") for line in data.split('\n') if '@' in line]
            for news, sentiment in data_mid:
                content['news'].append(news)
                content['sentiment'].append(sentiment)
            print(f"{file_path} handled successfully")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    if to_save:
        to_return = pd.DataFrame(content)
        to_return.to_csv(to_save, index=False)
        print(f"Data saved as {to_save}")
    else:
        to_return = content
        print("Data processed but not saved as DataFrame")
    return to_return

# -------------------------------------------------------------------------------------------------------

def reencode_file_to_utf8(filepath: str):
    """
    Re-encodes a file from ISO-8859-1 to UTF-8.

    Args:
        filepath (str): Path to the file to be re-encoded.

    Prints:
        - Confirmation message after re-encoding.
    """
    with open(filepath, 'r', encoding='ISO-8859-1') as f:
        content = f.read()
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"File re-encoded to UTF-8 and saved at: {filepath}")

# -------------------------------------------------------------------------------------------------------

def save_corpus(data: pd.DataFrame, filepath: str) -> None:
    """
    Saves a corpus of news text to a file and ensures it is UTF-8 encoded.

    Args:
        data (pd.DataFrame): DataFrame containing a 'news' column with text data.
        filepath (str): Path to save the corpus.

    Prints:
        - Confirmation message after saving.
    """
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)
    text = ' '.join(data['news'].tolist())
    with open(filepath, 'w') as f:
        f.write(text)
    print(f"News Corpus saved at: {filepath}")
    reencode_file_to_utf8(filepath)

# -------------------------------------------------------------------------------------------------------

def get_tokenizer(train: bool = True, data_path: list = None, vocab_size: int = None, 
                  tokenizer_path: str = "models/finbpe_tokenizer.json", special_tokens: list = None):
    """
    Trains or loads a BPE tokenizer for text tokenization.

    Args:
        train (bool): If True, trains a new tokenizer; if False, loads an existing tokenizer.
        data_path (list): List of file paths for training the tokenizer (required if train=True).
        vocab_size (int): Vocabulary size for the tokenizer (required if train=True).
        tokenizer_path (str): Path to save/load the tokenizer.
        special_tokens (list): Special tokens to include in the tokenizer.

    Returns:
        Tokenizer: The trained or loaded tokenizer.

    Prints:
        - Confirmation message after saving/loading the tokenizer.
    """
    if train:
        special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]"] if special_tokens is None else special_tokens
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=special_tokens)
        tokenizer.train(data_path)
        tokenizer.save(tokenizer_path)
        print(f"Tokenizer saved at: {tokenizer_path}")
    else:
        tokenizer = Tokenizer.from_file(tokenizer_path)
        print(f"Tokenizer loaded from {tokenizer_path}")
    return tokenizer

if __name__ == '__main__':
    pass
