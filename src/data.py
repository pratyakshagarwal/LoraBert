import torch
from sklearn.model_selection import train_test_split

class FinData(torch.utils.data.Dataset):
    """
    A custom dataset class for financial data processing using PyTorch's Dataset API.
    
    This class is designed to work with text data for tasks like sentiment analysis,
    classification, or other NLP tasks. It tokenizes the input text using a specified tokenizer,
    prepares input IDs and attention masks for transformer models, and handles corresponding labels.

    Attributes:
        x (list or array-like): A list or array of input text samples.
        y (list or array-like): A list or array of labels corresponding to the text samples.
        tokenizer (PreTrainedTokenizer): A tokenizer instance from the Hugging Face Transformers library 
                                         or any compatible tokenizer.
        maxlen (int): The maximum length of input sequences after tokenization. Longer sequences 
                      are truncated, and shorter ones are padded.
        device (str): The device to which tensors will be moved (default: 'cpu').
    """

    def __init__(self, x, y, tokenizer, maxlen, device='cpu'):
        """
        Initialize the FinData dataset.

        Args:
            x (list or array-like): Input text samples.
            y (list or array-like): Corresponding labels for the text samples.
            tokenizer (PreTrainedTokenizer): Tokenizer for text data.
            maxlen (int): Maximum sequence length for tokenized inputs.
            device (str): Device for the tensors, either 'cpu' or 'cuda' (default: 'cpu').
        """
        self.x = x  # Input text data
        self.y = y  # Corresponding labels
        self.tokenizer = tokenizer  # Tokenizer to process text
        self.maxlen = maxlen  # Maximum sequence length
        self.device = device  # Device for tensor operations

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.x)

    def __getitem__(self, idx):
        """
        Retrieve a single data sample at the given index.

        Args:
            idx (int): Index of the desired sample.

        Returns:
            dict: A dictionary containing tokenized input IDs and attention mask.
            torch.Tensor: The label corresponding to the input sample.
        """
        # Tokenize the input text at the given index
        inputs = self.tokenizer(
            self.x[idx],
            return_tensors="pt",
            padding="max_length",  # Pad to the specified max length
            truncation=True,       # Truncate if the sequence exceeds max length
            max_length=self.maxlen # Maximum length of the sequence
        )
        
        # Extract tokenized input IDs and attention mask
        input_ids = inputs["input_ids"].squeeze()  # Remove extra dimension
        attention_mask = inputs["attention_mask"].squeeze()  # Remove extra dimension

        # Convert label to a tensor
        label = torch.tensor(self.y[idx], dtype=torch.long)

        # Return tokenized inputs and the corresponding label
        return {"input_ids": input_ids, "attention_mask": attention_mask}, label

class GET_DLS:
    """
    A utility class to prepare DataLoaders for training and testing datasets.

    This class splits the data into training and testing sets, tokenizes the text inputs,
    and creates PyTorch DataLoaders for efficient batching and shuffling.

    Attributes:
        data (pandas.DataFrame): Input dataset containing text and labels.
        tokenizer (PreTrainedTokenizer): Tokenizer instance for text processing.
        maxlen (int): Maximum sequence length for tokenized inputs.
        tsz (float): Test size proportion for splitting the dataset.
        random_state (int): Random state seed for reproducible splits.
    """

    def __init__(self, data, tokenizer, maxlen, tsz, random_state):
        """
        Initialize the GET_DLS class.

        Args:
            data (pandas.DataFrame): Input dataset with text and sentiment labels.
            tokenizer (PreTrainedTokenizer): Tokenizer to process the text.
            maxlen (int): Maximum sequence length for tokenized inputs.
            tsz (float): Proportion of data to be used for testing (between 0 and 1).
            random_state (int): Random seed for consistent data splitting.
        """
        # Initialize and store the input parameters
        self.data = data  # Dataset containing text and labels
        self.tokenizer = tokenizer  # Tokenizer for text processing
        self.maxlen = maxlen  # Maximum sequence length for tokenization
        self.tsz = tsz  # Test size proportion
        self.random_state = random_state  # Seed for reproducibility

    def get_dls(self, batch_size):
        """
        Create DataLoaders for training and testing data.

        Args:
            batch_size (int): Batch size for the DataLoaders.

        Returns:
            train_dl (DataLoader): DataLoader for the training dataset.
            test_dl (DataLoader): DataLoader for the testing dataset.
        """
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            self.data['news'],  # Text data
            self.data['sentiment'],  # Sentiment labels
            test_size=self.tsz,  # Proportion of test data
            random_state=self.random_state  # Seed for reproducibility
        )

        # Reset the indices of the resulting splits for consistency
        X_train, X_test, y_train, y_test = [
            df.reset_index(drop=True) for df in (X_train, X_test, y_train, y_test)
        ]

        # Create the training dataset and DataLoader
        train_dataset = FinData(X_train, y_train, self.tokenizer, self.maxlen)
        train_dl = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True  # Shuffle training data
        )

        # Create the testing dataset and DataLoader
        test_dataset = FinData(X_test, y_test, self.tokenizer, self.maxlen)
        test_dl = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=batch_size * 2,  # Larger batch size for testing
            shuffle=True  # Shuffle test data
        )

        # Return the training and testing DataLoaders
        return train_dl, test_dl
    
if __name__ == '__main__':
    pass