import torch

class FinBERTSentimentClassifier(torch.nn.Module):
    """
    A sentiment classification model based on FinBERT.

    This class wraps a pretrained FinBERT model and adds a classification head for 
    sentiment analysis, with three output classes (e.g., positive, neutral, negative).

    Attributes:
        finbert (PreTrainedModel): A pretrained FinBERT model from Hugging Face Transformers.
        classifier (torch.nn.Linear): A linear layer for sentiment classification.
    """
    def __init__(self, finbert):
        """
        Initialize the FinBERT sentiment classifier.

        Args:
            finbert (PreTrainedModel): A pretrained FinBERT model instance.
        """
        super(FinBERTSentimentClassifier, self).__init__()
        self.finbert = finbert  # Pretrained FinBERT model
        self.classifier = torch.nn.Linear(
            self.finbert.config.hidden_size, 3
        )  # Linear classification head for 3 sentiment classes

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the FinBERT sentiment classifier.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, seq_len).
            attention_mask (torch.Tensor): Attention masks of shape (batch_size, seq_len).

        Returns:
            logits (torch.Tensor): Logits for the three sentiment classes of shape (batch_size, 3).
        """
        # Get the pooled output from FinBERT
        outputs = self.finbert(input_ids=input_ids, attention_mask=attention_mask)
        pooler_output = outputs.pooler_output  # Shape: (batch_size, hidden_size)
        
        # Pass the pooled output through the classifier to get logits
        logits = self.classifier(pooler_output)
        return logits
