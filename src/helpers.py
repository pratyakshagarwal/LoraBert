import pytesseract
from PIL import Image
import cv2
from transformers import pipeline
import warnings

# Suppress unnecessary warnings for a cleaner output
warnings.filterwarnings("ignore")

# Class to convert image to text using Tesseract OCR
class ImageToText:
    def __init__(self):
        # Set the path to the Tesseract executable
        # You may need to adjust the path depending on your system
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    # Method to convert image to string (text)
    def img_to_str(self, image_path): 
        """
        Convert the image located at 'image_path' into text using OCR (Optical Character Recognition).
        
        Parameters:
            image_path (str): The file path to the image you want to process.

        Returns:
            str: The extracted text from the image.
        """
        # Load the image using PIL (Python Imaging Library)
        image =  Image.open(image_path) 
        # Perform OCR using Tesseract and return the extracted text
        return pytesseract.image_to_string(image) 

# Class to analyze the sentiment of a given text using a pre-trained sentiment model
class TextSentiment:
    def __init__(self, model_name:str="ProsusAI/finbert"):
        """
        Initialize the sentiment analysis pipeline using the provided model name.
        
        Parameters:
            model_name (str): The name of the pre-trained model to use for sentiment analysis.
                              Defaults to "ProsusAI/finbert" for financial sentiment analysis.
        """
        # Initialize a sentiment-analysis pipeline using Hugging Face's Transformers library
        self.ppline = pipeline("sentiment-analysis", model=model_name)

    # Method to analyze the sentiment of a given text
    def get_sentiment(self, text): 
        """
        Get the sentiment of the provided text using the pre-trained sentiment analysis model.
        
        Parameters:
            text (str): The text whose sentiment you want to analyze.

        Returns:
            list: A list containing the sentiment and its associated score (confidence).
        """
        return self.ppline(text)
    
# Main program entry point
if __name__ == '__main__':
    # No code is running here yet, but you can add a sample usage of ImageToText and TextSentiment
    pass
