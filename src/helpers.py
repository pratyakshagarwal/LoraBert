import pytesseract
from PIL import Image
import cv2
from transformers import pipeline
import warnings; warnings.filterwarnings("ignore")

class ImageToText:
    def __init__(self):pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    def img_to_str(self, image_path): 
        image =  Image.open(image_path) # Load the image using PIL or OpenCV
        return pytesseract.image_to_string(image) # Perform OCR
    
class TextSentiment:
    def __init__(self, model_name:str="ProsusAI/finbert"):self.ppline = pipeline("sentiment-analysis", model=model_name)
    def get_sentiment(self, text): return self.ppline(text)
    
if __name__ == '__main__':
    pass