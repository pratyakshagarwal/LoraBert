# LoraBert

## Summary
LoraBERT is a fine-tuned model aimed at predicting the sentiment of financial texts, such as news articles and reports, with enhanced accuracy using LoRA (Low-Rank Adaptation) for model optimization.

## Demo
- The project includes an app.py file that spins up a local server using Flask. This app can take either text inputs (for sentiment analysis) or image inputs (extract the text using ocr from images) and returns the predicted sentiment for financial text/news.
- To run the project locally
```bash
cd LoraBert
python app.py
```
visit the localhost.

## Installation
- Create an enviorment
```
python -m venv env
```
- Activate the enviorment
```
myenv\Scripts\activate
```
- Run this script 
```
pip install -r requirements.txt
```


#### Will update the repo and readme later