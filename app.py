from flask import Flask, request, render_template
import torch
from transformers import AutoTokenizer
from params import LoraBertParams
from src.helpers import ImageToText
from infrence.helpers import get_trnd_model, tokenize_text, get_sentiment

# Initialize app
app = Flask(__name__)

# Initialize the text extraction and sentiment analysis objects
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
params = LoraBertParams()
extracter = ImageToText()
tokenizer = AutoTokenizer.from_pretrained(params.MODEL_NAME)
finbert = get_trnd_model(params=params, device=device)

# Flask route for the main page
@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    extracted_text = None

    if request.method == "POST":
        # Check if a text input was provided
        user_text = request.form.get("user_text")

        if user_text:
            # Use provided text for sentiment analysis
            extracted_text = user_text
        else:
            # Check if an image file was uploaded
            img = request.files.get("image_file")
            if img:
                # Extract text from the image
                extracted_text = extracter.img_to_str(img)

        # check if the user has inputed something and get the sentiment for it
        if extracted_text:
            ids, mask = tokenize_text(extracted_text, tokenizer=tokenizer, params=params, device=device)
            sentiment = get_sentiment(finbert, ids, mask, params)

    # Rendering the result
    return render_template("index.html", sentiment=sentiment, extracted_text=extracted_text)

if __name__ == "__main__":
    app.run(debug=True)
