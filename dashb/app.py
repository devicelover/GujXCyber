import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_file
import numpy as np
import re
import string
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')  # Ensure this resource is available

# Initialize Flask app
app = Flask(__name__)

# Load models and encoders
print("Loading new models and encoders...")
category_model = joblib.load("models/category_model.joblib")
sub_category_model = joblib.load("models/sub_category_model.joblib")
category_encoder = joblib.load("models/category_encoder.joblib")
sub_category_encoder = joblib.load("models/sub_category_encoder.joblib")
w2v_model = Word2Vec.load("models/custom_word2vec.model")
vector_size = w2v_model.vector_size

# Setup upload and processed folders
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def tokenize_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    tokens = word_tokenize(text)
    return tokens

def get_avg_word2vec(text):
    """Computes the average Word2Vec embedding for the given text."""
    tokens = tokenize_text(text)
    valid_tokens = [token for token in tokens if token in w2v_model.wv]
    if not valid_tokens:
        return np.zeros(vector_size)
    vectors = [w2v_model.wv[token] for token in valid_tokens]
    return np.mean(vectors, axis=0)

def predict_crime_category(description):
    """Predicts the category and sub-category for a given crime description."""
    embedding = get_avg_word2vec(description).reshape(1, -1)
    category_pred = category_model.predict(embedding)
    sub_category_pred = sub_category_model.predict(embedding)
    category = category_encoder.inverse_transform(category_pred)[0]
    sub_category = sub_category_encoder.inverse_transform(sub_category_pred)[0]
    return category, sub_category

@app.route("/", methods=["GET", "POST"])
def index():
    """Renders the homepage with input options."""
    if request.method == "POST":
        description = request.form.get("description")
        if not description or len(description.split()) < 5:
            return render_template("index.html", error="Description must be at least 5 words.")
        category, sub_category = predict_crime_category(description)
        return render_template("result.html", description=description, category=category, sub_category=sub_category)
    return render_template("index.html")

@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    """Handles batch file uploads and processes them."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file type. Please upload a CSV file.'})

    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)
    output_filepath = process_file(filename)

    # If process_file returns an error response (tuple), return it directly.
    if isinstance(output_filepath, tuple):
        return output_filepath

    return render_template("download.html", output_filename=os.path.basename(output_filepath))

def process_file(filepath):
    """Processes a CSV file to predict categories for each crime description."""
    print(f"Processing file: {filepath}")
    try:
        # Try reading with the default (C) engine
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading CSV with C engine: {e}")
        try:
            # Fall back to the Python engine and skip bad lines
            df = pd.read_csv(filepath, engine='python', on_bad_lines='skip')
        except Exception as e2:
            print(f"Error reading CSV with Python engine: {e2}")
            return jsonify({'success': False, 'error': f"Processing error: {str(e2)}"}), 500

    if 'crime_description' not in df.columns:
        return jsonify({'success': False, 'error': 'CSV must contain a "crime_description" column.'}), 400

    try:
        # Apply prediction row by row
        df['Predicted Category'] = df['crime_description'].apply(lambda x: predict_crime_category(x)[0])
        df['Predicted Sub-category'] = df['crime_description'].apply(lambda x: predict_crime_category(x)[1])
        output_filename = os.path.join(app.config['PROCESSED_FOLDER'], os.path.basename(filepath).replace('.csv', '_processed.csv'))
        df.to_csv(output_filename, index=False)
        print(f"File processed and saved: {output_filename}")
        return output_filename
    except Exception as e:
        print(f"Error processing CSV file: {e}")
        return jsonify({'success': False, 'error': f"Processing error: {str(e)}"}), 500

@app.route("/download/<filename>")
def download_file(filename):
    """Allows downloading the processed CSV file."""
    file_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "File not found", 404

if __name__ == "__main__":
    app.run(debug=True)
