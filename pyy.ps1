# create_python_files.ps1
# This script creates the necessary Python files for the Crime Classification AI project.

# Create preprocessing.py
$preprocessingContent = @"
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.utils import resample

# Download stopwords if not already downloaded
nltk.download('stopwords')

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters, punctuation, and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize and remove stopwords
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    required_columns = ["category", "sub_category", "crime_description"]
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"Missing expected column: {col}")
    df.drop_duplicates(inplace=True)
    df.dropna(subset=required_columns, inplace=True)
    df['crime_description'] = df['crime_description'].apply(clean_text)
    return df

def balance_dataset(df):
    # Oversample minority classes for the "category" column
    max_size = df['category'].value_counts().max()
    lst = [df]
    for class_index, group in df.groupby('category'):
        lst.append(resample(group,
                            replace=True,
                            n_samples=max_size - len(group),
                            random_state=42))
    df_balanced = pd.concat(lst)
    return df_balanced.reset_index(drop=True)

if __name__ == "__main__":
    # Example usage:
    file_path = "data/crime_data.csv"  # Update path as needed
    df = load_and_clean_data(file_path)
    df = balance_dataset(df)
    df.to_csv("data/cleaned_crime_data.csv", index=False)
"@

Set-Content -Path "preprocessing.py" -Value $preprocessingContent
Write-Output "Created preprocessing.py"

# Create word2vec_train.py
$word2vecTrainContent = @"
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import re
import string

nltk.download('punkt')

def tokenize_text(text):
    # Remove punctuation, convert to lowercase, and tokenize
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    tokens = word_tokenize(text)
    return tokens

def train_word2vec(sentences, vector_size=200, window=5, min_count=1, sg=1, epochs=10):
    # sg=1 for Skip-gram (set sg=0 for CBOW)
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, sg=sg)
    model.train(sentences, total_examples=len(sentences), epochs=epochs)
    return model

def main():
    # Load the preprocessed data
    df = pd.read_csv("data/cleaned_crime_data.csv")
    # Tokenize each crime description into a list of words
    sentences = df['crime_description'].apply(tokenize_text).tolist()
    # Train Word2Vec model from scratch
    w2v_model = train_word2vec(sentences)
    # Save the model
    w2v_model.save("models/custom_word2vec.model")
    # Optionally, save the vectors in text format
    w2v_model.wv.save_word2vec_format("models/custom_word2vec.txt", binary=False)

if __name__ == "__main__":
    main()
"@

Set-Content -Path "word2vec_train.py" -Value $word2vecTrainContent
Write-Output "Created word2vec_train.py"

# Create train_model.py
$trainModelContent = @"
import pandas as pd
import numpy as np
import joblib
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

def tokenize_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    tokens = word_tokenize(text)
    return tokens

def get_avg_word2vec(text, model, vector_size):
    tokens = tokenize_text(text)
    # Use only tokens present in the model's vocabulary
    valid_tokens = [token for token in tokens if token in model.wv]
    if not valid_tokens:
        return np.zeros(vector_size)
    vectors = [model.wv[token] for token in valid_tokens]
    return np.mean(vectors, axis=0)

def create_features(df, w2v_model, vector_size):
    # Generate averaged Word2Vec features for each crime description
    features = df['crime_description'].apply(lambda x: get_avg_word2vec(x, w2v_model, vector_size))
    return np.vstack(features.values)

def train_lightgbm(X, y, model_params, model_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LGBMClassifier(**model_params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric="multi_logloss")
    joblib.dump(model, f"models/{model_name}.joblib")
    y_pred = model.predict(X_test)
    print(f"Classification Report for {model_name}:\n")
    print(classification_report(y_test, y_pred))
    return model

def main():
    # Load the cleaned and balanced data
    df = pd.read_csv("data/cleaned_crime_data.csv")
    
    # Load the custom Word2Vec model
    w2v_model = Word2Vec.load("models/custom_word2vec.model")
    vector_size = w2v_model.vector_size

    # Create features using averaged Word2Vec embeddings
    X_w2v = create_features(df, w2v_model, vector_size)

    # Alternative: Create features using TF-IDF
    tfidf = TfidfVectorizer()
    X_tfidf = tfidf.fit_transform(df['crime_description']).toarray()
    joblib.dump(tfidf, "models/tfidf_vectorizer.joblib")

    # Encode labels for category and sub_category
    from sklearn.preprocessing import LabelEncoder
    cat_encoder = LabelEncoder()
    subcat_encoder = LabelEncoder()
    df['category_enc'] = cat_encoder.fit_transform(df['category'])
    df['sub_category_enc'] = subcat_encoder.fit_transform(df['sub_category'])
    joblib.dump(cat_encoder, "models/category_encoder.joblib")
    joblib.dump(subcat_encoder, "models/sub_category_encoder.joblib")

    # LightGBM parameters
    params = {
        "n_estimators": 300,
        "learning_rate": 0.03,
        "class_weight": "balanced",
        "num_leaves": 31,
        "max_depth": -1,
        "eval_metric": "multi_logloss"
    }

    print("Training category model using Word2Vec features...")
    train_lightgbm(X_w2v, df['category_enc'], params, "category_model")

    print("Training sub-category model using Word2Vec features...")
    train_lightgbm(X_w2v, df['sub_category_enc'], params, "sub_category_model")

    # Optionally, train using TF-IDF features as an alternative
    print("Training category model using TF-IDF features...")
    train_lightgbm(X_tfidf, df['category_enc'], params, "category_model_tfidf")

    print("Training sub-category model using TF-IDF features...")
    train_lightgbm(X_tfidf, df['sub_category_enc'], params, "sub_category_model_tfidf")

if __name__ == "__main__":
    main()
"@

Set-Content -Path "train_model.py" -Value $trainModelContent
Write-Output "Created train_model.py"

# Create app.py
$appContent = @"
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import re
import string
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

app = Flask(__name__)

# Load trained models and encoders
category_model = joblib.load("models/category_model.joblib")
sub_category_model = joblib.load("models/sub_category_model.joblib")
cat_encoder = joblib.load("models/category_encoder.joblib")
subcat_encoder = joblib.load("models/sub_category_encoder.joblib")
w2v_model = Word2Vec.load("models/custom_word2vec.model")
vector_size = w2v_model.vector_size

def tokenize_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    tokens = word_tokenize(text)
    return tokens

def get_avg_word2vec(text, model, vector_size):
    tokens = tokenize_text(text)
    valid_tokens = [token for token in tokens if token in model.wv]
    if not valid_tokens:
        return np.zeros(vector_size)
    vectors = [model.wv[token] for token in valid_tokens]
    return np.mean(vectors, axis=0)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if a CSV file was uploaded
        if 'file' in request.files and request.files['file'].filename != "":
            file = request.files['file']
            import pandas as pd
            df = pd.read_csv(file)
            predictions = []
            for desc in df.get('crime_description', []):
                if len(desc.split()) < 5:
                    predictions.append({"error": "Input too short", "description": desc})
                else:
                    embedding = get_avg_word2vec(desc, w2v_model, vector_size)
                    cat_pred = cat_encoder.inverse_transform([category_model.predict(embedding.reshape(1, -1))[0]])[0]
                    subcat_pred = subcat_encoder.inverse_transform([sub_category_model.predict(embedding.reshape(1, -1))[0]])[0]
                    predictions.append({
                        "description": desc,
                        "category": cat_pred,
                        "sub_category": subcat_pred
                    })
            return render_template("result.html", predictions=predictions)
        else:
            # Process text input
            description = request.form.get("description", "")
            if len(description.split()) < 5:
                return jsonify({"error": "Crime description must have at least 5 words."}), 400
            embedding = get_avg_word2vec(description, w2v_model, vector_size)
            cat_pred = cat_encoder.inverse_transform([category_model.predict(embedding.reshape(1, -1))[0]])[0]
            subcat_pred = subcat_encoder.inverse_transform([sub_category_model.predict(embedding.reshape(1, -1))[0]])[0]
            return render_template("result.html", description=description, category=cat_pred, sub_category=subcat_pred)
    return render_template("index.html")

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "An error occurred during processing."}), 500

if __name__ == "__main__":
    app.run(debug=True)
"@

Set-Content -Path "app.py" -Value $appContent
Write-Output "Created app.py"

Write-Output "All Python files have been created successfully."
