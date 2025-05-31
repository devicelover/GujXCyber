import os
import pandas as pd
import numpy as np
import joblib
import re
import string
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
from text_preprocessor import TextPreprocessor
import time
import warnings

warnings.filterwarnings('ignore')
nltk.download('punkt')

# --- Configuration ---
use_gpu = True   # Set to True if you have CUDA-enabled LightGBM installed
TRAIN_FILE = "data/cc.csv"    # 7k labeled data (must contain columns: crime_description, predicted_category, predicted_sub_cat)
EMBEDDINGS_PATH = "data/embeddings.npy"

# Initialize custom text preprocessor and custom Word2Vec model (trained from scratch)
preprocessor = TextPreprocessor()
w2v_model = Word2Vec.load("models/custom_word2vec.model")
vector_size = w2v_model.vector_size

# --- Normalization Dictionary ---
NORMALIZATION_DICT = {
    "upi": "upi",
    "uupi": "upi",
    "phonepe": "phonepe",
    "phnpe": "phonepe",
    "bharatpe": "bharatpe",
    "pytm": "paytm",
    "paytm": "paytm",
    "gpay": "gpay",
    "googlepay": "gpay",
    "goglepay": "gpay",
    "rs": "rupees",
    "rupe": "rupees",
    "rupay": "rupees",
    "rupee": "rupees",
    "rupia": "rupees",
    "paisa": "rupees"
}

def normalize_text(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    normalized_tokens = [NORMALIZATION_DICT.get(token, token) for token in tokens]
    return " ".join(normalized_tokens)

def regex_tokenize(text):
    return re.findall(r'\b\w+\b', text)

def get_avg_word2vec(text, model, vector_size):
    tokens = regex_tokenize(text.lower())
    valid_tokens = [token for token in tokens if token in model.wv]
    if not valid_tokens:
        return np.zeros(vector_size)
    vectors = [model.wv[token] for token in valid_tokens]
    return np.mean(vectors, axis=0)

def create_features(df, w2v_model, vector_size):
    # Apply custom preprocessor and then additional normalization.
    norm_texts = preprocessor.transform(df["crime_description"])
    norm_texts = [normalize_text(t) for t in norm_texts]
    df["normalized_text"] = norm_texts
    features = df["normalized_text"].apply(lambda x: get_avg_word2vec(x, w2v_model, vector_size))
    return np.vstack(features.values)

def plot_eval_results(evals_result, model_name):
    train_loss = evals_result['training']['multi_logloss']
    valid_loss = evals_result['valid_0']['multi_logloss']
    iterations = range(1, len(train_loss) + 1)
    plt.figure(figsize=(8,6))
    plt.plot(iterations, train_loss, label="Training LogLoss")
    plt.plot(iterations, valid_loss, label="Validation LogLoss")
    plt.xlabel("Iteration")
    plt.ylabel("Multi LogLoss")
    plt.title(f"Evaluation Metrics for {model_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"models/{model_name}_eval.png")
    plt.show()

def main():
    start_time = time.time()
    print("Loading training data from", TRAIN_FILE)
    df = pd.read_csv(TRAIN_FILE)
    print("Data shape:", df.shape)

    # Compute or load cached embeddings
    if os.path.exists(EMBEDDINGS_PATH):
        X = np.load(EMBEDDINGS_PATH)
        if X.shape[0] != len(df):
            print("Cached embeddings count mismatch. Recomputing embeddings...")
            X = create_features(df, w2v_model, vector_size)
            np.save(EMBEDDINGS_PATH, X)
        else:
            print("Loaded cached embeddings.")
    else:
        X = create_features(df, w2v_model, vector_size)
        np.save(EMBEDDINGS_PATH, X)
        print("Computed and cached embeddings.")

    # Encode labels (using columns predicted_category and predicted_sub_cat from your 7k file)
    from sklearn.preprocessing import LabelEncoder
    cat_encoder = LabelEncoder()
    subcat_encoder = LabelEncoder()
    df["category_enc"] = cat_encoder.fit_transform(df["predicted_category"])
    df["sub_category_enc"] = subcat_encoder.fit_transform(df["predicted_sub_cat"])
    joblib.dump(cat_encoder, "models/final_category_encoder.joblib")
    joblib.dump(subcat_encoder, "models/final_sub_category_encoder.joblib")

    # Split data (stratify by category)
    X_train, X_test, y_train_cat, y_test_cat = train_test_split(
        X, df["category_enc"], test_size=0.2, random_state=42, stratify=df["category_enc"])
    _, _, y_train_sub, y_test_sub = train_test_split(
        X, df["sub_category_enc"], test_size=0.2, random_state=42, stratify=df["sub_category_enc"])

    # Define LightGBM parameters
    params = {
        "n_estimators": 300,
        "learning_rate": 0.03,
        "class_weight": "balanced",
        "num_leaves": 31,
        "max_depth": -1,
        "eval_metric": "multi_logloss",
        "n_jobs": -1
    }
    if use_gpu:
        params["device"] = "gpu"
        params["gpu_platform_id"] = 0
        params["gpu_device_id"] = 0

    print("Training final main category model...")
    cat_model = LGBMClassifier(**params)
    cat_model.fit(X_train, y_train_cat, eval_set=[(X_test, y_test_cat)])
    joblib.dump(cat_model, "models/final_category_model.joblib")
    y_pred_cat = cat_model.predict(X_test)
    print("\nMain Category Classification Report:")
    print(classification_report(y_test_cat, y_pred_cat, zero_division=0))
    if hasattr(cat_model, 'evals_result_'):
        plot_eval_results(cat_model.evals_result_, "final_category_model")

    print("Training final sub-category model...")
    sub_model = LGBMClassifier(**params)
    sub_model.fit(X_train, y_train_sub, eval_set=[(X_test, y_test_sub)])
    joblib.dump(sub_model, "models/final_sub_category_model.joblib")
    y_pred_sub = sub_model.predict(X_test)
    print("\nSub-Category Classification Report:")
    print(classification_report(y_test_sub, y_pred_sub, zero_division=0))
    if hasattr(sub_model, 'evals_result_'):
        plot_eval_results(sub_model.evals_result_, "final_sub_category_model")

    end_time = time.time()
    print(f"Final models trained and saved. Total time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
