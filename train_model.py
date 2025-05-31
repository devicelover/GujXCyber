import os
import pandas as pd
import numpy as np
import joblib
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
from text_preprocessor import TextPreprocessor

# Download necessary NLTK resources
nltk.download('punkt')

# --- Configuration Flags ---
# Set use_gpu to True if you want to run LightGBM on GPU (requires GPU-enabled LightGBM)
use_gpu = False  # Change to True if GPU is available

# Set augment_data to True if you want to perform data augmentation (requires data_augmentation.py)
augment_data = False

# Initialize text preprocessor
preprocessor = TextPreprocessor()

def get_avg_word2vec(text, model, vector_size):
    """
    Compute the average Word2Vec embedding for a given text.
    Assumes the text is already normalized.
    """
    tokens = word_tokenize(text)
    valid_tokens = [token for token in tokens if token in model.wv]
    if not valid_tokens:
        return np.zeros(vector_size)
    vectors = [model.wv[token] for token in valid_tokens]
    return np.mean(vectors, axis=0)

def create_features(df, w2v_model, vector_size):
    """
    Generate averaged Word2Vec features using the normalized text.
    """
    features = df['normalized_text'].apply(lambda x: get_avg_word2vec(x, w2v_model, vector_size))
    return np.vstack(features.values)

def augment_dataset(df, num_aug=2):
    """
    Optionally augment the dataset using data augmentation techniques.
    This function assumes you have a data_augmentation.py module with an augment_text function.
    """
    from data_augmentation import augment_text  # Ensure this module is available
    augmented_rows = []
    for idx, row in df.iterrows():
        original_text = row['normalized_text']
        aug_texts = augment_text(original_text, num_aug=num_aug)
        for aug in aug_texts:
            new_row = row.copy()
            new_row['normalized_text'] = aug
            augmented_rows.append(new_row)
    augmented_df = pd.DataFrame(augmented_rows)
    combined_df = pd.concat([df, augmented_df], ignore_index=True)
    return combined_df

def train_lightgbm(X_train, X_test, y_train, y_test, model_params, model_name):
    """
    Train a LightGBM classifier with the specified parameters, save the model, and print a classification report.
    """
    model = LGBMClassifier(**model_params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric="multi_logloss")
    joblib.dump(model, f"models/{model_name}.joblib")
    y_pred = model.predict(X_test)
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_test, y_pred))
    return model

def main():
    # Load your dataset (CSV must include columns: category, sub_category, crime_description)
    df = pd.read_csv("data/cc.csv")

    # Create a new column with normalized text using our preprocessor
    df['normalized_text'] = preprocessor.transform(df['crime_description'])

    # Optionally augment the data
    if augment_data:
        print("Augmenting dataset...")
        df = augment_dataset(df, num_aug=2)
        print(f"Dataset augmented. New shape: {df.shape}")

    # Load the custom Word2Vec model (trained from scratch)
    w2v_model = Word2Vec.load("models/custom_word2vec.model")
    vector_size = w2v_model.vector_size

    # Cache embeddings to avoid recomputation on subsequent runs
    embeddings_path = "data/embeddings.npy"
    if os.path.exists(embeddings_path):
        X_w2v = np.load(embeddings_path)
        print("Loaded cached embeddings.")
    else:
        X_w2v = create_features(df, w2v_model, vector_size)
        np.save(embeddings_path, X_w2v)
        print("Computed and cached embeddings.")

    # Encode labels into numeric values
    from sklearn.preprocessing import LabelEncoder
    cat_encoder = LabelEncoder()
    subcat_encoder = LabelEncoder()
    df['category_enc'] = cat_encoder.fit_transform(df['category'])
    df['sub_category_enc'] = subcat_encoder.fit_transform(df['sub_category'])
    joblib.dump(cat_encoder, "models/category_encoder.joblib")
    joblib.dump(subcat_encoder, "models/sub_category_encoder.joblib")

    # Perform a single train-test split for both targets
    X_train, X_test, y_cat_train, y_cat_test, y_sub_train, y_sub_test = train_test_split(
        X_w2v, df['category_enc'], df['sub_category_enc'], test_size=0.2, random_state=42)

    # LightGBM parameters
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
        # GPU-specific parameters for LightGBM (requires GPU-enabled LightGBM)
        params["device"] = "gpu"
        params["gpu_platform_id"] = 0
        params["gpu_device_id"] = 0

    print("Training category model using Word2Vec features...")
    train_lightgbm(X_train, X_test, y_cat_train, y_cat_test, params, "category_model")

    print("Training sub-category model using Word2Vec features...")
    train_lightgbm(X_train, X_test, y_sub_train, y_sub_test, params, "sub_category_model")

if __name__ == "__main__":
    main()
