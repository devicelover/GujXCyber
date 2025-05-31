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

def tokenize_text(text):
    # Ensure text is a string; if NaN, convert to empty string
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    tokens = word_tokenize(text)
    return tokens


def main():
    # Load the preprocessed data
    df = pd.read_csv("data/cc.csv")
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
