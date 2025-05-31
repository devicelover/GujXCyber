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
