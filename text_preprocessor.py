# text_preprocessor.py
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download necessary NLTK resources (if not already downloaded)
nltk.download('stopwords')
nltk.download('wordnet')

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Extensive normalization dictionary for Hinglish/Hindi-English cybercrime inputs, including payment and amount terms
        self.normalization_dict = {
            # Common variations and informal spellings
            'kyu': 'why', 'ky': 'why', 'kyo': 'why', 'kuy': 'why',
            'hai': 'is', 'ha': 'is', 'h': 'is',
            'nhi': 'no', 'nahi': 'no', 'nahin': 'no', 'nah': 'no',
            'kaise': 'how', 'kaisee': 'how', 'kese': 'how', 'kaisa': 'how',
            'kya': 'what', 'kia': 'what', 'ky': 'what', 'kyya': 'what',
            'koe': 'anyone', 'koi': 'anyone',
            'kuch': 'some', 'kch': 'some', 'kucch': 'some',
            'mein': 'in', 'mai': 'in', 'me': 'in',
            'ho': 'is', 'hoon': 'am', 'hu': 'am', 'hun': 'am',
            'hindhi': 'hindi', 'hindi': 'hindi', 'hnd': 'hindi',
            'angrezi': 'english', 'eng': 'english', 'ang': 'english',
            'agar': 'if', 'agr': 'if', 'ager': 'if',
            'karna': 'do', 'krna': 'do', 'kr': 'do', 'kar': 'do', 'kre': 'do',
            'ke': 'of', 'k': 'of',
            'lye': 'for', 'liye': 'for', 'liye': 'for',
            'mera': 'my', 'meri': 'my', 'mra': 'my',
            'ter': 'your', 'tera': 'your', 'tere': 'your', 'tumhara': 'your', 'tumhare': 'your',
            'unka': 'their', 'unke': 'their', 'unk': 'their',
            'wo': 'they', 'woah': 'they', 'woh': 'they',
            'bataya': 'told', 'btaya': 'told',
            'batana': 'tell', 'bta': 'tell', 'bata': 'tell',
            'pata': 'know', 'ptha': 'know',
            'sath': 'with', 'saath': 'with', 'sathh': 'with',
            'takk': 'until', 'tak': 'until',
            'ab': 'now',
            'sab': 'all', 'sbb': 'all',
            'sabhi': 'everyone', 'sbhi': 'everyone',
            'jldi': 'quickly', 'jaldi': 'quickly',
            'km': 'work', 'kam': 'work', 'kaam': 'job',
            'paisa': 'money', 'pese': 'money',
            'aap': 'you', 'app': 'you', 'ap': 'you',
            'tum': 'you', 'tu': 'you', 'ty': 'you',
            'kyunki': 'because', 'kuki': 'because', 'kyuki': 'because',
            'j': 'go', 'jaa': 'go',
            'lo': 'take',
            'de': 'give', 'dediya': 'given', 'dedo': 'give',
            'mujhe': 'me', 'mjh': 'me', 'mjhe': 'me', 'mujh': 'me',
            'bhi': 'also',
            'bhae': 'brother', 'bhaiya': 'brother',
            'bhayo': 'brothers', 'bhaiyo': 'brothers',
            'baat': 'talk', 'bat': 'talk',
            'matlab': 'meaning', 'mtlb': 'meaning', 'mltb': 'meaning',
            'glat': 'wrong', 'galat': 'wrong', 'galath': 'wrong',
            'phn': 'phone', 'fone': 'phone', 'fon': 'phone',
            'fraud': 'fraud', 'frd': 'fraud',
            'phising': 'phishing',  # common misspelling
            'dhoka': 'cheating', 'dhokha': 'cheating',
            'scam': 'scam', 'scammm': 'scam',
            # Cybercrime & banking domain specifics
            'otp': 'OTP',
            'aadhar': 'aadhar', 'adhar': 'aadhar',
            'upi': 'upi', 'uupi': 'upi',
            'phonepe': 'phonepe', 'phnpe': 'phonepe',
            'bharatpe': 'bharatpe',
            'pytm': 'paytm', 'paytm': 'paytm',
            'gpay': 'gpay', 'googlepay': 'gpay', 'goglepay': 'gpay',
            'rs': 'rupees', 'rupe': 'rupees', 'rupay': 'rupees', 'rupee': 'rupees', 'rupia': 'rupees',
            # Informal/slang and texting terms common in India
            'yup': 'yes', 'ya': 'yes', 'yo': 'hey',
            'bhai': 'brother', 'yaar': 'friend',
            'chill maar': 'relax', 'mauj': 'fun',
            'timepass': 'entertainment', 'lol': 'laugh',
            'sahi hai': 'correct', 'fadu': 'awesome'
            # Expand further as needed...
        }

    def clean_text(self, text):
        # Ensure text is a string
        if not isinstance(text, str):
            text = '' if pd.isnull(text) else str(text)
        # Lowercase and remove punctuation (keeping alphanumeric and whitespace)
        text = text.lower()
        text = ''.join(char for char in text if char.isalnum() or char.isspace())
        words = text.split()
        # Remove stopwords and lemmatize
        filtered_words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        # Apply normalization using the dictionary
        normalized_words = [self.normalization_dict.get(word, word) for word in filtered_words]
        return ' '.join(normalized_words)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.clean_text(text) for text in X]

if __name__ == "__main__":
    # Testing the preprocessor
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    nltk.download('wordnet')
    nltk.download('stopwords')

    # Initialize components (for demonstration purposes)
    preprocessor = TextPreprocessor()
    sample_texts = [
        "Mujhe dikkat hui, kyu ki paise cut gaye! OTP galat aaya.",
        "Kaise hua fraud? Yeh sab bilkul galat hai. Upi, phonepe aur gpay sab sahi nahi hai."
    ]
    cleaned_texts = preprocessor.transform(sample_texts)
    for orig, clean in zip(sample_texts, cleaned_texts):
        print(f"Original: {orig}\nCleaned: {clean}\n")
