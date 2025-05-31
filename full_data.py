import re
import pandas as pd
import joblib
import numpy as np
from text_preprocessor import TextPreprocessor
from gensim.models import Word2Vec

# Load final trained models and encoders
cat_model = joblib.load("models/final_category_model.joblib")
sub_model = joblib.load("models/final_sub_category_model.joblib")
cat_encoder = joblib.load("models/final_category_encoder.joblib")
subcat_encoder = joblib.load("models/final_sub_category_encoder.joblib")

# Initialize preprocessor and custom Word2Vec model
preprocessor = TextPreprocessor()
w2v_model = Word2Vec.load("models/custom_word2vec.model")
vector_size = w2v_model.vector_size

# --- Normalization Dictionary (same as training) ---
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

def get_features(text):
    norm_text = preprocessor.transform([text])[0]
    norm_text = normalize_text(norm_text)
    return get_avg_word2vec(norm_text, w2v_model, vector_size)

# --- Rule Dictionary ---
RULES = {
    "Crime Against Women & Children": {
        "Cyber Stalking": ["stalking", "unknown number messages", "follow kar raha hai", "online tracking", "repeat calls"],
        "Cyber Bullying": ["cyberbullying", "trolling", "online harassment", "continuous abuse", "insulting comments"],
        "Child Pornography / CSAM": ["child abuse", "child pornography", "underage content", "illegal minor content"],
        "Child Sexual Exploitative Material (CSEM)": ["child sex abuse material", "csem", "child exploitation content", "kid porn"],
        "Publishing and Transmitting Obscene Material": ["obscene content", "vulgar video", "sexually explicit material", "offensive content shared"],
        "Computer Generated CSAM / CSEM": ["ai generated csam", "fake csam video", "digital child porn", "animated csem"],
        "Fake Social Media Profile": ["fake id", "duplicate account", "impersonation", "someone using my name"],
        "Cyber Blackmailing & Threatening": ["blackmail", "photo leak threat", "nude photo demand", "video viral kar dunga"],
        "Online Human Trafficking": ["human trafficking", "illegal selling of people", "online slavery"],
        "Defamation": ["fake news spread", "image kharab kar raha hai", "rumors about me"],
        "Others": ["general crime", "uncategorized", "unspecified"]
    },
    "Financial Crimes": {
        "Investment Scam / Trading Scam": ["fake investment", "ponzi scheme", "crypto fraud", "forex fraud"],
        "Online Job Fraud": ["job scam", "fake naukri", "job fraud", "employment fraud"],
        "Tech Support Scam": ["fake customer support", "tech help fraud", "fake IT support"],
        "Online Loan Fraud": ["loan scam", "fake loan approval", "loan verification fraud"],
        "Matrimonial / Romance Scam / Honey Trapping Scam": ["matrimonial fraud", "love scam", "honey trap", "shaadi ka fraud"],
        "Impersonation of Govt. Servant": ["fake govt officer", "govt impersonation fraud"],
        "Cheating by Impersonation (Other than Govt. Servant)": ["identity fraud", "fake identity", "someone pretending to be me"],
        "SIM Swap Fraud": ["sim blocked", "phone number hacked", "sim change fraud"],
        "Sextortion / Nude Video Call": ["nude video call", "blackmail after video", "girl asked for money"],
        "Aadhar Enabled Payment System (AEPS) Fraud": ["aadhaar fraud", "biometric scam", "aeps unauthorized transaction"],
        "Identity Theft": ["aadhaar fraud", "pan card chori", "passport fraud"],
        "Courier / Parcel Scam": ["parcel nahi aaya", "fake delivery call", "courier fraud"],
        "Phishing": ["fake bank call", "otp scam", "fraud email", "phissing"],
        "Online Shopping / E-Commerce Fraud": ["product nahi mila", "fake shopping site", "online order fraud"],
        "Advance Fee Fraud": ["advance payment scam", "money demanded in advance", "fee paid but no service"],
        "Real Estate / Rental Payment Fraud": ["fake landlord", "rent fraud", "property fraud"],
        "Others": ["uncategorized financial fraud", "unclassified scam"]
    },
    "Cyber Attack / Dependent Crimes": {
        "Malware Attack": ["virus infection", "malware attack", "computer infected"],
        "Ransomware Attack": ["ransomware", "files locked", "data encrypted for ransom"],
        "Hacking / Defacement": ["hacked", "account hacked", "unauthorized access"],
        "Data Breach / Theft": ["data leak", "personal information exposed", "data stolen"],
        "Tampering with Computer Source Documents": ["data tampered", "file manipulation", "code modified illegally"],
        "Denial of Service (DoS) / DDoS Attacks": ["server down", "ddos attack", "service disruption"],
        "SQL Injection": ["database hacked", "sql attack", "website security breach"],
        "Others": ["general cyber attack", "unclassified breach"]
    },
    "Other Cyber Crimes": {
        "Fake Profile": ["fake account", "duplicate profile", "impersonation"],
        "Phishing": ["email fraud", "fake login page", "otp ka scam"],
        "Cyber Terrorism": ["terror funding", "radicalization", "cyber terrorism"],
        "Social Media Account Hacking": ["facebook hack", "instagram hack", "twitter account compromised"],
        "Online Gambling / Betting Frauds": ["betting scam", "illegal gambling site", "sports betting fraud"],
        "Business Email Compromise / Email Takeover": ["email hacked", "corporate email fraud"],
        "Provocative Speech for Unlawful Acts": ["hate speech", "provocative messages", "anti-national speech"],
        "Matrimonial / Honey Trapping Scam": ["love scam", "romance fraud", "marriage fraud"],
        "Fake News": ["misinformation", "false news viral", "rumors spreading"],
        "Cyber Stalking / Bullying": ["harassment", "trolling", "cyberbullying"],
        "Defamation": ["false accusations", "public image ruined", "wrongful rumors"],
        "Cyber Pornography": ["illegal pornographic content", "obscene videos", "adult content fraud"],
        "Sending Obscene Material": ["obscene messages", "offensive material sent"],
        "Intellectual Property (IPR) Thefts": ["copyright violation", "stolen designs", "trademark fraud"],
        "Cyber Enabled Human Trafficking / Cyber Slavery": ["human trafficking online", "forced labor through internet"],
        "Cyber Blackmailing & Threatening": ["blackmail", "threats over internet", "extortion"],
        "Online Piracy": ["pirated movies", "software piracy", "illegal downloads"],
        "Spoofing": ["fake caller id", "identity spoofing", "spoofed messages"],
        "Others": ["uncategorized cyber crime", "unspecified online fraud"]
    }
}

def get_avg_word2vec(text, model, vector_size):
    tokens = regex_tokenize(text.lower())
    valid_tokens = [token for token in tokens if token in model.wv]
    if not valid_tokens:
        return np.zeros(vector_size)
    vectors = [model.wv[token] for token in valid_tokens]
    return np.mean(vectors, axis=0)

def get_features(text):
    norm_text = preprocessor.transform([text])[0]
    norm_text = normalize_text(norm_text)
    return get_avg_word2vec(norm_text, w2v_model, vector_size)

def hybrid_label(text):
    """
    First applies the rule-based classifier.
    If a rule fires, returns that label.
    Otherwise, falls back to ML model prediction.
    """
    rule_main, rule_sub = rule_based_classifier(text)
    if rule_main is not None and rule_sub is not None:
        return rule_main, rule_sub
    else:
        features = get_features(text)
        pred_main = cat_model.predict(features.reshape(1, -1))[0]
        pred_sub = sub_model.predict(features.reshape(1, -1))[0]
        main_label = cat_encoder.inverse_transform([pred_main])[0]
        sub_label = subcat_encoder.inverse_transform([pred_sub])[0]
        return main_label, sub_label

# Load final trained models and encoders for inference
cat_model = joblib.load("models/final_category_model.joblib")
sub_model = joblib.load("models/final_sub_category_model.joblib")
cat_encoder = joblib.load("models/final_category_encoder.joblib")
subcat_encoder = joblib.load("models/final_sub_category_encoder.joblib")

def rule_based_classifier(text):
    """
    Applies the rule-based classifier using the RULES dictionary.
    Checks each main category and its sub-categories for keywords.
    Returns (main_category, sub_category) if a match is found; otherwise, (None, None).
    """
    text = normalize_text(text)
    for main_cat, subcats in RULES.items():
        for sub_cat, keywords in subcats.items():
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', text):
                    return (main_cat, sub_cat)
    return (None, None)

def main():
    full_data_file = "data/unclassified_full.csv"  # Full dataset (e.g. 2.2M records)
    output_file = "data/fully_labeled_data.csv"

    print("Labeling full dataset from", full_data_file)
    chunksize = 50000  # Process in chunks (adjust based on memory)
    first_chunk = True

    for chunk in pd.read_csv(full_data_file, chunksize=chunksize):
        hybrid_main = []
        hybrid_sub = []
        for text in chunk["crime_description"]:
            main_cat, sub_cat = hybrid_label(text)
            hybrid_main.append(main_cat)
            hybrid_sub.append(sub_cat)
        chunk["final_category"] = hybrid_main
        chunk["final_sub_category"] = hybrid_sub

        if first_chunk:
            chunk.to_csv(output_file, index=False, mode='w')
            first_chunk = False
        else:
            chunk.to_csv(output_file, index=False, mode='a', header=False)
        print(f"Processed a chunk of {len(chunk)} records.")

    print("Full dataset labeled and saved to", output_file)

if __name__ == "__main__":
    main()
