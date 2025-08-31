# src/preprocess.py
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources available (one-time in venv)
# nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, remove urls/non-letters, tokenize,
    remove stopwords, lemmatize. Returns cleaned string."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # remove URLs
    text = re.sub(r'http\S+|www\S+', ' ', text)
    # remove non-letters (retain spaces)
    text = re.sub(r'[^a-z\s]', ' ', text)
    # tokenize
    tokens = nltk.word_tokenize(text)
    # remove stopwords and short tokens, lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 1]
    return " ".join(tokens)
