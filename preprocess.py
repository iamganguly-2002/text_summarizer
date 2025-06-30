import re
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub('"','', text)
    text = re.sub(r"'s\b","",text)
    text = re.sub("[^a-zA-Z]", " ", text)
    text = re.sub('[m]{2,}', 'mm', text)
    text = re.sub('\s+', ' ', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text.strip()
