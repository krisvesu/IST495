import spacy

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    # Convert to lowercase and create spaCy doc
    doc = nlp(text.lower())

    # Remove stopwords, punctuation, and non-alphabetic tokens
    cleaned_tokens = [
        token.text for token in doc
        if token.is_alpha and not token.is_stop
    ]

    return cleaned_tokens