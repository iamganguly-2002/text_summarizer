from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

def create_tokenizer(lines, num_words=None):
    tokenizer = Tokenizer(num_words=num_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(lines)
    return tokenizer

def encode_and_pad(tokenizer, lines, max_len):
    sequences = tokenizer.texts_to_sequences(lines)
    return pad_sequences(sequences, maxlen=max_len, padding='post')
