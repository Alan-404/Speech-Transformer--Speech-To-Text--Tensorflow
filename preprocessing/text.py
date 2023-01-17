from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import os
import pickle
import re

class TextProcessor:
    def __init__(self, tokenizer_path: str):
        self.tokenizer_path = tokenizer_path
        self.tokenizer = None

    def preprocess_sequence(self, sequence: str):
        sequence = sequence.strip()
        sequence = re.sub(r"([?.!,¿])", "", sequence)
        sequence = re.sub(r'\s\s+', ' ', sequence)
        return sequence

    def __load_tokenizer(self):
        if os.path.exists(self.tokenizer_path + "/tokenizer.pickle") == True:
            with open(self.tokenizer_path + "/tokenizer.pickle", 'rb') as handle:
                self.tokenizer = pickle.load(handle)
        else:
            self.tokenizer = Tokenizer(filters=None)
    
    def __save_tokenizer(self):
        if os.path.exists(self.tokenizer_path) == False:
            os.mkdir(self.tokenizer_path)
        with open(self.tokenizer_path + "/tokenizer.pickle", 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __build_tokenizer(self, sequences: list):
        for i in range(len(sequences)):
            sequences[i] = self.preprocess_sequence(sequence=sequences[i])
            sequences[i] = "{} {} {}".format("__CLS__", sequences[i], "__SEP__")
        self.tokenizer.fit_on_texts(sequences)

    def process(self, sequences: list, max_length: int, padding: str = "post", truncating: str = "post"):
        self.__load_tokenizer()
        if len(self.tokenizer.word_counts) == 0:
            self.__build_tokenizer(sequences)
            self.__save_tokenizer()
        digit_sequences = self.tokenizer.texts_to_sequences(sequences)
        padded_sequences = pad_sequences(digit_sequences, maxlen=max_length, padding=padding, truncating=truncating)

        return padded_sequences
