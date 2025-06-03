import tensorflow as tf
import nltk
import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import pickle

# load the trained model
model = tf.keras.models.load_model("chatbot_model.h5")
model.save("chatbot_model.keras")  # convert model to native keras file format

# load the metadata
with open('metadata.pkl', 'rb') as file:
    data = pickle.load(file)
    word_stems = data['words']
    classes = data['classes']

ps = PorterStemmer()

# collect user input and formulate a response
user_tokens = []

while True:
    user_input = input("You: ")
    if (user_input.lower() == "quit"): # break if user enters quit
        break

    # separate user input into tokens
    user_tokens = word_tokenize(user_input)
    processed_tokens = []

    # preprocess tokens (stem, lower)
    for token in user_tokens:
        lower = token.lower()
        stem = ps.stem(lower)
        processed_tokens.append(stem)

    # create bag of words vector
    bow_vector = [0] * len(word_stems)

    for token in processed_tokens:
        if token in word_stems:
            bow_vector[word_stems.index(token)] = 1

    bow_vector = np.array(bow_vector).reshape(1, -1)

    


