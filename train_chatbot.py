import json
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
nltk.download('punkt')

ps = PorterStemmer()

'''
Preprocessing the data from the json file
'''

with open('intents.json') as file:
    data = json.load(file)



words = [] # list of words
classes = [] # intent tags
documents = [] # list of all (processed pattern, intent tag) pairs (tokens and associated tags)
word_stems = [] # all unique stemmed+lowercase words (model vocabulary)

for intent in data['intents']:
    for pattern in intent['patterns']:
        tokens = word_tokenize(pattern) # grab the individual words from the list of patterns
        processed_tokens = []
        
        for token in tokens: # process the words (tokens) by taking them to lowercase and stripping them down to their stems
            lower = token.lower()
            stem = ps.stem(lower)
            processed_tokens.append(stem) # add the processed words to a list
            if stem not in word_stems:
                word_stems.append(stem) # add non-duplicate word stems to a list

        words.extend(processed_tokens) # add the processed tokens to a list

        documents.append((processed_tokens, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

word_stems = sorted(set(word_stems))
classes = sorted(set(classes))

'''
Creating the bag-of-words vector
'''

training_data = []

for processed_token, intent in documents:
    bow_vector = [0] * len(word_stems)
    for token in processed_token:
        if token in word_stems:
            bow_vector[word_stems.index(token)] = 1

    label_vector = [0] * len(classes) # intent labels
    label_index = classes.index(intent) # label index
    label_vector[label_index] = 1 # one-hot encoding the labels

    training_data.append((bow_vector, label_vector))

