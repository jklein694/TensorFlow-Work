import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter


lemmatizer = WordNetLemmatizer()
hm_lines = 10000000

def create_lexicon(pos, neg):
    lexicon = []
    for file in [pos, neg]:
        with open(file, 'r') as f:
            contents = f.readlines()
            for l in contents[:hm_lines]:
                # creates a list for each line
                all_words = word_tokenize(l.lower())
                # add that list to to the lexicon
                lexicon += list(all_words)

    # Seperates each word in the list
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    # Creates a dictionary for every word for how many times they appear
    # Exp. {'the': 5325}
    w_counts = Counter(lexicon)
    l2 = []
    for w in w_counts:
        # Filter out words that show up too often like 'the' and 'and'
        # Filter out words that are too rare and barely show up
        if 1000 > w_counts[w] > 50:
            l2.append(w)
    print(len(l2))
    return l2

def sample_handeling(sample, lexicon, classification):
    feature_set = []
    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]

            # create a list of zeros to create a word vector for each word in the list
            # exp. lexicon = [dog, meat, grass, park]
            # exp. sentence = [My dog and I went to the park]
            # exp. features = [1, 0, 0, 1]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    # return the index location for the word in the lexicon
                    index_value = lexicon.index(word.lower())
                    # add a 1 to the one hot list of words each time it occurs
                    features[index_value] += 1

            # convert features to list
            features = list(features)
            feature_set.append([features, classification])
    return feature_set

def create_features_set_and_labels(pos, neg, test_size=0.1):
    lexicon = create_lexicon(pos, neg)
    features = []
    features += sample_handeling('pos.txt', lexicon, [1, 0])
    features += sample_handeling('neg.txt', lexicon, [0, 1])
    random.shuffle(features)

    features = np.array(features)

    testing_size = int(test_size * len(features))

    train_x = list(features[:, 0][:-testing_size])
    train_y = list(features[:, 1][:-testing_size])

    test_x = list(features[:, 0][-testing_size:])
    test_y = list(features[:, 1][-testing_size:])

    return train_x, train_y, test_x, test_y

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = create_features_set_and_labels('pos.txt', 'neg.txt')
    with open('sentiment_set.pickle', 'wb') as f:
        pickle._dump([train_x, train_y, test_x, test_y], f)