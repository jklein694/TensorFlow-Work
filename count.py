import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from collections import Counter

# Not used yet, for big data! Does not fit in repo

data = pd.read_csv('train.csv', index_col=1)
lexicon =np.array(data.cleaned)
lemmatizer = WordNetLemmatizer()
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
print(l2)
