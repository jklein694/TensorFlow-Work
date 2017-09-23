
def to_words(file_path):

    ## NOT used yet for big data

    import pandas as pd
    import re
    from nltk.corpus import stopwords # Import the stop word list

    train = pd.read_csv(file_path, encoding='latin-1', header=None)
    train.columns = ['sentiment', 'id', 'date', 'no_query', 'user', 'comment']
    train = train.drop('id', 1)
    train = train.drop('date', 1)
    train = train.drop('no_query', 1)
    train = train.drop('user', 1)
    print(train.head())


    # Use regular expressions to do a find-and-replace
    new = []
    for line in train.comment:
        letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for
                              " ",                   # The pattern to replace it with
                              line)  # The text to search
        words = letters_only.lower().split()
        # 4. In Python, searching a set is much faster than searching
        #   a list, so convert the stop words to a set
        stops = set(stopwords.words("english"))
        #
        # 5. Remove stop words
        comment = [w for w in words if not w in stops]
        new.append(comment)
    train['cleaned'] = new
    train.to_csv('train.csv')
    return train

if __name__ == '__main__':
    data = to_words()