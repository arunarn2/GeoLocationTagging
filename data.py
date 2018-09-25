import numpy as np
import csv
import pandas as pd
import os
import logging
import keras.preprocessing.text
from keras.preprocessing import sequence

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)


def load_state_labels(userids, user_text_seq_full, user_loc, labels):
    y, X, userIDs = ([] for i in range(3))
    for user in userids:
        if 'UKN' not in user_loc[user][0]:
            if 'District of Columbia' in user_loc[user][2]:
                if user_loc[user][0] == 'United States of America':
                    X.append(user_text_seq_full[user])
                    userIDs.append(user)
                    if not user_loc[user][2] in labels:
                        labels.append(user_loc[user][2])
                    y.append(labels.index(user_loc[user][2]))
        if 'UKN' not in user_loc[user][1]:
            if user_loc[user][0] == 'United States of America':
                X.append(user_text_seq_full[user])
                userIDs.append(user)
                if not user_loc[user][1] in labels:
                    labels.append(user_loc[user][1])
                y.append(labels.index(user_loc[user][1]))
    return userIDs, y, labels


def load_region_labels(userids, user_text_seq_full, user_loc, labels):
    regions = {}
    regions['Connecticut'] = 0
    regions['Maine'] = 0
    regions['Massachusetts'] = 0
    regions['New Hampshire'] = 0
    regions['Rhode Island'] = 0
    regions['Vermont'] = 0
    regions['New Jersey'] = 0
    regions['New York'] = 0
    regions['Pennsylvania'] = 0
    regions['Indiana'] = 1
    regions['Illinois'] = 1
    regions['Michigan'] = 1
    regions['Ohio'] = 1
    regions['Wisconsin'] = 1
    regions['Iowa'] = 1
    regions['Kansas'] = 1
    regions['Minnesota'] = 1
    regions['Missouri'] = 1
    regions['Nebraska'] = 1
    regions['North Dakota'] = 1
    regions['South Dakota'] = 1
    regions['Delaware'] = 2
    regions['District of Columbia'] = 2
    regions['Florida'] = 2
    regions['Georgia'] = 2
    regions['Maryland'] = 2
    regions['North Carolina'] = 2
    regions['South Carolina'] = 2
    regions['Virginia'] = 2
    regions['West Virginia'] = 2
    regions['Alabama'] = 2
    regions['Kentucky'] = 2
    regions['Mississippi'] = 2
    regions['Tennessee'] = 2
    regions['Arkansas'] = 2
    regions['Louisiana'] = 2
    regions['Oklahoma'] = 2
    regions['Texas'] = 2
    regions['Arizona'] = 3
    regions['Colorado'] = 3
    regions['Idaho'] = 3
    regions['New Mexico'] = 3
    regions['Montana'] = 3
    regions['Utah'] = 3
    regions['Nevada'] = 3
    regions['Wyoming'] = 3
    regions['California'] = 3
    regions['Oregon'] = 3
    regions['Washington'] = 3

    X, y, userIDs = [], [], []
    for user in userids:
        if 'UKN' not in user_loc[user][0]:
            if 'District of Columbia' in user_loc[user][2]:
                if user_loc[user][0] == 'United States of America':
                    region = regions[user_loc[user][2]]
                    if not region in labels:
                        labels.append(region)
                    X.append(user_text_seq_full[user])
                    userIDs.append(user)
                    y.append(labels.index(region))
        if 'UKN' not in user_loc[user][1]:
            if user_loc[user][0] == 'United States of America':
                region = regions[user_loc[user][1]]
                if not region in labels:
                    labels.append(region)
                y.append(labels.index(region))
                X.append(user_text_seq_full[user])
                userIDs.append(user)
    return userIDs, y, labels


# Load "einstein_locations.csv"
def read_user_location(dataset):
    user_locations = {}
    with open(dataset, 'r') as f:
        i = 0
        for line in f:
            if i > 0:
                content = line.split(',')
                # user_location['user'] = ['country', 'state', 'county', 'city']
                user_locations[content[0]] = [content[1], content[2], content[3], content[4]]
            i += 1
    f.close()
    return user_locations


def convert_y_coord(y_train, y_dev, y_test):
    y_train = np.array(y_train).astype(np.float)
    y_dev = np.array(y_dev).astype(np.float)
    y_test = np.array(y_test).astype(np.float)
    return y_train, y_dev, y_test


def load_data(data_home, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    dtype = kwargs.get('dtype', 'float32')
    task = kwargs.get('task')
    dl = DataLoader(data_home=data_home, encoding=encoding)

    logging.info('loading dataset...')
    dl.load_data()

    Y_train, Y_dev, Y_test, labels = [], [], [], []
    if task == "regression":
        print("Using latitude and longitude")
        Y_train = np.array([[a[0], a[1]] for a in dl.df_train[['lat', 'lon']].values.tolist()], dtype=dtype)
        Y_dev = np.array([[a[0], a[1]] for a in dl.df_dev[['lat', 'lon']].values.tolist()], dtype=dtype)
        Y_test = np.array([[a[0], a[1]] for a in dl.df_test[['lat', 'lon']].values.tolist()], dtype=dtype)
    elif task == "classify_states":
        print("Using states")
        user_locations_file = "eisenstein_locations.csv"
        user_loc = read_user_location(user_locations_file)
        user_train, Y_train, labels = load_state_labels(list(dl.df_train.index), dl.df_train['text'].to_dict(),
                                                        user_loc, labels)
        user_dev, Y_dev, labels = load_state_labels(list(dl.df_dev.index), dl.df_dev['text'].to_dict(), user_loc,
                                                    labels)
        user_test, Y_test, labels = load_state_labels(list(dl.df_test.index), dl.df_test['text'].to_dict(), user_loc,
                                                      labels)
        dl.df_train = dl.df_train[dl.df_train.index.isin(user_train)]
        dl.df_dev = dl.df_dev[dl.df_dev.index.isin(user_dev)]
        dl.df_test = dl.df_test[dl.df_test.index.isin(user_test)]
    elif task == "classify_regions":
        print("Using regions")
        user_locations_file = "eisenstein_locations.csv"
        user_loc = read_user_location(user_locations_file)
        user_train, Y_train, labels = load_region_labels(list(dl.df_train.index), dl.df_train['text'].to_dict(),
                                                         user_loc, labels)
        user_dev, Y_dev, labels = load_region_labels(list(dl.df_dev.index), dl.df_dev['text'].to_dict(), user_loc,
                                                     labels)
        user_test, Y_test, labels = load_region_labels(list(dl.df_test.index), dl.df_test['text'].to_dict(), user_loc,
                                                       labels)
        dl.df_train = dl.df_train[dl.df_train.index.isin(user_train)]
        dl.df_dev = dl.df_dev[dl.df_dev.index.isin(user_dev)]
        dl.df_test = dl.df_test[dl.df_test.index.isin(user_test)]

    dl.tosequence()

    U_test = dl.df_test.index.tolist()
    U_dev = dl.df_dev.index.tolist()
    U_train = dl.df_train.index.tolist()
    X_train = dl.X_train.astype(dtype)
    X_dev = dl.X_dev.astype(dtype)
    X_test = dl.X_test.astype(dtype)
    dl.max_features = X_train.shape[1]
    Y_train, Y_dev, Y_test = convert_y_coord(Y_train, Y_dev, Y_test)
    data = (X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, labels)
    return data


class DataLoader:
    def __init__(self, data_home, encoding='utf-8', maxlen=None, max_features=None, char_level=False):
        self.data_home = data_home
        self.maxlen = maxlen
        self.max_features = max_features
        self.encoding = encoding
        self.char_level = char_level

    def load_data(self):
        logging.info('loading the dataset from %s' % self.data_home)
        train_file = os.path.join(self.data_home, 'user_info.train.gz')
        dev_file = os.path.join(self.data_home, 'user_info.dev.gz')
        test_file = os.path.join(self.data_home, 'user_info.test.gz')

        df_train = pd.read_csv(train_file, delimiter='\t', encoding=self.encoding, names=['user', 'lat', 'lon', 'text'],
                               quoting=csv.QUOTE_NONE, error_bad_lines=False)
        df_dev = pd.read_csv(dev_file, delimiter='\t', encoding=self.encoding, names=['user', 'lat', 'lon', 'text'],
                             quoting=csv.QUOTE_NONE, error_bad_lines=False)
        df_test = pd.read_csv(test_file, delimiter='\t', encoding=self.encoding, names=['user', 'lat', 'lon', 'text'],
                              quoting=csv.QUOTE_NONE, error_bad_lines=False)
        df_train.dropna(inplace=True)
        df_dev.dropna(inplace=True)
        df_test.dropna(inplace=True)

        df_train.drop_duplicates(['user'], inplace=True, keep='last')
        df_train.set_index(['user'], drop=True, append=False, inplace=True)
        df_train.sort_index(inplace=True)

        df_dev.drop_duplicates(['user'], inplace=True, keep='last')
        df_dev.set_index(['user'], drop=True, append=False, inplace=True)
        df_dev.sort_index(inplace=True)

        df_test.drop_duplicates(['user'], inplace=True, keep='last')
        df_test.set_index(['user'], drop=True, append=False, inplace=True)
        df_test.sort_index(inplace=True)

        self.df_train = df_train
        self.df_dev = df_dev
        self.df_test = df_test

    def tosequence(self):
        self.vectorizer = SequenceVectorizer(self.char_level, self.maxlen, self.max_features)
        logging.info(self.vectorizer)
        self.X_train = self.vectorizer.fit(self.df_train.text.values)
        self.X_dev = self.vectorizer.transform(self.df_dev.text.values)
        self.X_test = self.vectorizer.transform(self.df_test.text.values)
        logging.info("training    n_samples: %d, n_features: %d" % self.X_train.shape)
        logging.info("development n_samples: %d, n_features: %d" % self.X_dev.shape)
        logging.info("test        n_samples: %d, n_features: %d" % self.X_test.shape)


class SequenceVectorizer:
    def __init__(self, char_level=False, maxlen=None, max_features=None):
        self.max_features = max_features
        self.char_level = char_level
        self.tokenizer = keras.preprocessing.text.Tokenizer(filters=" ", char_level=self.char_level,
                                                            num_words=self.max_features)
        self.maxlen = maxlen
        self.vocabulary_ = None

    def fit(self, X):
        self.tokenizer.fit_on_texts(X)
        X_seq = self.tokenizer.texts_to_sequences(X)
        # pad 4987 to 5000
        X_seq = sequence.pad_sequences(X_seq, maxlen=5000, padding='post')

        self.maxlen = X_seq.shape[1]
        self.vocabulary_ = self.tokenizer.word_index
        self.vocabulary_ = sorted(self.tokenizer.word_counts, key=self.tokenizer.word_counts.get, reverse=True)
        if (self.max_features):
            self.vocabulary_ = self.vocabulary_[:self.max_features]
        logging.info('SequenceVectorizer maxlen:{}, #words:{}, most common words:{}'.
                     format(self.maxlen, len(self.vocabulary_), 0))
        return X_seq

    def transform(self, X):
        logging.info('Fitting SequenceVectorizer in {} texts'.format(len(X)))
        X_seq = self.tokenizer.texts_to_sequences(X)
        X_seq = sequence.pad_sequences(X_seq, maxlen=self.maxlen, padding='post')
        return X_seq


if __name__ == '__main__':
    data_loader = DataLoader(data_home='./data/', encoding='latin1')
    data_loader.load_data()
    data_loader.tosequence()
    dtype = 'float32'

    U_test = data_loader.df_test.index.tolist()
    U_dev = data_loader.df_dev.index.tolist()
    U_train = data_loader.df_train.index.tolist()
    X_train = data_loader.X_train.astype(dtype)
    X_dev = data_loader.X_dev.astype(dtype)
    X_test = data_loader.X_test.astype(dtype)

    Y_train = np.array([[a[0], a[1]] for a in data_loader.df_train[['lat', 'lon']].values.tolist()], dtype=dtype)
    Y_dev = np.array([[a[0], a[1]] for a in data_loader.df_dev[['lat', 'lon']].values.tolist()], dtype=dtype)
    Y_test = np.array([[a[0], a[1]] for a in data_loader.df_test[['lat', 'lon']].values.tolist()], dtype=dtype)

    data = (X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test)
