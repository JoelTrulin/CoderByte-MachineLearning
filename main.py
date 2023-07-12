import nltk

import gui
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from others.popup import popup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras import metrics
from keras.models import load_model
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_validate
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def full_analysis(run):
    def pre_process():
        raw = pd.read_csv('dataset/articles.csv', encoding='latin-1')
        data = raw.Full_Article
        lb, type = raw.Article_Type.factorize()
        run = False
        if run:
            final_rev = []
            "stop word and stemming is used for pre_processing"
            try:
                stop_words = set(stopwords.words('english'))
                stemmer = PorterStemmer()
            except LookupError:
                nltk.download('stopwords')
                #from nltk.corpus import stopwords
                #from nltk.stem import PorterStemmer
                stop_words = set(stopwords.words('english'))
                stemmer = PorterStemmer()

            for idx, words in enumerate(data):
                print(idx)
                # stop_words = set(stopwords.words('english'))
                rev_pr = [w for w in words.split() if not w in stop_words]
                # stemmer = PorterStemmer()
                stemmed_rev = [stemmer.stem(word) for word in rev_pr]
                long_words_rev, long_words_revt = [], []
                for i in stemmed_rev:
                    if len(i) >= 3:  # removing short word
                        long_words_rev.append(i)
                final_rev.append((" ".join(long_words_rev)).strip())
            # np.save('pre_data', final_rev)
        pre_data = np.load('pre_evaluated/pre_data.npy', allow_pickle=True)
        return data, pre_data, lb

    def vectorization_(data):
        run=False
        " Sentence bert is used "
        if run:
            model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
            embeddings = model.encode(data)
            # np.save('vect_data', embeddings)
        vect_data = np.load('pre_evaluated/vect_data.npy', allow_pickle=True)
        return vect_data

    def smote_(feat, label):
        run=False
        if run:
            smote = SMOTE(random_state=42)
            x_res, y_res = smote.fit_resample(feat, label)
            #np.save('x_res.npy', x_res)
            #np.save('y_res.npy', y_res)

        x_res = np.load('pre_evaluated/x_res.npy', allow_pickle=True)
        y_res = np.load('pre_evaluated/y_res.npy', allow_pickle=True)
        return x_res, y_res

    def train(data, lb):
        global pm
        if run:
            X_train, X_test, y_train, y_test = train_test_split(data, lb, test_size=0.2, random_state=42)
            lstm_X_train = X_train.reshape(-1, 32, 24)
            lstm_X_test = X_test.reshape(-1, 32, 24)

            model = Sequential()
            model.add(LSTM(64, input_shape=lstm_X_train[0].shape))

            model.add(Dense(len(np.unique(lb)), activation='sigmoid'))
            model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
            model.fit(lstm_X_train, y_train, epochs=40, batch_size=350, verbose=True, validation_data=(lstm_X_test, y_test))
            # model.save('lstm.h5')
            #model = load_model('lstm.h5')

            pred = np.argmax(model.predict(lstm_X_test), axis=-1)

            pm = np.array([accuracy_score(y_test, pred), precision_score(y_test, pred, average='weighted'),
                           recall_score(y_test, pred, average='weighted'),
                           f1_score(y_test, pred, average='weighted')])
            # np.save('pre_evaluated/pm.npy', pm)

        pm = np.load('pre_evaluated/pm.npy', allow_pickle=True)
        return pm

    data, pre_data, lb = pre_process()
    vect_data = vectorization_(pre_data)
    balanced_data, lb = smote_(vect_data, lb)
    pm = train(balanced_data, lb)
    pm_df = pd.DataFrame(pm, columns=['Performance metrics'], index=['Accuracy', 'Precision', 'Recall', 'F1'])


    print('performance metrics Results')
    print(pm_df.to_markdown())


popup(full_analysis, full_analysis, gui.guii)
