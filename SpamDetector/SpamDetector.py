# Impact of punctuation and stopwords words on email spam detector(ai)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import string

import nltk # library for delete stopwords
from nltk.corpus import stopwords


def process_text(text): # deletes puncuations

  nopunc = [char for char in text if char not in string.punctuation]
  nopunc = ''.join(nopunc)

  clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

  return clean_words

df = pd.read_csv("lingSpam.csv") # dataset from kaggle link -> https://www.kaggle.com/datasets/nitishabharathi/email-spam-dataset

x_train, x_test, y_train, y_test = train_test_split(df.Body, df.Label,test_size= 0.20) # spliting 20%

cv = CountVectorizer()
x_train_count = cv.fit_transform(x_train.values)

model = MultinomialNB()

model.fit(x_train_count, y_train)

x_test_count = cv.transform(x_test)


#------------------ process text and teach again (check from 13 line)-----------------------#

print(df['Body'].head(1).apply(process_text)) # without punctuations and stopwords

message_bow = CountVectorizer(analyzer=process_text).fit_transform(df['Body'])

x_train2, x_test2, y_train2, y_test2 = train_test_split(message_bow, df['Label'],test_size = 0.20, random_state = 0)  # spliting 20%

#print(message_bow.shape)

classifier = MultinomialNB().fit(x_train2, y_train2)

from sklearn.metrics import accuracy_score
pred = classifier.predict(x_train2)

print("Model score learned without puncuations and stopwords accuracy score - > ", accuracy_score(y_train2, pred))
print("Model score learned with puncuations and stowords accuracy score - > ",model.score(x_test_count, y_test))










