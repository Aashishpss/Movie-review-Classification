# -*- coding: utf-8 -*-
"""02-Text-Classification-Assessment .ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Bfb7b1Te6Lz2TCDapBkstMZEjnTtpewX

# Movie Review Assessment

### Goal: Given a set of text movie reviews we have to predict whether it is positive or negative

For more information on this dataset visit http://ai.stanford.edu/~amaas/data/sentiment/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('moviereviews.csv')

df[df['label']=='pos'].iloc[0]['review']

df.head()

"""**Checking  if there are any missing values in the dataframe.**"""

df.isnull().sum()

"""**Removing any reviews that are NaN**"""

df.dropna(inplace=True)

df.isnull().sum()

"""**Checking  to see if any reviews are blank strings and not just NaN. Note: This means a review text could just be: "" or "  " or some other larger blank string.**"""

df[df['review'].str.isspace()].count()

k=df[df['review'].str.isspace()]
p=k.index
df.drop(p,inplace=True)

df.info()

"""**Confirming the value counts per label:**"""

df['label'].value_counts()

"""### Training and Data

**Splitting the data into features and a label (X and y) and then preforming a train/test split.`**
"""

X,y=df['review'],df['label']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)

"""### Training a Model

**Creating a PipeLine that will both create a TF-IDF Vector out of the raw text data and fit a linear svc  and then fitting that pipeline on the training data.**
"""

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

pipe=Pipeline([('tfidf',TfidfVectorizer()),('svc',LinearSVC())])
pipe.fit(X_train,y_train)

"""**Creating e a classification report and plotting a confusion matrix based on the results of the PipeLine.**"""

svc_pred=pipe.predict(X_test)

from sklearn.metrics import classification_report,ConfusionMatrixDisplay

print(classification_report(y_test,svc_pred))

ConfusionMatrixDisplay.from_predictions(y_test,svc_pred)

"""### Training the pipeline on the whole dataset"""

pipe.fit(X,y)

"""### Checking the performance by giving some review by ourself"""

review1="The action scenes of this movie are great which makes it popular among the youths and it results in the higher rating of this movie"

print(pipe.predict([review1]))

review2=" This is a very sarcastic movie"

print(pipe.predict([review2]))