import streamlit as st 
from sklearn.preprocessing import LabelEncoder
import numpy as np 
import os
import pandas as pd

import matplotlib.pyplot as plt
import time
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score







#################################################################################################
class Predict:
    def display_output(self):
        st.image(self.img.to_thumb(500,500), caption='Uploaded Image')

    def get_prediction(self):

        if st.button('Classify'):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.write(f'Classifier = {classifier_name}')
            st.write(f'Accuracy =', acc)
        else: 
            st.write(f'Click the button to classify') 


#################################################################################################







st.title('Streamlit Example')

st.write("""
# Explore different classifier and datasets
Which one is the best?
""")


def plot_column( col ):
    fig, ax = plt.subplots()
    df.hist(
        bins=8,
        column= col ,
        grid=False,
        figsize=(8, 8),
        color="#86bf91",
        zorder=2,
        rwidth=0.9,
        ax=ax,
      )
    st.write(fig)

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params


uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
  encoder = LabelEncoder()
  df['service'] = encoder.fit_transform(df['service'])
  df['protocol_type'] = encoder.fit_transform(df['protocol_type'])
  df['flag'] = encoder.fit_transform(df['flag'])
  df['label'] = encoder.fit_transform(df['label'])
  st.write(df.head(5))
  X=df.iloc[:,0:39]
  y=df.iloc[:,[-1]]
  st.write('Shape of dataset:', X.shape)
  st.write('number of classes:', len(np.unique(y)))
  plot_column(col='service')



classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('KNN', 'SVM', 'Random Forrest ')
)
params = add_parameter_ui(classifier_name)
def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C=params['C'])
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
            max_depth=params['max_depth'], random_state=1234)
    return clf

clf = get_classifier(classifier_name, params)
predictor = Predict()