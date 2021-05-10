import streamlit as st
from sklearn.preprocessing import LabelEncoder
import numpy as np 
import os
import pandas as pd
import sklearn

import matplotlib.pyplot as plt
import time
from sklearn import datasets
from sklearn.model_selection import train_test_split
from streamlit_echarts import st_echarts

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


from sklearn.metrics import accuracy_score






st.title('Anomaly-Based IDS Workbench System')

st.write("""
#
""")

with st.form(key='my_form'):
	text_input = st.text_input(label='Enter some text')
	submit_button = st.form_submit_button(label='Submit')

ratio=0.33

#Defining visualization plot here
def plot_column( col ):
    if df[col].dtype == 'object':
        encoder=LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
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


#@st.cache(suppress_st_warning=True)
def write_statistics(statistics, visualizaitons):
    if 'Dataset Shape' in statistics:
        st.write('Shape of Dataset:', X.shape)
    if 'Number of Classes' in statistics:
        st.write('Number of Classes:', len(np.unique(y)))
    if 'Dataset Head' in statistics:
        st.write('Dataset Head:', df.head(5))
    if 'Describe Features' in statistics:
        st.write('Feature Description:', df.describe())
    if 'View Packet Types' in statistics:
        st.write('Packet Types:', np.unique(y))

    if visualizaitons:
        write_visualizations(visualizaitons)

#@st.cache(suppress_st_warning=True)
def write_visualizations(visualizaitons):
    for column in visualizaitons:
        plot_column(col=column)


def populate_statistics(df):
    st.sidebar.header('Data Exploration')
    #Print statistics and visualizations sidebar items
    statistics=False
    visualizaitons=False
    statistics = st.sidebar.multiselect(
        'Select Desired Statistics',
        ('Dataset Head', 'Dataset Shape', 'Number of classes', 'Describe Features', 'View Packet Types', 'Plot Feature Visualizations')
    )
    if 'Plot Feature Visualizations' in statistics:
        visualizaitons = st.sidebar.multiselect(
            'Select Desired Visualizations',
            (df.columns)
        )
    
    if statistics:
        write_statistics(statistics, visualizaitons)


def populate_preprocessors(df, X, y):
    st.sidebar.header('Preprocessing')

    #Drop null values here:
    drop_nulls_btn = st.sidebar.checkbox('Drop Rows with Null Values')
    if drop_nulls_btn:
        df = df.dropna(axis=0)
    
    #Print preprocessing sidebar items
    scaling_btn = st.sidebar.checkbox('Apply Logarithmic Scaling')
    if scaling_btn:
        #Applying logarithmic scaling here
        sc = MinMaxScaler()
        X = sc.fit_transform(X)

    ratio_btn = st.sidebar.selectbox('Select Custom/Default Test-Train Ratio',('Default', 'Custom'))
    global ratio
    if ratio_btn == 'Default':
        ratio = 0.33
    if ratio_btn == 'Custom':
        ratio = st.sidebar.number_input('ratio', min_value=0.01, max_value=0.99)
    
    
    
    return df, X, y


uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    #Reading uploading dataset csv file here
    df = pd.read_csv(uploaded_file)

    #Check if dataframe has NaN/Infinite values
    #st.write(df.isna().any())
    #st.write(df.isin([np.inf, -np.inf]).any())
    


    #Replace NaN/Infinite values with 0
    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], 0)


    #Splitting x & y dataframes here
    y = df.iloc[:,[-1]]
    X = df.iloc[: , :-1]



    #Label encoding categorical features here
    encoder = LabelEncoder()
    num_cols = X._get_numeric_data().columns
    cate_cols = list(set(X.columns)-set(num_cols))
    for item in cate_cols:
        X[item] = encoder.fit_transform(X[item])


    #Displaying dataset statistics and visualizaitons here
    populate_statistics(df)
    df, X, y = populate_preprocessors(df, X, y)



#stats = add_parameter_ui(classifier_name)


#Defining dynamic parameter generation here
def add_parameters(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
        kernel = st.sidebar.selectbox('Select kernel',('rbf', 'linear', 'poly', 'sigmoid', 'precomputed'))
        params['kernel'] = kernel
        degree = st.sidebar.selectbox('Select Custom/Default degree',('Default', 'Custom'))
        if degree == 'Default':
            params['degree'] = 3
        if degree == 'Custom':
            degree = st.sidebar.number_input('degree', min_value=1, max_value=99999999)
            params['degree'] = degree
        probability = st.sidebar.checkbox('Enable probability estimates (uses 5-fold cross-validation)')
        if probability:
            params['probability'] = True
        else:
            params['probability'] = False

    if clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
        algorithm = st.sidebar.selectbox('Select algorithm',('auto', 'ball_tree', 'kd_tree', 'brute'))
        params['algorithm'] = algorithm
        p = st.sidebar.selectbox('Select Custom/Default Power (p)',('Default', 'Custom'))
        if p == 'Default':
            params['p'] = 2
        if p == 'Custom':
            p = st.sidebar.number_input('p', min_value=1, max_value=99999999)
            params['p'] = p
        n_jobs = st.sidebar.selectbox('Select Custom/Default n_jobs (Parallel Jobs)',('Default', 'Custom'))
        if n_jobs == 'Default':
            params['n_jobs'] = None
        if n_jobs == 'Custom':
            n_jobs = st.sidebar.number_input('n_jobs', min_value=-1, max_value=99999999)
            params['n_jobs'] = n_jobs

    if clf_name == 'Random Forest':
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
        min_samples_split = st.sidebar.selectbox('Select Custom/Default min_samples_split',('Default', 'Custom'))
        if min_samples_split == 'Default':
            params['min_samples_split'] = 2
        if min_samples_split == 'Custom':
            min_samples_split = st.sidebar.number_input('min_samples_split', min_value=1, max_value=99999999)
            params['min_samples_split'] = min_samples_split
        n_jobs = st.sidebar.selectbox('Select Custom/Default n_jobs (Parallel Jobs)',('Default', 'Custom'))
        if n_jobs == 'Default':
            params['n_jobs'] = None
        if n_jobs == 'Custom':
            n_jobs = st.sidebar.number_input('n_jobs', min_value=-1, max_value=99999999)
            params['n_jobs'] = n_jobs
        criterion = st.sidebar.selectbox('Select criterion',('gini', 'entropy'))
        params['criterion'] = criterion

    if clf_name == 'Decision Tree':
        criterion = st.sidebar.selectbox('Select criterion',('gini', 'entropy'))
        params['criterion'] = criterion
        splitter = st.sidebar.selectbox('Select splitter',('best', 'random'))
        params['splitter'] = splitter
        depth_type = st.sidebar.selectbox('Select Custom/Default Tree Depth',('Default', 'Custom'))
        if depth_type == 'Default':
            params['max_depth'] = None
        if depth_type == 'Custom':
            max_depth = st.sidebar.slider('max_depth', 2, 15)
            params['max_depth'] = max_depth
        min_samples_split = st.sidebar.selectbox('Select Custom/Default min_samples_split',('Default', 'Custom'))
        if min_samples_split == 'Default':
            params['min_samples_split'] = 2
        if min_samples_split == 'Custom':
            min_samples_split = st.sidebar.number_input('min_samples_split', min_value=-1, max_value=99999999)
            params['min_samples_split'] = min_samples_split
        min_samples_leaf = st.sidebar.selectbox('Select Custom/Default min_samples_leaf',('Default', 'Custom'))
        if min_samples_leaf == 'Default':
            params['min_samples_leaf'] = 1
        if min_samples_leaf == 'Custom':
            min_samples_leaf = st.sidebar.number_input('min_samples_leaf', min_value=1, max_value=99999999)
            params['min_samples_leaf'] = min_samples_leaf

    if clf_name == 'Logistic Regression':
        max_iter = st.sidebar.selectbox('Select Custom/Default Iterations Number',('Default', 'Custom'))
        if max_iter == 'Default':
            params['max_iter'] = 100
        if max_iter == 'Custom':
            max_iter = st.sidebar.number_input('max_iter', min_value=1, max_value=999999999999999)
            params['max_iter'] = max_iter
        solver = st.sidebar.selectbox('Select Solver',('lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'))
        params['solver'] = solver
        penalty = st.sidebar.selectbox('Select penalty',('l2', 'l1', 'elasticnet', 'none'))
        params['penalty'] = penalty
        dual = st.sidebar.checkbox('Enable Dual formulation')
        if dual:
            params['dual'] = True
        else:
            params['dual'] = False
        n_jobs = st.sidebar.selectbox('Select Custom/Default n_jobs (Parallel Jobs)',('Default', 'Custom'))
        if n_jobs == 'Default':
            params['n_jobs'] = None
        if n_jobs == 'Custom':
            n_jobs = st.sidebar.number_input('n_jobs', min_value=-1, max_value=99999999)
            params['n_jobs'] = n_jobs
        

    if clf_name == 'Gradient Boosting Classifier':
        n_estimators = st.sidebar.selectbox('Select Custom/Default No. of Estimators',('Default', 'Custom'))
        if n_estimators == 'Default':
            params['n_estimators'] = 100
        if n_estimators == 'Custom':
            n_estimators = st.sidebar.number_input('n_estimators', min_value=1, max_value=999999999999999)
            params['n_estimators'] = n_estimators
        loss = st.sidebar.selectbox('Loss Function',('deviance', 'exponential'))
        params['loss'] = loss
        max_depth = st.sidebar.selectbox('Select Custom/Default max_depth',('Default', 'Custom'))
        if max_depth == 'Default':
            params['max_depth'] = 3
        if max_depth == 'Custom':
            max_depth = st.sidebar.number_input('max_depth', min_value=1, max_value=999999999999)
            params['max_depth'] = max_depth
    
    if clf_name == 'Artificial Neural Networks':
        st.write("hello")

    return params



#Populating classification sidebar here
classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('Naive Bayes', 'KNN', 'SVM', 'Random Forest', 'Decision Tree', 'Logistic Regression', 'Gradient Boosting Classifier', 'Artificial Neural Networks')
)
params = add_parameters(classifier_name)




#Instantiating the classifier selected in the sidebar
def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C=params['C'], kernel=params['kernel'], degree=params['degree'])
    if clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'], algorithm=params['algorithm'], p=params['p'], n_jobs=params['n_jobs'])
    if clf_name == 'Naive Bayes':
        clf = GaussianNB()
    if clf_name == 'Random Forest':
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
            max_depth=params['max_depth'], random_state=1234, min_samples_split=params['min_samples_split'], n_jobs=params['n_jobs'], criterion=params['criterion'])
    if clf_name == 'Decision Tree':
        clf = DecisionTreeClassifier(criterion=params['criterion'], splitter=params['splitter'], max_depth = params['max_depth'], min_samples_split=params['min_samples_split'], min_samples_leaf=params['min_samples_leaf'])
    if clf_name == 'Logistic Regression':
        clf = LogisticRegression(max_iter=params['max_iter'], solver=params['solver'], penalty=params['penalty'], n_jobs=params['n_jobs'])
    if clf_name == 'Gradient Boosting Classifier':
        clf = GradientBoostingClassifier(n_estimators=params['n_estimators'], loss=params['loss'], max_depth=params['max_depth'])
    if clf_name == 'Artificial Neural Networks':
        from keras.wrappers.scikit_learn import KerasClassifier
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import LSTM
        def lstm():
            model = Sequential()
            model.add(Dense(1,input_dim=41,activation = 'relu',kernel_initializer='random_uniform'))
            model.add(Dense(1,activation='sigmoid',kernel_initializer='random_uniform'))

            #model.add(LSTM(1, input_shape=(50, 41)))
            model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
            return model
        clf = KerasClassifier(build_fn=lstm,epochs=1,batch_size=64)
    return clf

clf = get_classifier(classifier_name, params)



#Defining prediction & accuracy metrics function here
def get_prediction():
        if st.button('Classify'):
            st.write('Train to Test Ratio = ', ratio)
            
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.write(f'Classifier = {classifier_name}')
            st.write('Accuracy =', acc)
            metrics = sklearn.metrics.classification_report(y_test, y_pred)
            st.text(metrics)
            
        else: 
            st.write('Click the button to classify')

get_prediction()

