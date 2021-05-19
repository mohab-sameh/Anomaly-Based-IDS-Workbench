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


import plotly.express as px




st.title('Anomaly-Based IDS Workbench System')

st.write("""
#
""")

program = """ 

st.write("hahah omg")

"""
#exec(program)

class Dataframe:
    ratio=0.3


#Defining visualization plot here
def plot_column( col ):
    if Dataframe.df[col].dtype == 'object':
        encoder=LabelEncoder()
        Dataframe.df[col] = encoder.fit_transform(Dataframe.df[col])
    fig, ax = plt.subplots()
    Dataframe.df.hist(
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
        st.write('Shape of Dataset:', Dataframe.df.shape)
    if 'Number of Classes' in statistics:
        st.write('Number of Classes:', len(np.unique(Dataframe.y)))
    if 'Dataset Head' in statistics:
        st.write('Dataset Head:', Dataframe.df.head(5))
    if 'Describe Features' in statistics:
        st.write('Feature Description:', Dataframe.df.describe())
    if 'View Packet Types' in statistics:
        st.write('Packet Types:', np.unique(Dataframe.y))
    if 'Scatter Plots' in statistics:
        st.subheader("Scatter Plot:")
        plot_dim = st.selectbox("Select plot dimensionality", ('2D Plot', '3D Plot'))
        with st.form('Scatter Plot Form'):
            if(Dataframe.df.shape[0] < 200):
                max_rows = len(Dataframe.df.index)
            else:
                max_rows = 200
            num_samples = st.slider(label="Select number of random samples", min_value=1, max_value=max_rows)
            sample_df = Dataframe.df.sample(num_samples)

            if plot_dim == '2D Plot':
                feature_x = st.selectbox('Select X-Axis Feature', (Dataframe.df.columns))
                feature_y = st.selectbox('Select Y-Axis Feature', (Dataframe.df.columns))
                if(feature_x and feature_y):
                    fig = px.scatter(sample_df, x= feature_x, y = feature_y, color=sample_df.columns[len(sample_df.columns)-1])
                    st.plotly_chart(fig)
            if plot_dim == '3D Plot':
                feature_x = st.selectbox('Select X-Axis Feature', (Dataframe.df.columns))
                feature_y = st.selectbox('Select Y-Axis Feature', (Dataframe.df.columns))
                feature_z = st.selectbox('Select Z-Axis Feature', (Dataframe.df.columns))
                if(feature_x and feature_y):
                    fig = px.scatter_3d(sample_df, feature_x, feature_y, feature_z, color = sample_df.columns[len(sample_df.columns)-1])
                    st.plotly_chart(fig)

            scatter_submit = st.form_submit_button('Apply Selected Options')

        

    if visualizaitons:
        write_visualizations(visualizaitons)

#@st.cache(suppress_st_warning=True)
def write_visualizations(visualizaitons):
    for column in visualizaitons:
        plot_column(col=column)


def populate_statistics():
    st.sidebar.header('Data Exploration')
    #Print statistics and visualizations sidebar items
    statistics=False
    visualizaitons=False
    with st.sidebar.form('Statistics Form'):
        statistics = st.multiselect(
            'Select Desired Statistics',
            ('Dataset Head', 'Dataset Shape', 'Number of Classes', 'Describe Features', 'View Packet Types', 'Scatter Plots', 'Plot Feature Visualizations')
        )
        statistics_submit = st.form_submit_button('Show Selected Options')

    if 'Plot Feature Visualizations' in statistics:
        with st.sidebar.form('Visualizations Form'):
            visualizaitons = st.multiselect(
                'Select Desired Visualizations',
                (Dataframe.df.columns)
            )
            visualizations_submit = st.form_submit_button('Show Selected Options')

    if statistics:
        write_statistics(statistics, visualizaitons)


def populate_preprocessors():
    st.sidebar.header('Preprocessing')

    #Drop null values here:
    drop_nulls_btn = st.sidebar.checkbox('Drop Rows with Null Values')
    if drop_nulls_btn:
        Dataframe.df = Dataframe.df.dropna(axis=0)
    
    #Print preprocessing sidebar items
    scaling_btn = st.sidebar.checkbox('Apply Logarithmic Scaling')
    if scaling_btn:
        #Applying logarithmic scaling here
        sc = MinMaxScaler()
        Dataframe.X[Dataframe.X.columns] = sc.fit_transform(Dataframe.X)

    ratio_btn = st.sidebar.selectbox('Select Custom/Default Test-Train Ratio',('Default', 'Custom'))
    if ratio_btn == 'Default':
        Dataframe.ratio = 0.33
    if ratio_btn == 'Custom':
        Dataframe.ratio = st.sidebar.number_input('ratio', min_value=0.01, max_value=0.99)
    
    


#Defining dynamic parameter generation here
def add_parameters(clf_name):
    params = dict()
    if clf_name == 'SVM':
        with st.sidebar.form('SVM Form'):
            C = st.slider('C', 0.01, 10.0)
            params['C'] = C
            kernel = st.selectbox('Select kernel',('rbf', 'linear', 'poly', 'sigmoid', 'precomputed'))
            params['kernel'] = kernel
            degree = st.selectbox('Select Custom/Default degree',('Default', 'Custom'))
            if degree == 'Default':
                params['degree'] = 3
            if degree == 'Custom':
                degree = st.number_input('degree', min_value=1, max_value=99999999)
                params['degree'] = degree
            probability = st.checkbox('Enable probability estimates (uses 5-fold cross-validation)')
            if probability:
                params['probability'] = True
            else:
                params['probability'] = False

            svm_submit = st.form_submit_button('Apply Selected Options')

    if clf_name == 'KNN':
        with st.sidebar.form('KNN Form'):
            K = st.slider('K', 1, 15)
            params['K'] = K
            algorithm = st.selectbox('Select algorithm',('auto', 'ball_tree', 'kd_tree', 'brute'))
            params['algorithm'] = algorithm
            p = st.selectbox('Select Custom/Default Power (p)',('Default', 'Custom'))
            if p == 'Default':
                params['p'] = 2
            if p == 'Custom':
                p = st.number_input('p', min_value=1, max_value=99999999)
                params['p'] = p
            n_jobs = st.selectbox('Select Custom/Default n_jobs (Parallel Jobs)',('Default', 'Custom'))
            if n_jobs == 'Default':
                params['n_jobs'] = None
            if n_jobs == 'Custom':
                n_jobs = st.number_input('n_jobs', min_value=-1, max_value=99999999)
                params['n_jobs'] = n_jobs

            knn_submit = st.form_submit_button('Apply Selected Options')
        

    if clf_name == 'Random Forest':
        with st.sidebar.form('RF Form'):
            max_depth = st.slider('max_depth', 2, 15)
            params['max_depth'] = max_depth
            n_estimators = st.slider('n_estimators', 1, 100)
            params['n_estimators'] = n_estimators
            min_samples_split = st.selectbox('Select Custom/Default min_samples_split',('Default', 'Custom'))
            if min_samples_split == 'Default':
                params['min_samples_split'] = 2
            if min_samples_split == 'Custom':
                min_samples_split = st.number_input('min_samples_split', min_value=1, max_value=99999999)
                params['min_samples_split'] = min_samples_split
            n_jobs = st.selectbox('Select Custom/Default n_jobs (Parallel Jobs)',('Default', 'Custom'))
            if n_jobs == 'Default':
                params['n_jobs'] = None
            if n_jobs == 'Custom':
                n_jobs = st.number_input('n_jobs', min_value=-1, max_value=99999999)
                params['n_jobs'] = n_jobs
            criterion = st.selectbox('Select criterion',('gini', 'entropy'))
            params['criterion'] = criterion
            
            rf_submit = st.form_submit_button('Apply Selected Options')

    if clf_name == 'Decision Tree':
        with st.sidebar.form('DT Form'):
            criterion = st.selectbox('Select criterion',('gini', 'entropy'))
            params['criterion'] = criterion
            splitter = st.selectbox('Select splitter',('best', 'random'))
            params['splitter'] = splitter
            depth_type = st.selectbox('Select Custom/Default Tree Depth',('Default', 'Custom'))
            if depth_type == 'Default':
                params['max_depth'] = None
            if depth_type == 'Custom':
                max_depth = st.slider('max_depth', 2, 15)
                params['max_depth'] = max_depth
            min_samples_split = st.selectbox('Select Custom/Default min_samples_split',('Default', 'Custom'))
            if min_samples_split == 'Default':
                params['min_samples_split'] = 2
            if min_samples_split == 'Custom':
                min_samples_split = st.number_input('min_samples_split', min_value=-1, max_value=99999999)
                params['min_samples_split'] = min_samples_split
            min_samples_leaf = st.selectbox('Select Custom/Default min_samples_leaf',('Default', 'Custom'))
            if min_samples_leaf == 'Default':
                params['min_samples_leaf'] = 1
            if min_samples_leaf == 'Custom':
                min_samples_leaf = st.number_input('min_samples_leaf', min_value=1, max_value=99999999)
                params['min_samples_leaf'] = min_samples_leaf
            
            dt_submit = st.form_submit_button('Apply Selected Options')
        
        

    if clf_name == 'Logistic Regression':
        with st.sidebar.form('LR Form'):
            max_iter = st.selectbox('Select Custom/Default Iterations Number',('Default', 'Custom'))
            if max_iter == 'Default':
                params['max_iter'] = 100
            if max_iter == 'Custom':
                max_iter = st.number_input('max_iter', min_value=1, max_value=999999999999999)
                params['max_iter'] = max_iter
            solver = st.selectbox('Select Solver',('lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'))
            params['solver'] = solver
            penalty = st.selectbox('Select penalty',('l2', 'l1', 'elasticnet', 'none'))
            params['penalty'] = penalty
            dual = st.checkbox('Enable Dual formulation')
            if dual:
                params['dual'] = True
            else:
                params['dual'] = False
            n_jobs = st.selectbox('Select Custom/Default n_jobs (Parallel Jobs)',('Default', 'Custom'))
            if n_jobs == 'Default':
                params['n_jobs'] = None
            if n_jobs == 'Custom':
                n_jobs = st.number_input('n_jobs', min_value=-1, max_value=99999999)
                params['n_jobs'] = n_jobs

            LR_submit = st.form_submit_button('Apply Selected Options')

        

    if clf_name == 'Gradient Boosting Classifier':
        with st.sidebar.form('GBC Form'):
            n_estimators = st.selectbox('Select Custom/Default No. of Estimators',('Default', 'Custom'))
            if n_estimators == 'Default':
                params['n_estimators'] = 100
            if n_estimators == 'Custom':
                n_estimators = st.number_input('n_estimators', min_value=1, max_value=999999999999999)
                params['n_estimators'] = n_estimators
            loss = st.selectbox('Loss Function',('deviance', 'exponential'))
            params['loss'] = loss
            max_depth = st.selectbox('Select Custom/Default max_depth',('Default', 'Custom'))
            if max_depth == 'Default':
                params['max_depth'] = 3
            if max_depth == 'Custom':
                max_depth = st.number_input('max_depth', min_value=1, max_value=999999999999)
                params['max_depth'] = max_depth

            GBC_submit = st.form_submit_button('Apply Selected Options')

        
    
    if clf_name == 'Artificial Neural Networks':
        st.write(" ")

    return params


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
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
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
            #model.add(Dense(41,input_dim=41,activation = 'relu',kernel_initializer='random_uniform'))
            #model.add(Dense(41,activation='sigmoid',kernel_initializer='random_uniform'))
            model.add(LSTM((1),batch_input_shape=(None, 1, Dataframe.X_train.shape[1]), return_sequences=False))
            #model.add(LSTM(units = 23, return_sequences = True, input_shape= (41, 1, 1)))
            #model.add(LSTM(1, input_shape=(50, 41)))
            
            model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
            return model
        clf = KerasClassifier(build_fn=lstm,epochs=10,batch_size=64)
    return clf



#Defining prediction & accuracy metrics function here
def get_prediction():
        if st.button('Classify'):
            st.write('Train to Test Ratio = ', Dataframe.ratio)
            
            #Splitting training and testing dataframes here
            Dataframe.X_train, Dataframe.X_test, Dataframe.y_train, Dataframe.y_test = train_test_split(Dataframe.X, Dataframe.y, test_size=0.2, random_state=1234)
            clf = get_classifier(classifier_name, params)

            #Reshape dataframes for ANN models
            if(classifier_name == 'Artificial Neural Networks'):
                Dataframe.X_train = np.reshape(np.ravel(Dataframe.X_train), (Dataframe.X_train.shape[0], 1, Dataframe.X_train.shape[1]))
                Dataframe.X_test = np.reshape(np.ravel(Dataframe.X_test), (Dataframe.X_test.shape[0], 1, Dataframe.X_test.shape[1]))
            
            #Start classifier fitting and evaluation
            clf.fit(Dataframe.X_train, Dataframe.y_train)
            Dataframe.y_pred = clf.predict(Dataframe.X_test)
            acc = accuracy_score(Dataframe.y_test, Dataframe.y_pred)
            st.write(f'Classifier = {classifier_name}')
            st.write('Accuracy =', acc)
            metrics = sklearn.metrics.classification_report(Dataframe.y_test, Dataframe.y_pred)
            st.text(metrics)
            
        else: 
            st.write('Click the button to classify')


def get_live_packet_prediction():
        if st.button('Classify'):
            st.write('Train to Test Ratio = ', Dataframe.ratio)
            
            #Splitting training and testing dataframes here
            Dataframe.X_train, Dataframe.X_test, Dataframe.y_train, Dataframe.y_test = train_test_split(Dataframe.X, Dataframe.y, test_size=0.2, random_state=1234)
            clf = get_classifier(classifier_name, params)

            #Reshape dataframes for ANN models
            if(classifier_name == 'Artificial Neural Networks'):
                Dataframe.X_train = np.reshape(np.ravel(Dataframe.X_train), (Dataframe.X_train.shape[0], 1, Dataframe.X_train.shape[1]))
                Dataframe.X_test = np.reshape(np.ravel(Dataframe.X_test), (Dataframe.X_test.shape[0], 1, Dataframe.X_test.shape[1]))


            #Import live packet capture data
            path2 = "custom-train-data.csv"
            Dataframe.X2_test = pd.read_csv(path2)
            #Keep only common columns between training data and live packet capture dataframes
            common_cols = [col for col in set(Dataframe.X_test.columns).intersection(Dataframe.X2_test.columns)]
            Dataframe.X_test = Dataframe.X_test[common_cols]
            Dataframe.X_train = Dataframe.X_train[common_cols]
            #Label encode the imported live packet capture dataframe
            num_cols2 = Dataframe.X2_test._get_numeric_data().columns
            cate_cols2 = list(set(Dataframe.X2_test.columns)-set(num_cols2))
            for item in cate_cols2:
                Dataframe.X2_test[item] = encoder.fit_transform(Dataframe.X2_test[item])
            
            
            #Start classifier fitting and evaluation
            clf.fit(Dataframe.X_train, Dataframe.y_train)
            Dataframe.y_pred = clf.predict(Dataframe.X_test)
            acc = accuracy_score(Dataframe.y_test, Dataframe.y_pred)
            st.write(f'Classifier = {classifier_name}')
            st.write('Accuracy =', acc)
            metrics = sklearn.metrics.classification_report(Dataframe.y_test, Dataframe.y_pred)
            st.text(metrics)
            

            #Predict and show attacks in live packet capture dataframe
            Dataframe.y2_pred = clf.predict(Dataframe.X2_test)
            st.write(Dataframe.y2_pred)
            st.bar_chart(Dataframe.y2_pred)
            unique_elements, counts_elements = np.unique(Dataframe.y2_pred, return_counts=True)
            st.write("Frequency of unique values of the said array:")
            st.text(np.asarray((unique_elements, counts_elements)))
            
        else: 
            st.write('Click the button to classify')

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    #Reading uploading dataset csv file here
    Dataframe.df = pd.read_csv(uploaded_file)

    #Replace NaN/Infinite values with 0
    Dataframe.df = Dataframe.df.fillna(0)
    Dataframe.df = Dataframe.df.replace([np.inf, -np.inf], 0)

    #Splitting x & y dataframes here
    Dataframe.y = Dataframe.df.iloc[:,[-1]]
    Dataframe.X = Dataframe.df.iloc[: , :-1]

    #Label encoding categorical features here
    encoder = LabelEncoder()
    num_cols = Dataframe.X._get_numeric_data().columns
    cate_cols = list(set(Dataframe.X.columns)-set(num_cols))
    for item in cate_cols:
        Dataframe.X[item] = encoder.fit_transform(Dataframe.X[item])


    #Displaying dataset statistics and visualizaitons here
    populate_statistics()
    populate_preprocessors()


#Populating classification sidebar here
classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('Naive Bayes', 'KNN', 'SVM', 'Random Forest', 'Decision Tree', 'Logistic Regression', 'Gradient Boosting Classifier', 'Artificial Neural Networks')
)
params = add_parameters(classifier_name)

get_prediction()

