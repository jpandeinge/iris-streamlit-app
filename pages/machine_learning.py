import imp
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

import streamlit as st
from sklearn import metrics
from sklearn import neighbors

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


from sklearn import tree

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier






def app():

    st.markdown("## Machine Learning Models")
    st.markdown(" A list of machine learning models that we will use to predict the species of the iris flower.")

    dataset_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"

    @st.experimental_memo
    def get_iris_data() -> pd.DataFrame:
        return pd.read_csv(dataset_url)

    # df = px.data.iris() # to get the data from plotly
    df = get_iris_data() # to get the data from a url
    
    X = df.drop(columns=['species'])
    y = df['species']

    # scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # encode the labels
    le = LabelEncoder()
    y = le.fit_transform(y)

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True)

    class MachineLearningModels:
        def __init__(self, X_train, X_test, y_train, y_test):
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
        
        def logistic_regression(self):
            model = LogisticRegression(solver='lbfgs', max_iter=1000)
            model.fit(self.X_train, self.y_train)
            self.y_pred = model.predict(self.X_test)
            self.acc_logreg = model.score(self.X_test, self.y_test)
            return self.acc_logreg

        def knn(self):
            model = KNeighborsClassifier(n_neighbors=5)
            model.fit(self.X_train, self.y_train)
            self.y_pred = model.predict(self.X_test)
            acc_knn = model.score(self.X_test, self.y_test)
            return acc_knn
            


        def plot_knn_accuracy(self):
            self.store_acc = []
            neighbors = range(1, 10)
            for k in neighbors:
                knn_improved = KNeighborsClassifier(n_neighbors=k)
                knn_improved.fit(self.X_train, self.y_train)
                self.y_pred = knn_improved.predict(self.X_test)
                self.acc_knn_improved = knn_improved.score(self.X_test, self.y_test)
                self.store_acc.append(self.acc_knn_improved)
            fig = px.line(self.store_acc, title='KNN Accuracy: {:.2f} {}'.format(self.acc_knn_improved * 100, '%'))
            fig.update_xaxes(title_text='Number of Neighbors')
            fig.update_yaxes(title_text='Accuracy')
            st.plotly_chart(fig)

        def plot_knn_confusion_matrix(self):
            cm = metrics.confusion_matrix(self.y_test, self.y_pred)
            cm  = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fig = px.imshow(cm, title='KNN Confusion Matrix: {:.2f} {}'.format(self.acc_knn_improved * 100, '%'), text_auto=True)
            fig.update_xaxes(title_text='Predicted')
            fig.update_yaxes(title_text='True')
            st.plotly_chart(fig)

        def random_forest(self):
            rf = RandomForestClassifier(n_estimators=100)
            rf.fit(self.X_train, self.y_train)
            self.y_pred = rf.predict(self.X_test)
            self.acc_rf = rf.score(self.X_test, self.y_test)

            # plot the confusion matrix
            cm = metrics.confusion_matrix(self.y_test, self.y_pred)
            cm  = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fig = px.imshow(cm, title='Random Forest Confusion Matrix: {:.2f} {}'.format(self.acc_rf * 100, '%'), text_auto=True)
            fig.update_xaxes(title_text='Predicted')
            fig.update_yaxes(title_text='True')
            st.plotly_chart(fig)


        def ada_boost(self):
            ada = AdaBoostClassifier(n_estimators=100)
            ada.fit(self.X_train, self.y_train)
            self.y_pred = ada.predict(self.X_test)
            self.acc_ada = ada.score(self.X_test, self.y_test)

            # plot the confusion matrix
            cm = metrics.confusion_matrix(self.y_test, self.y_pred)
            cm  = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fig = px.imshow(cm, title='Ada Boost Confusion Matrix: {:.2f} {}'.format(self.acc_ada * 100, '%'), text_auto=True)
            fig.update_xaxes(title_text='Predicted')
            fig.update_yaxes(title_text='True')
            st.plotly_chart(fig)

        def decision_tree(self):
            dt = DecisionTreeClassifier(max_depth=5)
            dt.fit(self.X_train, self.y_train)
            self.y_pred = dt.predict(self.X_test)
            self.acc_dt = dt.score(self.X_test, self.y_test)
            
            # plot the confusion matrix
            cm = metrics.confusion_matrix(self.y_test, self.y_pred)
            cm  = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fig = px.imshow(cm, title='Decision Tree Confusion Matrix: {:.2f} {}'.format(self.acc_dt * 100, '%'), text_auto=True)
            fig.update_xaxes(title_text='Predicted')
            fig.update_yaxes(title_text='True')
            st.plotly_chart(fig)


        def plot_all_accuracy(self):
            fig = px.bar(x=['KNN', 'Random Forest', 'Ada Boost', 'Decision Tree'], y=[self.acc_knn_improved * 100, self.acc_rf * 100, self.acc_ada *100, self.acc_dt *100], text_auto=True)
            fig.update_layout(title='Accuracy of Models', xaxis_title='Model', yaxis_title='Accuracy')
            st.plotly_chart(fig)


    
    ml = MachineLearningModels(X_train, X_test, y_train, y_test)
    ml.plot_knn_accuracy()
    ml.plot_knn_confusion_matrix()
    ml.random_forest() 
    ml.ada_boost()
    ml.decision_tree()
    ml.plot_all_accuracy()

    




            

        

