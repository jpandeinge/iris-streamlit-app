import re
import time
import numpy as np
import pandas as pd

import plotly.express as px
import matplotlib.pyplot as plt
from st_aggrid import AgGrid # streamlit grid components


import streamlit as st



def app():

    st.markdown("## Exploratory Data Analysis")
   
    st.markdown(
        """
        This an exploratory data analysis dashboard for the Iris dataset, we we will
        use the [streamlit](https://streamlit.io/) framework to build this dashboard.
        """
        )

    st.sidebar.success('EDA')

    dataset_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"

    @st.experimental_memo
    def get_iris_data() -> pd.DataFrame:
        return pd.read_csv(dataset_url)

    df = get_iris_data()

    class Plot:
        def __init__(self, df):
            self.df = df

        def plot_class_distribution(self):
            count_list = [(df['species'] == 'setosa').sum(), (df['species'] == 'versicolor').sum(), (df['species'] == 'virginica').sum()]
            labels = ['setosa', 'versicolor', 'virginica']
            fig = px.pie(count_list, labels=labels, values=count_list, title='Iris Species Distribution')
            st.plotly_chart(fig)
        
        def plot_correlation_matrix(self):
            fig = px.imshow(df.corr(), title='Correlation Matrix', text_auto=True)
            st.plotly_chart(fig)
        
        def plot_multivariate_scatter(self):
            fig = px.scatter_matrix(df, dimensions=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
                                    color='species',
                                    template='plotly_dark')  
            st.plotly_chart(fig)

        def plot_histogram(self):   
            fig = px.histogram(df, x='sepal_length', nbins=20, title='Sepal Length Histogram')
            st.plotly_chart(fig)

        def plot_species_class(self):
            fig = px.scatter(df, x="sepal_width", y="sepal_length", facet_col="species")
            fig.update_xaxes(title_font=dict(size=14, family='Courier', color='crimson'))
            fig.update_yaxes(title_font=dict(size=14, family='Courier', color='crimson'))
            st.plotly_chart(fig)


    plot = Plot(df)
    plot.plot_class_distribution()
    plot.plot_species_class()
    plot.plot_correlation_matrix()
    plot.plot_multivariate_scatter()
    plot.plot_histogram()

    


    


