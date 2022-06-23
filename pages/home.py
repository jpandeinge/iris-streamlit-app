import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
# from st_aggrid import AgGrid  # streamlit grid components

# set up dashboard
st.set_page_config(
    page_title="Iris Dataset EDA",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Iris Dataset EDA")
st.markdown(
    """
    This an exploratory data analysis dashboard for the Iris dataset, we we will
    use the [streamlit](https://streamlit.io/) framework to build this dashboard.
    """
)

# insert an image
st.image("src/images/iris-dataset.png", width=700)
# credit the source
st.markdown(
    """
    Source: [machine learning HD](https://machinelearninghd.com/iris-dataset-uci-machine-learning-repository-project/)
    """
)



dataset_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"

@st.cache
def get_iris_data() -> pd.DataFrame:
    return pd.read_csv(dataset_url)

df = get_iris_data()

#  show the data
st.markdown("## Data Preview")
st.write(df)


    
