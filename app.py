import streamlit as st
import streamlit_book as stb

# custom imports
from multipage import MultiPage
from pages import  exploratory_data_analysis, machine_learning

# creay an instance of the MultiPage class
app = MultiPage()


st.set_page_config(
    page_title="Iris EDA Application",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="expanded",
)

# add pages to the app
app.add_page("Exploratory Data Analysis", exploratory_data_analysis.app)
app.add_page("Machine Learning", machine_learning.app)
# app.add_page("Metadata", metadata.run)

# run the app
app.run()
