import streamlit as st
import pandas as pd
import numpy as np
from bokeh.plotting import figure

st.title('Windscore')
st.subheader("**A tool for analyzing wind energy**")
st.subheader("Created by Javkhlan Enkhbold")

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')

data_path = "rawdata/wind_berlin.csv"


@st.cache
def load_data(file):
    data = pd.read_csv(file, sep=";")
    data["Datum&Uhrzeit"] = pd.to_datetime(data["Datum&Uhrzeit"])
    return data



# Load 10,000 rows of data into the dataframe.
data = load_data(file=data_path)

st.write(type(data["Datum&Uhrzeit"]))

# Notify the reader that the data was successfully loaded.
data_load_state.text("Data is done!")


st.subheader('Raw data')
st.write(data)


st.line_chart(data)