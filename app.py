import streamlit as st
import pandas as pd
import numpy as np
from windrose import WindroseAxes
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
from datetime import datetime
import time

#st.metric(label="Temperature", value="70 °F", delta="1.2 °F")

st.title('Windscore')
st.subheader("**A tool for analyzing wind energy**")

@st.cache
def load_data(file):
    data = pd.read_csv(file, sep=";")
    data["Datum&Uhrzeit"] = pd.to_datetime(data["Datum&Uhrzeit"])
    return data

with st.sidebar:
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # Can be used wherever a "file-like" object is accepted:
        dataframe = load_data(uploaded_file)
        st.write("Sucess.")


st.subheader('Raw data')
show_raw_data = st.checkbox('Show Raw Data')
if show_raw_data:
     st.dataframe(data = dataframe)

st.subheader("Statistik")
show_statistic = st.checkbox("Show Statistic")
if show_statistic:
    st.write(dataframe.describe())
st.subheader("Plotts")
show_plot = st.checkbox("Show Plotts")
if show_plot:
    
    st.bar_chart(dataframe)






