import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
from datetime import datetime, timezone, date


#st.metric(label="Temperature", value="70 °F", delta="1.2 °F")

st.title('Windscore')
st.subheader("**A tool for analyzing wind energy**")
st.write("This is an app designed to analyze energy output from a Windprofil.")
st.write("The core functions of this app should be : 1. Upload raw windspeed data, 2. Calculate the power output from this wind speed profil. 3. Calculate the power of a wind farm based on location information. 4. Calculation LCOE")

@st.cache
def load_data(file):
    data = pd.read_csv(file, sep=";")
    data["Datum&Uhrzeit"] = pd.to_datetime(data["Datum&Uhrzeit"])
    return data

@st.cache
def plot_FFT(df):
    fft = abs(pd.Series(np.fft.rfft(df - df.mean()),
    index = np.fft.rfftfreq(len(df), d = 1./8760))**2)
    fft.plot()
    plt.xlim(0, 768)
    plt.xlabel("1/a")
    plt.grid(True)

with st.sidebar:
    st.subheader("Upload raw data")

    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # Can be used wherever a "file-like" object is accepted:
        dataframe = load_data(uploaded_file)
        dataframe = dataframe.set_index(pd.to_datetime(dataframe["Datum&Uhrzeit"])).reset_index(drop=True)
        #dataframe = pd.to_datetime(dataframe)
    
    st.subheader('Raw data')
    show_raw_data = st.checkbox('Show Raw Data')
    
    st.subheader("Statistic")
    show_statistic = st.checkbox("Show Statistic")    
    
    st.subheader("Plotts")
    show_plot = st.checkbox("Show Plotts")  
    
    st.subheader("Anlagenparameter")
    option = st.selectbox('Windkraftanlagen', ('Enercon E30', 'Siemens', 'Nordex'))

    st.write('You selected:', option)
    st.subheader("Wirtschaftlichkeit")
    
if show_raw_data:
    st.subheader("Raw Data")
    st.dataframe(data = dataframe)

if show_statistic:
    st.subheader("Statistic")
    st.write(dataframe.describe())
    
if show_plot:
    st.subheader("Plotts")
    #Plott1
    arr = dataframe.iloc[:,-1]
    fig, ax = plt.subplots()
    ax.hist(arr, bins=30, edgecolor='black', label="Verteilung der Werte")  
    ax.grid(color="black") 
    ax.legend()
    st.pyplot(fig)
    
    #Plott2
    fig, ax = plt.subplots() 
    ax.scatter(dataframe.iloc[:,0], dataframe.iloc[:,-1],
               alpha=0.5, animated=True, marker="o", label="Windgeschwindigkeit")
    ax.legend()
    ax.grid(color="black")
    st.pyplot(fig)
    
    dataframe['date'] = pd.to_datetime(dataframe['Datum&Uhrzeit'])
    df = dataframe.set_index('date') 
    daily = df["Windgeschwindigkeit [m/s]"].resample('D').mean()
    st.write(daily)

    fig, ax = plt.subplots()
    ax = abs(pd.Series(np.fft.rfft(daily - daily.mean())))
    index = np.fft.rfftfreq(len(df), d = 1./8760)**2
    ax.plot()

    st.pyplot(fig)
    







