import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
from datetime import datetime, timezone, date
from scipy import stats
from PIL import Image
from yaml import load
from windrose import WindroseAxes
from mpl_toolkits.axes_grid.inset_locator import inset_axes



st.title('Windscore')
st.subheader("**A tool for analyzing wind energy**")
st.write("This is an app designed to analyze energy output from a Windprofil.")
st.write("The core functions of this app should be : 1. Upload raw windspeed data, 2. Calculate the power output from this wind speed profil. 3. Calculate the power of a wind farm based on location information. 4. Calculation LCOE")

#@st.cache
def load_data(file):
    place = "Wageningen, Netherlands"
    dataframe = pd.read_csv(file,sep=",")
    dataframe["Datetime"] = pd.to_datetime(dict(year = dataframe.YEAR,
                                         month = dataframe.MO,
                                         day = dataframe.DY,
                                         hour = dataframe.HR))
    dataframe.set_index("Datetime", inplace = True)
    dataframe = dataframe[["MO","WD50M","WS50M"]]
    dataframe.columns = ["Month","Direction","Speed"]
    return dataframe

#@st.cache
def plot_FFT(dataframe):
    fft = abs(pd.Series(np.fft.rfft(dataframe - dataframe.mean()),
    index = np.fft.rfftfreq(len(dataframe), d = 1./8760))**2)
    fft.plot()
    plt.xlim(0, 768)
    plt.xlabel("1/a")
    plt.grid(True)

with st.sidebar:
    st.subheader("Upload raw data")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        dataframe = load_data(uploaded_file)
    
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
    st.header("Plotts")

    #Plott1
    st.subheader("Time-Series")
    fig, ax = plt.subplots()
    dataframe["Speed"].plot(label="Speed [m/s]")
    daily = dataframe["Speed"].resample('D').mean()
    daily.plot(color="#BB0000", label="Daily Mean")
    ax.legend()
    ax.grid(color="black")
    st.pyplot(fig)
    
    fig, ax = plt.subplots()
    dataframe.resample("M")["Speed"].mean().plot(label="Monthly Mean")
    ax.legend()
    ax.grid(color="black")
    st.pyplot(fig)
    
    fig, ax = plt.subplots(2,6, sharey = True, figsize = (20, 8))
    plt.rcParams["font.size"] = 12
    ax[0,0].plot(dataframe[dataframe.Month == 1]["Speed"],color="darkred")
    ax[0,0].set_title("January")
    ax[0,0].set_xticks([])

    ax[0,1].plot(dataframe[dataframe.Month == 2]["Speed"],color="darkred")
    ax[0,1].set_title("February")
    ax[0,1].set_xticks([])

    ax[0,2].plot(dataframe[dataframe.Month == 3]["Speed"],color="darkred")
    ax[0,2].set_title("March")
    ax[0,2].set_xticks([])

    ax[0,3].plot(dataframe[dataframe.Month == 4]["Speed"] ,color="black")
    ax[0,3].set_title("April")
    ax[0,3].set_xticks([])

    ax[0,4].plot(dataframe[dataframe.Month == 5]["Speed"],color="black")
    ax[0,4].set_title("May")
    ax[0,4].set_xticks([])

    ax[0,5].plot(dataframe[dataframe.Month == 6]["Speed"],color="black")
    ax[0,5].set_title("June")
    ax[0,5].set_xticks([])

    ax[1,0].plot(dataframe[dataframe.Month == 7]["Speed"],color="blue")
    ax[1,0].set_title("July")
    ax[1,0].set_xticks([])

    ax[1,1].plot(dataframe[dataframe.Month == 8]["Speed"],color="blue")
    ax[1,1].set_title("August")
    ax[1,1].set_xticks([])

    ax[1,2].plot(dataframe[dataframe.Month == 9]["Speed"],color="blue")
    ax[1,2].set_title("September")
    ax[1,2].set_xticks([])

    ax[1,3].plot(dataframe[dataframe.Month == 10]["Speed"],color="aqua")
    ax[1,3].set_title("October")
    ax[1,3].set_xticks([])

    ax[1,4].plot(dataframe[dataframe.Month == 11]["Speed"],color="aqua")
    ax[1,4].set_title("November")
    ax[1,4].set_xticks([])

    ax[1,5].plot(dataframe[dataframe.Month == 12]["Speed"],color="aqua")
    ax[1,5].set_title("December")
    ax[1,5].set_xticks([])

    ax[0,0].set_ylabel("Speed (m/s)")
    ax[1,0].set_ylabel("Speed (m/s)")

    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    fig.suptitle(f"Monthly wind speed in 2020 for Berlin")
    st.pyplot(fig)
    
  

    #Plott2
    st.subheader("Fast-Fourier Transformation")
    fig, ax = plt.subplots()
    ax = abs(pd.Series(np.fft.rfft(daily - daily.mean())))
    index = np.fft.rfftfreq(len(daily), d = 1./8760)**2
    ax.plot(grid=True)
    st.pyplot(fig)
    
    #Plott3
    st.subheader("Wind speed distribution")
    image = Image.open('/Users/javkhlanenkhbold/Documents/wind-power-analyzer/rawdata/Weibull-Distribution.png')
    st.image(image, caption='Weibull-Distribution')
    fig, ax = plt.subplots()
    dist = stats.weibull_min.fit(daily, floc=0, scale=2)
    speed_range = np.arange(0, daily.max())
    ax.plot(stats.weibull_min.pdf(speed_range, *(dist)),
         color = "blue",
        label = "Weibull Distribution k = {}".format(round(dist[0], 2)))
    ax.set_xlabel("Wind speed [m/s]")
    ax.set_ylabel("Probability")
    ax.hist(daily, bins=np.linspace(0, daily.max()), edgecolor='black', density=True, stacked=True, alpha=0.5, label="Daily Wind data")
    ax.grid(color="black", alpha=0.5) 
    ax.legend()
    st.pyplot(fig)
    
    
    st.subheader("Windrose")
    plt.figure(figsize = (8, 6))
    ax = WindroseAxes.from_ax()
    ax.bar(dataframe.Direction,
        dataframe.Speed,
        normed=True,    #get % of number of hours
        opening= 0.8,    #width of bars
        edgecolor='white',
        )

    ax.set_legend(loc = "best")
    plt.title(f"Wind rose diagram for Berlin")
    st.pyplot(plt)
    
    st.subheader("Windrose - Filled Mode")
    plt.figure(figsize = (8, 6))
    ax = WindroseAxes.from_ax()
    ax.contourf(dataframe.Direction, dataframe.Speed, bins=np.arange(0, 8, 1), cmap=cm.hot)
    ax.set_legend(loc = "best")
    plt.title(f"Wind rose diagram for Berlin")
    st.pyplot(plt)

   