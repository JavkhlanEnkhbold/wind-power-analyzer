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
import math



st.title('Windscore')
st.subheader("**A tool for analyzing wind energy**")
st.write("This is an app designed to analyze energy output from a Windprofil.")
st.write("The core functions of this app should be : 1. Upload raw windspeed data, 2. Calculate the power output from this wind speed profil. 3. Calculate the power of a wind farm based on location information. 4. Calculation LCOE")

#@st.cache
def load_data(file):
    dataframe = pd.read_csv(file,sep=",")
    dataframe["Datetime"] = pd.to_datetime(dict(year = dataframe.YEAR,
                                         month = dataframe.MO,
                                         day = dataframe.DY,
                                         hour = dataframe.HR))
    dataframe.set_index("Datetime", inplace = True)
    dataframe = dataframe[["MO","WD50M","WS50M"]]
    dataframe.columns = ["Month","Direction","Speed"]
    return dataframe

def read_units(file):
    df = pd.read_csv(file, sep=",")
    return df

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
    df = read_units("/Users/javkhlanenkhbold/Documents/wind-power-analyzer/rawdata/supply__wind_turbine_library.csv")
    type = st.selectbox('Modell', (name for name in df["name"]))
    
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
    
    #Plott4


    
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
    
    st.subheader("Wind Turbine")
    st.write("The power curve of a wind turbine is a graph that depicts how much electrical power output is produced by a wind turbine at different wind speeds. These curves are found by field measurements, where the wind speed reading from a device called an anemometer (which is placed on a mast at a reasonable distance to the wind turbine) is read and plotted against the electrical power output from the turbine.")
    
if type:
    st.subheader("Anlagentype:" )
    df = read_units("/Users/javkhlanenkhbold/Documents/wind-power-analyzer/rawdata/supply__wind_turbine_library.csv")
    df['power_curve_wind_speeds'] = df['power_curve_wind_speeds']
    df['power_curve_values'] = df['power_curve_values']
    st.write(df)
    selected_unit = df.loc[df["name"] == type]
    
    lst_wind_speed = selected_unit['power_curve_wind_speeds'].to_list()[0]
    lst_wind_speed = [float(x.strip(' []')) for x in lst_wind_speed.split(',')]
    
    lst_curve_values = selected_unit['power_curve_values'].to_list()[0]
    lst_curve_values = [float(x.strip(' []')) for x in lst_curve_values.split(',')]
    
    lst_merged = [lst_wind_speed, lst_curve_values]
    power_curve = pd.DataFrame(lst_merged).transpose().set_axis(['wind_speed_class', 'Power at given speed'], axis=1, inplace=False)
    power_curve.set_index("wind_speed_class", inplace = True)
    
    cut_in_speed = 3 
    cut_out_speed = 25
    rated_speed = 14
    rated_power = 810
    
    st.write(power_curve)
    
    st.subheader("Power Curve")
    fig, ax = plt.subplots()

    ax = power_curve["Power at given speed"].plot(color = "darkblue", linewidth = 4, label = "Power Curve")

    ax.vlines(x = cut_in_speed,
            ymin = 0,
            ymax = power_curve.loc[cut_in_speed, "Power at given speed"],
            linestyle = "dashed",
            color = "black"
            )

    ax.vlines(x = cut_out_speed,
            ymin = 0,
            ymax = power_curve.loc[cut_out_speed, "Power at given speed"],
            color = "darkblue",
            linewidth = 4
            )

    ax.hlines(y = 0,
            xmin = cut_out_speed,
            xmax = 30,
            color = "darkblue",
            linewidth = 4
            )

    plt.xlabel("Speed (m/s)")
    plt.ylabel("Power at given speed (kW)")
    plt.title("Power curve diagram of given wind turbine")

    #plt.xlim(0, 30)
    #plt.ylim(0, 1000)
    plt.legend(loc = "upper left")
    st.pyplot(plt)

    