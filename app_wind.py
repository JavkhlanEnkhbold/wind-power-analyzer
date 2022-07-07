from tkinter.ttk import Style
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
import seaborn as sns



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
    show_analysis = st.checkbox("Show analysis")
    
if show_raw_data:
    st.subheader("Raw Data")
    st.dataframe(data = dataframe)

if show_statistic:
    st.subheader("Statistic")
    st.write(dataframe.describe())
    
if show_plot:
    st.header("Plotts")

    #Plott1. Time Series
    st.subheader("Time-Series")
    fig, ax = plt.subplots()
    dataframe["Speed"].plot(label="Daily", ylabel="Speed [m/s]")
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
    
    #Plott2: Seasonal Pattern
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
    
    #Plott3: Windrose
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
    
    
    #Plott4: Anlage
    if type:
        #st.subheader("Selected unit/modell: {}".format(str(type)) )
        df = read_units("/Users/javkhlanenkhbold/Documents/wind-power-analyzer/rawdata/supply__wind_turbine_library.csv")
        df['power_curve_wind_speeds'] = df['power_curve_wind_speeds']
        df['power_curve_values'] = df['power_curve_values']
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
    
    # Definition of Dataframe
    df = dataframe
    data = df.Speed.values
    params = stats.weibull_min.fit(data,
                             floc = 0,     #Fix the location at zero
                             scale = 2    #keeps the first scale parameter of the exponential weibull fixed at once
                            )
    
    df["Speed Class"] = ""
    for index in df.index:
        df.loc[index, "Speed Class"] = math.ceil(df.loc[index, "Speed"])
    
    #st.subheader("Wind Class in details")
    #st.write(df)
    
    st.subheader("Frequency in Details")
    df_new = df.groupby(["Speed Class"]).count()
    df_new = df_new[["Speed"]]
    df_new.columns = ["Observed Frequency"]
    df_new["Cumulative Frequency"] = df_new["Observed Frequency"].cumsum()
    df_new.sort_index(inplace = True)    
   
    
    #Plott5: Weibull Verteilung
    fig, ax = plt.subplots()
    speed_range = np.arange(0, 26)
    ax.plot(df_new["Observed Frequency"],
            color = "navy",
            label = "Observed frequency",
            linewidth = 3
        )
    ax.plot(stats.weibull_min.pdf(speed_range, *params)*len(df),
            color = "turquoise",
            label = "Theoretical frequency - Weibull Distr.",
        linewidth = 3)

    ax.hist(df["Speed"],
            bins = 21,      #number of bars or classes
            rwidth = 0.85,    #width of bars
            edgecolor = "black",
            color = "cornflowerblue",
            label = "Wind speed histogram"
            )
    ax.set_ylabel("Frequency (Number of hours)", color = "navy")
    ax2 = ax.twinx()
    ax2.plot(df_new["Cumulative Frequency"],
            color = "red",
            label = "Observed Cumulative frequency",
            linewidth = 3)

    ax2.set_ylabel("Cumulative frequency (number of hours)",
                color = "red",
                )

    ax2.plot(stats.weibull_min.cdf(speed_range, *params)*len(df),
            color = "pink",
            label = "Theoretical Cumulative density",
            linewidth = 3)

    ax.set_xlabel("Speed Class (m/s)")

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    lgd = fig.legend(lines, labels, ncol = 2, 
            bbox_to_anchor = (0.87, 0.05))

    plt.title("Theoretical and observed wind speed distribution in Berlin")
    st.pyplot(plt)

    #Plott6: Power Curve
    #Enercon E-82/3000    
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
    plt.legend(loc = "upper left")
    st.pyplot(plt)
        
    cut_in_speed = 3 
    cut_out_speed = 25
    rated_speed = 17
    rated_power = 3020
    
    new_row = {"Observed Frequency": [0, 0, 0, 0]}
    df_new = df_new.append(pd.DataFrame(new_row), ignore_index=True)
    
    power_curve["Hours"] = df_new.index.map(df_new["Observed Frequency"])
    power_curve["Frequency (%)"] = power_curve["Hours"]/power_curve["Hours"].sum()
    power_curve["Power production distribution"] = power_curve["Frequency (%)"] * power_curve["Power at given speed"]/100
    power_curve["Energy yield"] = power_curve["Power at given speed"] * power_curve["Hours"]
    st.write(power_curve)
    #st.write(df_new)
    
    rated_power = 3020
    capacity_factor = power_curve["Energy yield"].sum() / (power_curve["Hours"].sum() * rated_power)
    capacity_factor = round(capacity_factor, 2)
    st.write(f"The capacity factor of the given wind turbine at Berlin in 2020 was ", capacity_factor*100, "%.")

    fig, ax = plt.subplots()
    ax.plot(stats.weibull_min.pdf(speed_range, *params),
            color = "blue",
            label = "Wind speed distribution",
        linewidth = 2)

    power_curve["Power production distribution"].plot(ax = ax,
                                                    color = "cornflowerblue",
                                                    label = "Power production distribution (%)")

    ax.set_ylabel("Frequency (%)", color = "blue")

    ax2 = ax.twinx()

    ax2 = power_curve["Power at given speed"].plot(color = "red", linewidth = 2, label = "Power Curve")


    ax2.vlines(x = cut_out_speed,
            ymin = 0,
            ymax = power_curve.loc[cut_out_speed, "Power at given speed"],
            color = "red",
            linewidth = 2
            )

    ax2.hlines(y = 0,
            xmin = cut_out_speed,
            xmax = 30,
            color = "red",
            linewidth = 2
            )

    ax2.set_ylabel("Power (kW)")

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, ncol = 2, 
            bbox_to_anchor = (0.87, 0.05))

    st.pyplot(fig)

if show_analysis:
    st.subheader("Economic Analysis")
    # i: Zinssatz bzw WACC, 
    # G: Kapazität (Nennleistung der Anlage) in kW, 
    # Q: Energieerzeugung pro Jahr in kWh (das muss aus der Standortanalyse kommen), 
    # C: Investitionskosten (Kapitalkosten) pro kW Kapazität, 
    # O_var= variable Betriebskosten in Euro/kWh, 
    # O_fix= konstante Betriebskosten in Euro/kW  
    # 'global' parameters habe ich für jupyter benutzt um jeden Parameter nachher beliebig abzurufen. 
    # Du kannst diese einfach weglassen mit Streamlit
    def lcoe(i, G, Q, C, O_var, O_fix, T):
        #aus Zinssatz und Lebensdauer den PVF "Present Value Factor" berechnen. Das Annuitätsfaktor "a" ist dann 1/PVF.
        pvf=(1/i)*(1-1/(1+i)**T)
        a=1/pvf
        #Investitionskosten für die Anlage in Euro (für das erste Jahr)
        I_0= G*C
        # Konstante Betriebskosten für die Anlage in Euro/a 
        B=G*O_fix
        # LCOE berechnen: 1) I_0 mit a multiplizieren um die Kapitalkosten auf die gesamte Lebensdauer von der Turbine zu verteilen und dabei Zeitwert des Geldes zu berücksichtigen. Daraus ergibt sich Investitionskosten pro jahr (Euro/a)
        #                 2) Summiere jährliche Investitionskosten mit jährlichen festen Betriebskosten (B) und Teile diese Summe dann die jährliche Erzeugung (kWh/a). Daraus bekommst du Euro/kWh  
        #                 3) Darauf kommen die varable Kosten (Marginal Costs inklusive den Brennstoffkosten, hier 0, und andere variable Kosten)
        p=(I_0*a+B)/Q+O_var
        
        return(I_0, B, pvf, a, p)
    #Abweichung für die Parameter 'C' Kapitalkosten 
    abw=np.arange(0, 2, 0.05, dtype=float)
    i = st.number_input('Zinssatz')
    G = st.slider('Kapazität', 1, 100, 3)
    Q = st.slider('Energieerzeugung pro Jahr in kWh', 1, 100, 50)
    C = 1700 * abw
    O_var = st.slider("variable Betriebskosten in €/kWh", 1, 100, 35)
    O_fix = st.slider("konstante Betriebskosten in €/kWh", 1, 100, 25)
    T = st.slider("Zeitraum", 1, 30, 10)
    #i=0.07
    #G=1
    #Q=1000
    #C=1700*abw
    #O_fix=20
    #O_var=0.008
    #T=20
    lcoe  = lcoe(i, G, Q, C, O_var, O_fix, T)
    column_names=['Abweichung']
    sensitivity=pd.DataFrame(abw,index=None, columns=column_names)
    sensitivity['LCOE [Euro/kWh]'] = lcoe[-1] 
    fig = plt.figure()
    sensitivity.plot(x='Abweichung' , grid=True, style= "bo-", y='LCOE [Euro/kWh]')
    st.pyplot(plt)
    st.write(sensitivity)
    