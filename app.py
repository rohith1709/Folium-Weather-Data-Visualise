from flask import Flask, render_template, request
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

app = Flask(__name__)

data = pd.read_csv("weather_data.csv")

city_names = data['location_name'].unique()

data['sunrise_time'] = pd.to_datetime(data['sunrise'], format='%I:%M:%S %p', errors='coerce').dt.time
data['sunrise_time'] = data['sunrise_time'].fillna(pd.to_datetime(data['sunrise'], format='%I:%M %p', errors='coerce').dt.time)

data['sunset_time'] = pd.to_datetime(data['sunset'], format='%I:%M:%S %p', errors='coerce').dt.time
data['sunset_time'] = data['sunset_time'].fillna(pd.to_datetime(data['sunset'], format='%I:%M %p', errors='coerce').dt.time)

def time_to_fractional_hours(t):
    if pd.isnull(t):
        return np.nan
    return t.hour + t.minute / 60 + t.second / 3600

data['sunrise_hours'] = data['sunrise_time'].apply(time_to_fractional_hours)
data['sunset_hours'] = data['sunset_time'].apply(time_to_fractional_hours)

data['moonrise_time'] = pd.to_datetime(data['moonrise'], format='%I:%M:%S %p', errors='coerce').dt.time
data['moonrise_time'] = data['moonrise_time'].fillna(pd.to_datetime(data['moonrise'], format='%I:%M %p', errors='coerce').dt.time)

data['moonset_time'] = pd.to_datetime(data['moonset'], format='%I:%M:%S %p', errors='coerce').dt.time
data['moonset_time'] = data['moonset_time'].fillna(pd.to_datetime(data['moonset'], format='%I:%M %p', errors='coerce').dt.time)

data['moonrise_hours'] = data['moonrise_time'].apply(time_to_fractional_hours)
data['moonset_hours'] = data['moonset_time'].apply(time_to_fractional_hours)

def plot_temperature(x,y,z):
    plt.figure(figsize=(13, 7))
    plt.plot(x['last_updated'], x['temperature_celsius'], marker='o', linestyle='-', label='Temperature')
    plt.plot(x['last_updated'], x['feels_like_celsius'], marker='o', linestyle='-', label='Feels Like')
    plt.title(f'Temperature in {y}')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.xticks(rotation=45)
    
    plt.xticks(z, x['last_updated'].iloc[z], rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/temperature_plot.png')  
    plt.close()

def plot_humidity(x,y,z):
    plt.figure(figsize=(13, 7))
    plt.plot(x['last_updated'], x['humidity'], marker='o', linestyle='-')
    plt.title(f'Humidity in {y}')
    plt.xlabel('Date')
    plt.ylabel('Humidity (%)')
    plt.xticks(rotation=45)
    plt.xticks(z, x['last_updated'].iloc[z], rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('static/humidity_plot.png')
    plt.close()

def plot_wind(x,y,z):
    plt.figure(figsize=(13, 7))
    plt.plot(x['last_updated'], x['wind_kph'], marker='o', linestyle='-')
    plt.title(f'Wind Speed in {y}')
    plt.xlabel('Date')
    plt.ylabel('Wind Speed (kph)')
    plt.xticks(rotation=45)
    plt.xticks(z, x['last_updated'].iloc[z], rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('static/wind_speed_plot.png') 
    plt.close()

def plot_temperature_humidity_scatter(x,y):
    plt.figure(figsize=(13, 7))
    sns.scatterplot(x='temperature_celsius', y='humidity', data=x)
    plt.title(f'Temperature vs Humidity in {y}')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Humidity (%)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('static/temperature_humidity_scatter.png')
    plt.close()

def plot_air_CO(x,y,z):
    plt.figure(figsize=(13, 7))
    plt.plot(x['last_updated'], x['air_quality_Carbon_Monoxide'], marker='o', linestyle='-', label='Carbon Monoxide')
    plt.title(f'Air Quality CO in {y}')
    plt.xlabel('Date')
    plt.ylabel('Air Quality')
    plt.xticks(rotation=45)
    plt.xticks(z, x['last_updated'].iloc[z], rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/air_quality_co_plot.png') 
    plt.close()

def plot_air_ozone(x,y,z):
    plt.figure(figsize=(13, 7))
    plt.plot(x['last_updated'], x['air_quality_Ozone'], marker='o', linestyle='-', label='Ozone')
    plt.title(f'Air Quality (CO & Ozone) in {y}')
    plt.xlabel('Date')
    plt.ylabel('Air Quality')
    plt.xticks(rotation=45)
    plt.xticks(z, x['last_updated'].iloc[z], rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/air_quality_ozone_plot.png') 
    plt.close()

def plot_air_NO(x,y,z):
    plt.figure(figsize=(13, 7))
    plt.plot(x['last_updated'], x['air_quality_Nitrogen_dioxide'], marker='o', linestyle='-', label='Nitrogen Dioxide')
    plt.title(f'Air Quality NO in {y}')
    plt.xlabel('Date')
    plt.ylabel('Air Quality')
    plt.xticks(rotation=45)
    plt.xticks(z, x['last_updated'].iloc[z], rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/air_quality_NO_plot.png') 
    plt.close()

def plot_air_SO(x,y,z):
    plt.figure(figsize=(13, 7))
    plt.plot(x['last_updated'], x['air_quality_Sulphur_dioxide'], marker='o', linestyle='-', label='Sulphur Dioxide')
    plt.title(f'Air Quality SO in {y}')
    plt.xlabel('Date')
    plt.ylabel('Air Quality')
    plt.xticks(rotation=45)
    plt.xticks(z, x['last_updated'].iloc[z], rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/air_quality_SO_plot.png') 
    plt.close()

def plot_air_PM(x,y,z):
    plt.figure(figsize=(13, 7))
    for var in ['air_quality_PM2.5', 'air_quality_PM10']:
        plt.plot(x['last_updated'], x[var], marker='o', linestyle='-', label=var)
    plt.title(f'Air Quality (PM2.5, PM10) in {y}')
    plt.xlabel('Date')
    plt.ylabel('Air Quality')
    plt.xticks(rotation=45)
    plt.xticks(z, x['last_updated'].iloc[z], rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/air_quality_PM_plot.png')
    plt.close()

def plot_sunrise_sunset(x,y,z):
    plt.figure(figsize=(13, 7))
    plt.plot(x['last_updated'], x['sunrise_hours'], marker='o', linestyle='-', label='Sunrise')
    plt.plot(x['last_updated'], x['sunset_hours'], marker='o', linestyle='-', label='Sunset')
    plt.title(f'Sunrise and Sunset Times in {y}')
    plt.xlabel('Date')
    plt.ylabel('Time of Day (Hours)')
    plt.xticks(rotation=45)
    plt.xticks(z, x['last_updated'].iloc[z], rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/sunrise_sunset_plot.png')
    plt.close()

def plot_moonrise_moonset(x,y,z):
    plt.figure(figsize=(13, 7))
    plt.plot(x['last_updated'], x['moonrise_hours'], marker='o', linestyle='-', label='Moonrise')
    plt.plot(x['last_updated'], x['moonset_hours'], marker='o', linestyle='-', label='Moonset')
    plt.title(f'Moonrise and Moonset Times in {y}')
    plt.xlabel('Date')
    plt.ylabel('Time of Day (Hours)')
    plt.xticks(rotation=45)
    plt.xticks(z, x['last_updated'].iloc[z], rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/moonrise_moonset_plot.png')
    plt.close()

def plot_moon_illumination(x,y,z):
    plt.figure(figsize=(13, 7))
    plt.plot(x['last_updated'], x['moon_illumination'], marker='o', linestyle='-', label='Moon Illumination')
    plt.title(f'Moon Illumination in {y}')
    plt.xlabel('Date')
    plt.ylabel('Moon Illumination (%)')
    plt.xticks(rotation=45)
    plt.xticks(z, x['last_updated'].iloc[z], rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/moon_illumination_plot.png')
    plt.close()

def plot_moon_phase(x,y):
    plt.figure(figsize=(13, 7))
    moon_phase_counts = x['moon_phase'].value_counts()
    sns.barplot(x=moon_phase_counts.index, y=moon_phase_counts.values)
    plt.title(f'Moon Phase Frequencies in {y}')
    plt.xlabel('Moon Phase')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('static/moon_phase_plot.png')
    plt.close()

def plot_city_data(city_name):
    city_data = data[data['location_name'] == city_name]
    n = 20
    x_ticks = np.linspace(0, len(city_data)-1, n, dtype=int)
    plot_temperature(city_data,city_name, x_ticks)
    plot_humidity(city_data,city_name, x_ticks)
    plot_wind(city_data,city_name, x_ticks)
    plot_temperature_humidity_scatter(city_data,city_name)
    plot_air_CO(city_data,city_name, x_ticks)
    plot_air_ozone(city_data,city_name, x_ticks)
    plot_air_NO(city_data,city_name, x_ticks)
    plot_air_SO(city_data,city_name, x_ticks)
    plot_air_PM(city_data,city_name, x_ticks)
    plot_sunrise_sunset(city_data,city_name, x_ticks)
    plot_moonrise_moonset(city_data,city_name, x_ticks)
    plot_moon_illumination(city_data,city_name, x_ticks)
    plot_moon_phase(city_data,city_name)

    

@app.route('/')
def index():
    return render_template('index.html', city_names=city_names)

@app.route('/plot', methods=['POST'])
def plot():
    city_name = request.form['city']
    plot_city_data(city_name)
    return render_template('plot.html', city=city_name)

if __name__ == '__main__':
    app.run(debug=True)