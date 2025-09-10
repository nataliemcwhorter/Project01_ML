'''This is weather.py
It creates a weather dataset using meteostat. It takes the lng/lat for each circuit
as well as the different session times and then gives the precipitation during
the session. It is then written into ../data/weather.csv'''
__author__ = 'Natalie McWhorter'
__version__ = '09.02.2025'

import ssl
import urllib3
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import pandas as pd
from datetime import datetime
from meteostat import Point, Hourly

# Load data
circuits_df = pd.read_csv('../data/circuits.csv')
races_df = pd.read_csv('../data/races.csv')

# Filter for 2000-2025
races_filtered = races_df[(races_df['year'] >= 2000) & (races_df['year'] <= 2025)]

# Merge to get lat/lon for each race
merged_df = races_filtered.merge(circuits_df, on='circuitId')

# Create  list for results
weather_results = []

total_races = len(merged_df)
print(f"Processing {total_races} races from 2000-2025...")


def get_session_weather(location, session_date, session_time, default_time='14:00:00'):
    """Get weather data for a specific session"""
    if pd.isna(session_date) or pd.isna(session_time):
        return None
    try:
        session_datetime = datetime.strptime(f"{session_date} {session_time}", '%Y-%m-%d %H:%M:%S')
        start = session_datetime.replace(hour=max(0, session_datetime.hour - 1))
        end = session_datetime.replace(hour=min(23, session_datetime.hour + 3))
        session_data = Hourly(location, start, end).fetch()
        return session_data['prcp'].sum()
    except:
        return None


for index, row in merged_df.iterrows():
    if index % 10 == 0:
        print(f"Processing race {index + 1}/{total_races}")
    try:
        raceId = row['raceId']
        circuitId = row['circuitId']
        latitude = row['lat']
        longitude = row['lng']
        location = Point(latitude, longitude)

        # Initialize result dictionary
        race_weather = {
            'circuitId': circuitId,
            'raceId': raceId
        }

        # Get weather for each session
        race_weather['race_prcp'] = get_session_weather(location, row['date'], row['time'])
        race_weather['quali_prcp'] = get_session_weather(location, row['quali_date'], row['quali_time'])
        race_weather['fp1_prcp'] = get_session_weather(location, row['fp1_date'], row['fp1_time'])
        race_weather['fp2_prcp'] = get_session_weather(location, row['fp2_date'], row['fp2_time'])
        race_weather['fp3_prcp'] = get_session_weather(location, row['fp3_date'], row['fp3_time'])
        race_weather['sprint_prcp'] = get_session_weather(location, row['sprint_date'], row['sprint_time'])

        # Add to results
        weather_results.append(race_weather)

    except Exception as e:
        print(f"Error processing race {raceId}: {e}")
        weather_results.append({
            'circuitId': circuitId,
            'raceId': raceId,
            'race_prcp': None,
            'quali_prcp': None,
            'fp1_prcp': None,
            'fp2_prcp': None,
            'fp3_prcp': None,
            'sprint_prcp': None
        })
        continue


# Save results
weather_df = pd.DataFrame(weather_results)
weather_df = weather_df.fillna('\\N')  # Replace null values with N/A
weather_df.to_csv('../data/weather.csv', index=False)
print(f"Weather data saved to ../data/weather.csv with {len(weather_df)} races")