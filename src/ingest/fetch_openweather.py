# src/ingest/fetch_openweather.py
import requests, os, json
import pandas as pd
from datetime import datetime

def fetch_and_save_raw(city='Karachi', out_dir='data/raw'):
    os.makedirs(out_dir, exist_ok=True)
    coords = {'Karachi': (24.8607, 67.0011)}
    lat, lon = coords.get(city)
    if lat is None:
        raise ValueError("Unknown city")
    key = os.environ.get('OPENWEATHER_API_KEY')
    if not key:
        raise EnvironmentError("OPENWEATHER_API_KEY not set")

    # Using One Call 2.5 or 3.0 endpoint as available
    url = f"https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude=minutely,daily,alerts&units=metric&appid={key}"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    raw_json_path = os.path.join(out_dir, f"raw_{city}_{ts}.json")
    with open(raw_json_path, 'w') as f:
        json.dump({'fetched_at': ts, 'response': data}, f)

    rows = []
    for h in data.get('hourly', []):
        rows.append({
            'dt': pd.to_datetime(h['dt'], unit='s', utc=True),
            'temp': h.get('temp'),
            'feels_like': h.get('feels_like'),
            'humidity': h.get('humidity'),
            'wind_speed': h.get('wind_speed'),
            'pressure': h.get('pressure'),
            'clouds': h.get('clouds'),
            'visibility': h.get('visibility', None),
            'weather_id': h['weather'][0]['id'] if h.get('weather') else None,
            'fetched_at': ts
        })
    df = pd.DataFrame(rows)
    parquet_path = os.path.join(out_dir, f"parsed_{city}_{ts}.parquet")
    df.to_parquet(parquet_path, index=False)
    return parquet_path

if __name__ == "__main__":
    print(fetch_and_save_raw())
