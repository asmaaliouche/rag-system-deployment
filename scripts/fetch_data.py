import os
import json
import requests
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

def fetch_openagenda_events(agenda_uid, api_key, city="Paris"):
    """
    Fetches events from a specific agenda via the OpenAgenda v2 API.
    """
    url = f"https://api.openagenda.com/v2/agendas/{agenda_uid}/events"
    params = {
        "key": api_key,
        "location[city]": city,
        "size": 100  # Number of events to fetch
    }
    
    print(f"Fetching events from agenda {agenda_uid} for city {city}...")
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

def save_raw_data(data, filename="events.json"):
    """
    Saves the raw data to the data/raw directory.
    """
    os.makedirs("data/raw", exist_ok=True)
    path = os.path.join("data/raw", filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Data saved to {path}")

if __name__ == "__main__":
    load_dotenv()
    
    API_KEY = os.getenv("OPENAGENDA_API_KEY")
    # Agenda "What to do in Paris?" - UID to be confirmed or changed according to your needs
    # If you don't have a specific UID, we can look for a public one.
    AGENDA_UID = os.getenv("OPENAGENDA_AGENDA_UID", "91244770") 
    
    if not API_KEY:
        print("Please configure OPENAGENDA_API_KEY in your .env file")
    else:
        events_data = fetch_openagenda_events(AGENDA_UID, API_KEY)
        if events_data:
            save_raw_data(events_data)
