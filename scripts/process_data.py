import json
import os
import pandas as pd

def process_events(raw_data_path, processed_data_path):
    """
    Cleans and structures the raw data from OpenAgenda.
    """
    if not os.path.exists(raw_data_path):
        print(f"File not found: {raw_data_path}")
        return

    with open(raw_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    events = data.get("events", [])
    processed_list = []

    for event in events:
        # Extract fields with default values
        # We assume the API returns content primarily in French ('fr')
        title = event.get("title", {}).get("fr", "Untitled")
        description = event.get("description", {}).get("fr", "")
        long_description = event.get("longDescription", {}).get("fr", "")
        location_name = event.get("location", {}).get("name", "Location not specified")
        city = event.get("location", {}).get("city", "")
        address = event.get("location", {}).get("address", "")
        
        # Combine short and long descriptions if necessary
        full_description = description if len(description) > len(long_description) else long_description
        
        # Format text for indexing
        # Create a context-rich string
        content = f"Event: {title}\nLocation: {location_name}, {address}, {city}\nDescription: {full_description}"
        
        processed_list.append({
            "uid": event.get("uid"),
            "title": title,
            "city": city,
            "content": content,  
            "url": event.get("canonicalUrl", "")
        })

    # Save to CSV for easy loading into LangChain/Pandas
    df = pd.DataFrame(processed_list)
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    df.to_csv(processed_data_path, index=False, encoding="utf-8")
    print(f"Processed data saved to {processed_data_path} ({len(df)} events)")

if __name__ == "__main__":
    RAW_PATH = "data/raw/events.json"
    PROCESSED_PATH = "data/processed/events_structured.csv"
    process_events(RAW_PATH, PROCESSED_PATH)
