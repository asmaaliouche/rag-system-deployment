"""
process_data.py
---------------
This script reads the raw JSON events fetched from OpenAgenda,
cleans the data, handles missing fields, structures the event
information into a cohesive text format, and exports it as a CSV
file ready for indexing.
"""

import json
import logging
import os
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def process_events(raw_data_path, processed_data_path):
    """
    Cleans and structures the raw data from OpenAgenda.
    """
    if not os.path.exists(raw_data_path):
        logger.error(f"File not found: {raw_data_path}")
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
        full_description = (
            description
            if len(description) > len(long_description)
            else long_description
        )
        if not full_description:
            full_description = "Aucune description disponible."

        # Extract and format timings
        timings = event.get("timings", [])
        formatted_timings = []
        for t in timings:
            start = t.get("start", "")
            end = t.get("end", "")
            if start:
                # Basic formatting for ISO strings: 2023-10-27T19:00:00.000Z -> 2023-10-27 19:00
                start_fmt = start.replace("T", " ")[:16]
                if end:
                    end_fmt = end.replace("T", " ")[:16]
                    formatted_timings.append(f"du {start_fmt} au {end_fmt}")
                else:
                    formatted_timings.append(f"le {start_fmt}")

        timing_str = (
            "; ".join(formatted_timings) if formatted_timings else "Dates non précisées"
        )

        # Format text for indexing
        # Create a context-rich string
        content = f"Event: {title}\nLocation: {location_name}, {address}, {city}\nDates: {timing_str}\nDescription: {full_description}"

        processed_list.append(
            {
                "uid": event.get("uid"),
                "title": title,
                "city": city,
                "content": content,
                "url": event.get("canonicalUrl", ""),
            }
        )

    # Save to CSV for easy loading into LangChain/Pandas
    df = pd.DataFrame(processed_list)
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    df.to_csv(processed_data_path, index=False, encoding="utf-8")
    logger.info(f"Processed data saved to {processed_data_path} ({len(df)} events)")


if __name__ == "__main__":
    RAW_PATH = "data/raw/events.json"
    PROCESSED_PATH = "data/processed/events_structured.csv"
    process_events(RAW_PATH, PROCESSED_PATH)
