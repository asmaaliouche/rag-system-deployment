"""
fetch_data.py
-------------
This script handles the acquisition of raw event data from the
OpenAgenda v2 API. It paginates through the events of specific
agendas, filters them to the last 12 months, and saves the raw
JSON output.
"""

import json
import logging
import os
import requests
import urllib3
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress SSL warnings (proxy/Zscaler workaround)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Two Paris agendas with active events
# 326750 = "Jardins ouverts 2026" (322 events, Paris)
# 677845 = "Saint-Nicolas des Champs" (48 events, Paris)
DEFAULT_AGENDA_UIDS = [326750, 677845]


def fetch_openagenda_events(agenda_uid, api_key, max_events=300):
    """
    Fetches events from a specific agenda via the OpenAgenda v2 API.
    Paginates through all results and filters to events within the last 12 months.
    """
    url = f"https://api.openagenda.com/v2/agendas/{agenda_uid}/events"
    one_year_ago = (datetime.now(timezone.utc) - timedelta(days=365)).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )

    all_events = []
    page = 1

    logger.info(f"Fetching events from agenda {agenda_uid} (last 12 months)...")
    while len(all_events) < max_events:
        params = {
            "key": api_key,
            "size": 100,
            "timings[gte]": one_year_ago,
            "lang": "fr",
            "page": page,
        }

        response = requests.get(url, params=params, verify=False)
        if response.status_code != 200:
            logger.error(f"  Error {response.status_code}: {response.text[:200]}")
            break

        data = response.json()
        events = data.get("events", [])
        if not events:
            break

        all_events.extend(events)
        total = data.get("total", 0)
        logger.info(
            f"  Page {page}: fetched {len(events)} events (total available: {total})"
        )

        if len(all_events) >= total:
            break
        page += 1

    logger.info(f"  -> {len(all_events)} events fetched from agenda {agenda_uid}")
    return all_events


def save_raw_data(events, filename="events.json"):
    """
    Saves the raw events list to data/raw, wrapped in the expected {events: [...]} format.
    """
    os.makedirs("data/raw", exist_ok=True)
    path = os.path.join("data/raw", filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"events": events}, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(events)} events to {path}")


if __name__ == "__main__":
    load_dotenv()

    API_KEY = os.getenv("OPENAGENDA_API_KEY")
    if not API_KEY:
        logger.error("Please configure OPENAGENDA_API_KEY in your .env file")
        raise SystemExit(1)

    agenda_uid_env = os.getenv("OPENAGENDA_AGENDA_UID")
    if agenda_uid_env:
        agenda_uids = [int(agenda_uid_env)]
    else:
        agenda_uids = DEFAULT_AGENDA_UIDS

    all_events = []
    for uid in agenda_uids:
        all_events.extend(fetch_openagenda_events(uid, API_KEY))

    logger.info(f"\nTotal events collected: {len(all_events)}")
    save_raw_data(all_events)
