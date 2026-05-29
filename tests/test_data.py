import os
import json
# pyrefly: ignore [missing-import]
import pytest
import pandas as pd
from scripts.process_data import process_events

def test_process_events(tmp_path):
    # 1. Create a mock raw JSON file
    raw_data = {
        "events": [
            {
                "uid": 123,
                "title": {"fr": "Jazz Concert"},
                "description": {"fr": "A great concert."},
                "timings": [{"start": "2023-10-27T19:00:00Z", "end": "2023-10-27T22:00:00Z"}],
                "location": {
                    "name": "Parc Floral",
                    "city": "Paris",
                    "address": "Route de la Pyramide"
                },
                "canonicalUrl": "https://example.com/jazz"
            }
        ]
    }
    
    raw_file = tmp_path / "raw_events.json"
    processed_file = tmp_path / "processed_events.csv"
    
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(raw_data, f)
    
    # 2. Execute processing
    process_events(str(raw_file), str(processed_file))
    
    # 3. Assertions
    assert os.path.exists(processed_file)
    df = pd.read_csv(processed_file)
    assert len(df) == 1
    assert df.iloc[0]["title"] == "Jazz Concert"
    assert "Parc Floral" in df.iloc[0]["content"]
    assert "2023-10-27 19:00" in df.iloc[0]["content"]
    assert df.iloc[0]["city"] == "Paris"

def test_process_events_empty_fields(tmp_path):
    # 1. Create a mock raw JSON file with missing fields
    raw_data = {
        "events": [
            {
                "uid": 456,
                # title is missing
                "location": {} # location fields missing
            }
        ]
    }
    
    raw_file = tmp_path / "raw_events_empty.json"
    processed_file = tmp_path / "processed_events_empty.csv"
    
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(raw_data, f)
    
    # 2. Execute processing
    process_events(str(raw_file), str(processed_file))
    
    # 3. Assertions
    df = pd.read_csv(processed_file)
    assert len(df) == 1
    assert df.iloc[0]["title"] == "Untitled"
    assert "Location not specified" in df.iloc[0]["content"]
    assert "Aucune description disponible" in df.iloc[0]["content"]
    assert "Dates non précisées" in df.iloc[0]["content"]
