"""
Author: Kobe Prior
Date: June 10

This script downloads the most recent TLE (Two-Line Element) data for satellites from Celestrak only every 8 hours,
saves it to a text file, parses the TLE data into separate arrays for satellite names and TLE lines,
and then pickles these arrays for later use.
"""

import os
import requests
from datetime import datetime, timedelta, timezone


def get_up_to_date_TLE():
    """
    Downloads and parses the most recent TLE data from Celestrak, but only once every 8 hours.

    Returns:
        dict: A dictionary with keys 'names', 'line1s', and 'line2s' containing lists of satellite names and TLE lines.
    """
    url = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=last-30-days&FORMAT=tle'
    tle_file = 'sattelite_tles.txt'
    timestamp_file = 'last_tle_fetch.txt'
    update_interval = timedelta(hours=8)

    def should_update():
        if not os.path.exists(timestamp_file):
            return True  # if the timestamp file is just now being created then we should update
        with open(timestamp_file, 'r') as f:
            last_time = datetime.fromisoformat(f.read().strip())
        # returns true if the current time is more than 8 hours the last time
        return datetime.now(timezone.utc) - last_time > update_interval

    def update_timestamp():
        with open(timestamp_file, 'w') as f:
            # write the date and time to the timestamp file
            f.write(datetime.now(timezone.utc).isoformat())

    # Only download if needed
    if should_update():
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; TLE-fetcher/1.0)'}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            with open(tle_file, 'w') as f:
                f.write(response.text)
            update_timestamp()
            print('TLE data updated and saved.')
        else:
            print(
                f'Failed to retrieve TLE data. Status: {response.status_code}')
    else:
        print('TLE data is up to date. No download needed.')

    # Parse the TLE file
    names, line1s, line2s = [], [], []
    with open(tle_file) as f:
        lines = f.readlines()
        for i in range(0, min(300, len(lines)), 3):  # Only take first 100 satellites
            if i + 2 < len(lines):
                names.append(lines[i].strip())
                line1s.append(lines[i+1].strip())
                line2s.append(lines[i+2].strip())

    return {'names': names, 'line1s': line1s, 'line2s': line2s}
