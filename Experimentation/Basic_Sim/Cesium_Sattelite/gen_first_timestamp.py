# just a helper file to create the first time stamp to avoid multiple request to get_TLE from celestrak
from datetime import datetime, timezone
timestamp_file = 'last_tle_fetch.txt'
with open(timestamp_file, 'w') as f:
    # write the date and time to the timestamp file
    f.write(datetime.now(timezone.utc).isoformat())
