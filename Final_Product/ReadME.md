
Software Dependencies
run the following to install all packages: 
pip install -r required_pip_install.txt

Python Library Dependencies:
- Numpy
- SciPy
- matplotlib.pyplot
- NiceGUI
- crc
- pyrtlsdr
- os
- requests
- datetime
- zmq
- asyncio
- asyncssh
- satellite_czml
- skyfield.api
- pathlib
- time
- queue
- threading
- signal
- sys


'''note ctrl click satellite_czml then comment out satellites = {} because it isn't instance specific then
at the beginning of __init__() add self.satellites = {}
at the top of the class from datetime import datetime, timedelta, timezone
also replace both instances of datetime.utcnow() with datetime.now(timezone.utc)'''


Hardware Dependencies

Hardware Driver/CLI Dependencies:
- VSG60A
- BladeRF
- RTL-SDR

TODO Skylar knows the setup for vsg 60
sudo dnf install rtl-sdr
download software development kit and the gnu radio block for VSG and the driver

Additional Software Requirements:
- GNU Radio