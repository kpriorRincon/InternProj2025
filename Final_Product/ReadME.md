
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

RTL-SDR
- sudo dnf install rtl-sdr
- sudo pip install pyrtlsdr

VSG60
- Install GNU Radio
- Install the VSG60 software and ensure the device works with it: https://signalhound.com/software/vsg60-software/
- Download the Software Development Kit from Signal Hound: https://signalhound.com/software/signal-hound-software-development-kit-sdk/
- Follow the directions in the SDK from device_apis/vsg60_series/lib/linux/README.txt
- Clone this repository to download the GNU Radio block: https://github.com/SignalHound/gr-vsg60.git
- From the root directory of the repository run the following commands 
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make
    $ sudo make install
    $ sudo ldconfig

Additional Software Requirements:
- GNU Radio