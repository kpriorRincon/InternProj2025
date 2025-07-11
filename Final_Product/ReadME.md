
Software Dependencies
run the following to install all packages: 
pip install -r required_pip_install.txt



'''note ctrl click satellite_czml then comment out satellites = {} because it isn't instance specific then
at the beginning of __init__() add self.satellites = {}
at the top of the class from datetime import datetime, timedelta, timezone
also replace both instances of datetime.utcnow() with datetime.now(timezone.utc)'''


Hardware Dependencies
TODO Skylar knows the setup for vsg 360
sudo dnf install rtl-sdr
download software development kit and the gnu radio block for VSG and the driver
