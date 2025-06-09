# run: npm install cesium
from nicegui import ui, app
import os
html_directory = os.path.dirname(__file__)#get the directory of the current script
app.add_static_files('/static', html_directory) # serve the script
#next steps I want the user to be able to copy paste several TLE's for satellites 
with ui.row().style('width: 100%; height: 100vh'):
    with ui.column().style('width: 90%'):
        ui.html('<iframe src=/static/viewer.html></iframe>').style('width: 100%; height: 90vh; border: none;')

ui.run()
