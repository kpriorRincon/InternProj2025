# run: npm install cesium
from nicegui import ui
with open('cesium_getting_started.html', 'r') as f:
    cesium_html = f.read()
ui.add_body_html(cesium_html)
ui.run()