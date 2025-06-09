from nicegui import ui
with open('viewer.html', 'r') as f:
    cesium_html = f.read()
ui.add_body_html(cesium_html)
ui.run()