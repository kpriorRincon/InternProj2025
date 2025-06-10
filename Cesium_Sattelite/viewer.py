from nicegui import ui, app
import os

text_box_container = ui.column().style('order: 2; width: 70%')
inputs = []

def update_text_boxes(e):
    global inputs
    count = int(e.value)
    text_box_container.clear()
    inputs.clear()

    with text_box_container:
        ui.label('Insert the TLE data for Satellites')
        ui.link('Get TLE\'s', target='https://orbit.ing-now.com/low-earth-orbit/', new_tab=True)
        for i in range(count):
            with ui.row().style('width:60%'):
                name = ui.input(label=f'Satellite {i + 1} Name').style('width: 30%')
                line1 = ui.input(label=f'Satellite {i + 1} TLE Line 1').style('width: 35%')
                line2 = ui.input(label=f'Satellite {i + 1} TLE Line 2').style('width: 35%')
                # Store the input references for later retrieval
                inputs.append([name, line1, line2])

def submit():
    tle_list = []
    for name, line1, line2 in inputs:
        n = name.value.strip() if name.value else ''
        l1 = line1.value.strip() if line1.value else ''
        l2 = line2.value.strip() if line2.value else ''
        if n and l1 and l2:
            tle_list.append([n, l1, l2])
    print(tle_list)  # This is your array of arrays
    #convert tle_list to czml 
    from satellite_czml import satellite_czml
    import time
    # Convert to CZML
    czml_string = satellite_czml(tle_list=tle_list).get_czml()
    #write this string to a file
    with open('sats.czml', 'w') as f:
        f.write(czml_string)
    #clear input
    inputs.clear()
    #navigate to the page to display
    ui.navigate.to('/Cesium_page')

ui.number(label='How many Satellites?', min=0, max=5, step=1, on_change=update_text_boxes).style('width: 10%')
ui.button('Submit', on_click=submit).style('order: 3;')

@ui.page('/Cesium_page')
def Cesium_page():
    import time
    html_directory = os.path.dirname(__file__)
    app.add_static_files('/static', html_directory)
    # Add a cache-busting query parameter using the file's modification time
    viewer_path = os.path.join(html_directory, 'viewer.html')
    mtime = int(os.path.getmtime(viewer_path)) if os.path.exists(viewer_path) else int(time.time())
    ui.html(
        f'''
        <div style="position: fixed; top: 0; right: 0; width: 70vw; height: 95vh; border: none; margin: 1vh 1vw 0 0; padding: 0; overflow: hidden; z-index: 999999; box-shadow: 0 0 10px rgba(0,0,0,0.2); background: #fff; border-radius: 12px;">
            <iframe style="width: 100%; height: 100%; border: none;" src="/static/viewer.html?cb={mtime}"></iframe>
        </div>
        '''
    )

ui.run()
