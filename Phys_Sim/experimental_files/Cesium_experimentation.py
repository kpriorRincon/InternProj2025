# run: npm install cesium
from nicegui import ui
cesium_html ='''
<div id="cesiumContainer" style="width:100%; height:100vh;"></div>

<script src="https://cesium.com/downloads/cesiumjs/releases/1.114/Build/Cesium/Cesium.js"></script>
<link href="https://cesium.com/downloads/cesiumjs/releases/1.114/Build/Cesium/Widgets/widgets.css" rel="stylesheet">

<script>
  // Set up the Cesium viewer
  window.CESIUM_BASE_URL = "https://cesium.com/downloads/cesiumjs/releases/1.114/Build/Cesium/";
  const viewer = new Cesium.Viewer('cesiumContainer', {
    terrainProvider: Cesium.createWorldTerrain(),
    timeline: false,
    animation: false
  });

  viewer.camera.setView({
    destination: Cesium.Cartesian3.fromDegrees(-122.4175, 37.655, 4000.0)
  });
</script>
'''
ui.html(cesium_html)
ui.run()