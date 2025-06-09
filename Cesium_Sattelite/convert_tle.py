#we would like to convert to czml in order to use the path in cesium
#patched tle2czml file from pygeoif.geometry import as_shape as asShape
from satellite_czml import satellite_czml
with open('2024-001A.txt', 'r') as file:
    tle_lines = [line.strip() for line in file]

#debug lines
print(tle_lines)
czml_string = satellite_czml(tle_list=tle_lines).getczml()