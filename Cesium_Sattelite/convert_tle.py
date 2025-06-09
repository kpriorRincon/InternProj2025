from satellite_czml import satellite_czml

# Read TLE file
with open('2024-001A.txt', 'r') as file:
    tle_line = [line.strip() for line in file if line.strip() != '']
    #the input to satellite_czml() must be a list of list
tle_lines = [tle_line]
print(type(tle_lines))
# Debug output
print(tle_lines)

# Convert to CZML
czml_string = satellite_czml(tle_list=tle_lines).get_czml()
#write this string to a file
with open('xposat.czml', 'w') as f:
    f.write(czml_string)
