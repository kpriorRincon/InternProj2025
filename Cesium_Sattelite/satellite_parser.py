# https://celestrak.org/NORAD/elements/
# go to this site so that you can get TLE current data to put into sattelite_tles.txt

import pickle

with open('sattelite_tles.txt') as f:
    names = []
    line1s = []
    line2s = []
    lines = f.readlines()
    for i in range(0, len(lines), 3):
        if i < len(lines)-3:
            names.append(lines[i].strip())
            line1s.append(lines[i+1].strip())
            line2s.append(lines[i+2].strip())

# print(len(line1s))
# print(len(names))
# print(len(line2s))

#create a pickle file that contains all of the arrays
with open('sattelite_tles.pkl', 'wb') as pf:
    pickle.dump({'names': names, 'line1s': line1s, 'line2s': line2s}, pf)