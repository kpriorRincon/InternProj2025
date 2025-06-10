import pickle

with open('Cesium_Sattelite/sattelite_tles.txt') as f:
    names = []
    line1s = []
    line2s = []
    lines = f.readlines()
    for i in range(0, len(lines), 3):
        if i < len(lines)-3:
            names.append(lines[i].strip())
            line1s.append(lines[i+1].strip())
            line2s.append(lines[i+2].strip())

#create a pickle file that contains all of the arrays
with open('sattelite_tles.pkl', 'wb') as pf:
    pickle.dump({'names': names, 'line1s': line1s, 'line2s': line2s}, pf)