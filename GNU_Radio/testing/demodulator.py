import numpy as np

def read_qpsk(symbols):
        # print("Reading bits from symbols")
        bits = np.zeros((len(symbols), 2), dtype=int)
        for i in range(len(symbols)):
            angle = np.angle(symbols[i], deg=True) % 360

            # codex for the phases to bits
            if 0 <= angle < 90:
                bits[i] = [1, 1]  # 45°
            elif 90 <= angle < 180:
                bits[i] = [0, 1]  # 135°
            elif 180 <= angle < 270:
                bits[i] = [0, 0]  # 225°
            else:
                bits[i] = [1, 0]  # 315°
        
        # put into a single list
        best_bits = ''.join(str(b) for pair in bits for b in pair)
        return best_bits