#!/bin/bash
python3 /home/empire/Documents/InternProj2025/Final_Product/transmitter/gen_iq.py $1
python3 /home/empire/Documents/InternProj2025/Final_Product/transmitter/GNU_Radio/GRC\ Files/data_to_vsg.py
