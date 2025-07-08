#!/bin/bash
python3 gen_iq.py "$1"
python3 GNU_Radio/GRC\ Files/data_to_vsg.py
