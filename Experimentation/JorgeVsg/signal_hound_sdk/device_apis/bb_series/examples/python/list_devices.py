# -*- coding: utf-8 -*-

# This example retrieves the list of currently connected BB60 devices,
# and displays their serial numbers and device types.
#
# Typically followed by bb_open_device_by_serial_number(serial).

from bbdevice.bb_api import *

def list_devices():
    # Get serial and device type lists
    result = bb_get_serial_number_list_2()

    # Display results
    print (f'Status: {result["status"]}')
    print (f'Count: {result["device_count"].value}')
    print (f'Serials: ', end='')
    serials = result["serials"]
    for serial in serials:
        print (f'{serial}, ', end='')
    print (f'\nTypes: ', end='')
    device_types = result["device_types"]
    for device_type in device_types:
        print (f'{device_type}, ', end='')

if __name__ == "__main__":
    list_devices()
