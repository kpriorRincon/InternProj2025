#include <stdio.h>
#include "bb_api.h"

int main(int argc, char **argv)
{
  bbStatus status;
  int handle = -1;
  int serials[8] = {0};
  int count = 0;

  printf("\nAPI Version: %s\n", bbGetAPIVersion());

  bbGetSerialNumberList(serials, &count);
  if(count < 1) { 
    printf("No BB60 devices found\n");
    return 0;
  }

  printf("Attempting to open device with serial number %d\n",
	 serials[0]);
  
  status = bbOpenDeviceBySerialNumber(&handle, serials[0]);
  //status = bbOpenDevice(&handle);

  if(status != bbNoError) {
    // Unable to open device
    printf("Unable to open device\n");
    printf("%s\n", bbGetErrorString(status));
    return 0;
  } else {
    printf("Device opened successfully\n");
  }

  printf("\n-- Device Info --\n");
  int type = 0;
  bbGetDeviceType(handle, &type);
  if(type == BB_DEVICE_BB60A) {
    printf("Device Type: BB60A\n");
  } else if(type == BB_DEVICE_BB60C) {
    printf("Device Type: BB60C\n");    
  }

  int firmware_ver = 0;
  bbGetFirmwareVersion(handle, &firmware_ver);
  printf("Firmware Version: %d\n", firmware_ver);
  
  float temp, voltage, current;
  bbGetDeviceDiagnostics(handle, &temp, &voltage, &current);
  printf("Device Internal Temp: %f C\n", temp);
  printf("Device Voltage: %f V\n", voltage);
  printf("Device Current Draw: %f mA\n", current);

  bbCloseDevice(handle);

  return 0;
}
