#include <cstdio>
#include <cstdlib>

#include "sm_api.h"

// Sleep routine
#if defined(_WIN32)
#include <Windows.h>
#else
#include <unistd.h>
inline void Sleep(int millis) {
    usleep(millis * 1000);
}
#endif

static void checkStatus(SmStatus status)
{
    if(status > 0) { // Warning
        printf("Warning: %s\n", smGetErrorString(status));
        return;
    } else if(status < 0) { // Error
        printf("Error: %s\n", smGetErrorString(status));
        exit(-1);
    }
}

// GPS antenna should be connected before running this test.
// Will wait up to 3 minutes for the API to report GPS lock
// Prints off GPS information

void sm_example_get_gps_info()
{
    int handle = -1;
    SmStatus status = smNoError;

    // Uncomment this to open a USB SM device
    status = smOpenDevice(&handle);
    // Uncomment this to open a networked SM device with a default network config
    //status = smOpenNetworkedDevice(&handle, SM_ADDR_ANY, SM_DEFAULT_ADDR, SM_DEFAULT_PORT);

    checkStatus(status);

    printf("Waiting 3 minutes for GPS lock\n");

    // Wait 3 minutes for GPS lock
    SmGPSState gpsState;
    int iter = 0;
    do {
        smGetGPSState(handle, &gpsState);
        Sleep(100);
    } while(gpsState == smGPSStateNotPresent && iter++ < 1800);

    // Read GPS Info
    if(gpsState == smGPSStateNotPresent) {
        printf("Unable to find GPS\n");
    } else {
        printf("Found GPS\n");

        // Now lets query any GPS information and print it
        SmBool updated = smFalse;
        int64_t secSinceEpoch;
        double latitude = 0.0, longitude = 0.0, altitude = 0.0;
        int nmeaLen = 4096;
        char *nmea = new char[nmeaLen];
        smGetGPSInfo(handle, smTrue, &updated, &secSinceEpoch,
            &latitude, &longitude, &altitude, nmea, &nmeaLen);

        printf("Seconds since epoch: %lld\n", secSinceEpoch);
        printf("Latitude: %f\n", latitude);
        printf("Longitude: %f\n", longitude);
        printf("Altitude: %f\n", altitude);
        printf("%s\n", nmea);

        delete [] nmea;
    }
}
