// Copyright (C) Signal Hound, Inc. (2020)

#ifndef TG_API_H
#define TG_API_H

#if defined(_WIN32)
    #ifdef TG_EXPORTS
        #define TG_API __declspec(dllexport)
    #else
        #define TG_API __declspec(dllimport)
    #endif
#else // Linux
    #define TG_API
#endif

#define TG_MAX_DEVICES (4)

#ifdef __cplusplus
extern "C" {
#endif

typedef enum tgDeviceType {
    tgDeviceTypeNone = 0,
    tgDeviceTypeTG44A = 1,
    tgDeviceTypeTG124A = 2    
} tgDeviceType;

typedef enum tgRef {
    tgRefUnused = 0,
    tgRefInternalOut = 1,
    tgRefExternalIn = 2    
} tgRef;

typedef enum tgStatus {
    tgFailedOperation            = -100,
    tgDeviceAlreadyOpen          = -6,
	tgReferenceSettingInvalidErr = -5,
	tgNullPtrErr                 = -4,
    tgDeviceInvalidErr		     = -3,
	tgDeviceNotConnectedErr      = -2,
	tgDeviceNotOpenErr           = -1,
    tgNoError                    = 0
} tgStatus;

TG_API tgStatus tgOpenDevice(int device);
// Initialize N devices (device 0 .. N-1) where N <= 4
TG_API tgStatus tgOpenAllDevices(int *numDevicesInitialized); 
TG_API tgStatus tgCloseDevice(int device);
TG_API tgStatus tgStatusCheck(int device);
// Call this after InitializeAll for each device
TG_API tgStatus tgGetSerialNumber(int device, int *serial); 
TG_API tgStatus tgGetDeviceType(int device, tgDeviceType *type);

TG_API tgStatus tgSetFreqAmp(int device, double freq, float ampl);
TG_API tgStatus tgSetReference(int device, tgRef ref);
TG_API tgStatus tgSetAttenuator(int device, float atten);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TG_API_H
