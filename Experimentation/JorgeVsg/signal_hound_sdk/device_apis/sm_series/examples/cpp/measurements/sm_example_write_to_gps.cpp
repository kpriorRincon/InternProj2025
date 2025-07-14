#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <Windows.h>

#include "sm_api.h"

struct UBXMsg {
    uint8_t hdr[2];
    // Class
    uint8_t cls; 
    uint8_t id;
    // Payload length inferred from vector size
    std::vector<uint8_t> payload;
    uint8_t ckA;
    uint8_t ckB;
};

static void UBXChecksum(const uint8_t *mem, int len, uint8_t &chk_a, uint8_t &chk_b)
{
    chk_a = 0;
    chk_b = 0;

    for(int i = 0; i < len; i++) {
        chk_a += mem[i];
        chk_b += chk_a;
    }
}

// Convert UBXMsg to a binary vector to be sent to the GPS
// The checksums do not have to be populated, although when the function returns,
//   the checks will be populated.
static std::vector<uint8_t> UBXBuildMessage(UBXMsg &ubx)
{
    std::vector<uint8_t> msg(8 + ubx.payload.size());

    msg[0] = ubx.hdr[0];
    msg[1] = ubx.hdr[1];
    msg[2] = ubx.cls;
    msg[3] = ubx.id;
    *((uint16_t*)&msg[4]) = ubx.payload.size();
    if(ubx.payload.size() > 0) {
        memcpy(&msg[6], &ubx.payload[0], ubx.payload.size());
    }

    UBXChecksum(&msg[2], 4 + ubx.payload.size(), ubx.ckA, ubx.ckB);

    msg[msg.size() - 2] = ubx.ckA;
    msg[msg.size() - 1] = ubx.ckB;

    return msg;
}

// Look for UBX messages in a nmea response
// Return all UBX messages found
static std::vector<UBXMsg> UBXParse(const uint8_t *nmea, int len)
{
    std::vector<UBXMsg> msgList;
    UBXMsg msg;

    // Need at least 8 bytes to parse a full, but empty payload, message. 
    for(int i = 0; i < len - 8;) {
        if(nmea[i] == 0xB5 && nmea[i+1] == 0x62) {
            // Found UBX message
            msg.hdr[0] = nmea[i];
            msg.hdr[1] = nmea[i+1];
            msg.cls = nmea[i+2];
            msg.id = nmea[i+3];

            // Get length
            uint16_t payloadLen;
            memcpy(&payloadLen, nmea + i + 4, 2);

            msg.payload.resize(payloadLen);

            // If we can't read the payload becuase there are not enough bytes,
            //   we know we are at the end of the data, return.
            if((i + 8 + payloadLen) > len) {
                //assert(false);
                return msgList;
            }

            if(payloadLen > 0) {
                memcpy(&msg.payload[0], nmea + i + 6, payloadLen);
            }

            msg.ckA = nmea[i + 6 + payloadLen];
            msg.ckB = nmea[i + 6 + payloadLen + 1];

            msgList.push_back(msg);

            i += 8 + msg.payload.size();
        } else {
            i++;
        }
    }

    return msgList;
}

static void UBXPrintMsg(UBXMsg msg)
{
    // Print UBX message info
    printf("UBX Message\n");
    printf("%x %x %x %x\n", msg.hdr[0], msg.hdr[1], msg.cls, msg.id);
    printf("Payload len: %d\n", msg.payload.size());
    printf("Payload: ");
    for(int i = 0; i < msg.payload.size(); i++) {
        printf("%x ", msg.payload[i]);
    }
    printf("\n\n");
}

void sm_example_write_to_gps()
{
    int handle;
    SmStatus status = smOpenDevice(&handle);
    if(status != smNoError) {
        printf("Error opening device\n");
        return;
    }

    // Build nav engine settings GET command
    UBXMsg msg;
    msg.hdr[0] = 0xB5;
    msg.hdr[1] = 0x62;
    msg.cls = 0x06;
    msg.id = 0x24;
    // Empty payload for GET request.
    std::vector<uint8_t> msgBin = UBXBuildMessage(msg);

    // Stores the nmea response from the SM200
    uint8_t nmea[1024];
    int nmealen;

    // Send get message
    smWriteToGPS(handle, msgBin.data(), msgBin.size());

    // Whether or not we have found the get response
    bool found = false;

    // Iterate until we get the GET response or ~1.5 seconds have passed.
    // If the GPS antenna is connected and the GPS is locked, the UBX message
    //   might not appear until the next PPS trigger, which could be 1 second away.
    int iter = 0;
    while(iter++ < 15) {
        // Prevents busy loop, no need to request that often
        Sleep(100);

        // Query the latest GPS/NMEA info
        // Any responses come in through the NMEA strings.
        SmBool updated;
        nmealen = 1024;
        smGetGPSInfo(handle, smTrue, &updated, 0, 0, 0, 0, (char*)nmea, &nmealen);

        std::vector<UBXMsg> msgList;

        // Grab all UBX messages, only need to parse the nmea data when the API reports
        //   that the data has been updated.
        if(updated == smTrue) {
            msgList = UBXParse(nmea, nmealen);
        }

        // Look for get response for nav engine configure
        for(auto &m : msgList) {
            if(m.cls == 0x06 && m.id == 0x24) {
                found = true;
                msg = m;
                break;
            }
        }

        if(found) {
            break;
        }
    }

    if(!found) {
        // Didn't find GET response.
        // In very rare circumstances the message might span two NMEA requests. This code
        //   does not handle that situation. In that event, the recommended solution is to
        //   send another GET request and query the NMEA data again.
        exit(-1);
    }

    // Let's print the response
    // This will print the NAV5 payload that is currently being used.
    UBXPrintMsg(msg);

    // Set the dymamic platform model to portable
    msg.payload[2] = 0;
    // Build message now as set message
    msgBin = UBXBuildMessage(msg);
    // Send set message
    smWriteToGPS(handle, msgBin.data(), msgBin.size());

    // At this point, re-running the program would yield the changed NAV engine settings
    //   when printing the retrieved settings. Verify this by rerunning the program.

    smCloseDevice(handle);
}
