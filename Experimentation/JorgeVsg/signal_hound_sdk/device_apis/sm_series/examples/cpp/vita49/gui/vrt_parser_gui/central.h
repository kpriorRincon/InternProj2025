#ifndef CENTRAL_H
#define CENTRAL_H

#include "vrt_parser.h"

#include <thread>
#include <mutex>
#include <deque>

#include <QWidget>
#include <QLineEdit>
#include <QPushButton>
#include <QCheckBox>
#include <QLabel>
#include <QGroupBox>
#include <QComboBox>
#include <QPainter>

struct HeaderLabels {
    HeaderLabels(QWidget *parent = 0) {
        packetType = new QLabel("", parent);
        packetCount = new QLabel("", parent);
        packetSize = new QLabel("", parent);
    }

    ~HeaderLabels() {
        if(packetType) delete packetType;
        if(packetCount) delete packetCount;
        if(packetSize) delete packetSize;
    }

    QLabel *packetType;
    QLabel *packetCount;
    QLabel *packetSize;
};

struct MetadataLabels {
    MetadataLabels(QWidget *parent = 0) {
        headerLabels = new HeaderLabels(parent);
        streamIdent = new QLabel("", parent);
        seconds = new QLabel("", parent);
        picoUpper = new QLabel("", parent);
        picoLower = new QLabel("", parent);
    }

    ~MetadataLabels() {
        if(headerLabels) delete headerLabels;
        if(streamIdent) delete streamIdent;
        if(seconds) delete seconds;
        if(picoUpper) delete picoUpper;
        if(picoLower) delete picoLower;
    }

    HeaderLabels *headerLabels;
    QLabel *streamIdent;
    QLabel *seconds;
    QLabel *picoUpper;
    QLabel *picoLower;
};

struct ContextPacketLabels {
    ContextPacketLabels(QWidget *parent = 0) {
        metadataLabels = new MetadataLabels(parent);
        fieldChanged = new QLabel("", parent);
        bandwidth = new QLabel("", parent);
        rfFreq = new QLabel("", parent);
        reflevel = new QLabel("", parent);
        atten = new QLabel("", parent);
        sampleRate = new QLabel("", parent);
        temperature = new QLabel("", parent);
        devUid = new QLabel("", parent);
        devModel = new QLabel("", parent);
    }

    ~ContextPacketLabels() {
        if(metadataLabels) delete metadataLabels;
        if(fieldChanged) delete fieldChanged;
        if(bandwidth) delete bandwidth;
        if(rfFreq) delete rfFreq;
        if(reflevel) delete reflevel;
        if(atten) delete atten;
        if(sampleRate) delete sampleRate;
        if(temperature) delete temperature;
        if(devUid) delete devUid;
        if(devModel) delete devModel;
    }

    MetadataLabels *metadataLabels;
    QLabel *fieldChanged;
    QLabel *bandwidth;
    QLabel *rfFreq;
    QLabel *reflevel;
    QLabel *atten;
    QLabel *sampleRate;
    QLabel *temperature;
    QLabel *devUid;
    QLabel *devModel;
};

struct TrailerLabels {
    TrailerLabels(QWidget *parent = 0) {
        isCalibratedTime = new QLabel("", parent);
        isValidData = new QLabel("", parent);
        isReferenceLock = new QLabel("", parent);
        isOverRange = new QLabel("", parent);
        isSampleLoss = new QLabel("", parent);
    }

    ~TrailerLabels() {
        if(isCalibratedTime) delete isCalibratedTime;
        if(isValidData) delete isValidData;
        if(isReferenceLock) delete isReferenceLock;
        if(isOverRange) delete isOverRange;
        if(isSampleLoss) delete isSampleLoss;
    }

    QLabel *isCalibratedTime;
    QLabel *isValidData;
    QLabel *isReferenceLock;
    QLabel *isOverRange;
    QLabel *isSampleLoss;
};

struct DataPacketLabels {
    DataPacketLabels(QWidget *parent = 0) {
        metadataLabels = new MetadataLabels(parent);
        trailerLabels = new TrailerLabels(parent);
    }

    ~DataPacketLabels() {
        if(metadataLabels) delete metadataLabels;
        if(trailerLabels) delete trailerLabels;
    }

    MetadataLabels *metadataLabels;
    TrailerLabels *trailerLabels;
};

class Central : public QWidget
{
    Q_OBJECT
public:
    explicit Central(QWidget *parent = 0);
    ~Central();

    void GetPacketSizes();
    void GetContextPacketSize();
    void GetDataPacketSize();

    int ParseBlob(VRTParser &parser, uint32_t *words, uint32_t wordCount,
                  std::deque<VRTUserContextPkt> &parsedContextPackets,
                  std::deque<VRTUserDataPkt> &parsedDataPackets);
    int ParsePacket(VRTParser &parser, uint32_t *pkt,
                    std::deque<VRTUserContextPkt> &parsedContextPackets,
                    std::deque<VRTUserDataPkt> &parsedDataPackets);

signals:
    void packetParsed();

public slots:

private slots:
    void connectUSBDevice();
    void connectNetworkedDevice();
    void configureDevice();

    void setSamplesPerPacket();

    void getContextPacket();
    void getDataPacket();
    void getBlob();

    void streamOrStop();
    void startStreaming();
    void stopStreaming();

    void streamDataPacketsInBackground();
    void updateDataPacketLabels();

    void contextIndexChanged(int index);
    void dataIndexChanged(int index);

private:
    int device;
    double reflevel;

    bool streaming;
    VRTUserDataPkt parsed;
    std::thread streamingThread;
    std::mutex parsedPacketLock;

    VRTParser parser;

    QPushButton *connectUSBBtn;
    QPushButton *connectNetworkedBtn;

    QLineEdit *centerEntry;
    QLineEdit *decimationEntry;
    QLineEdit *bandwidthEntry;
    QLineEdit *reflevelEntry;
    QCheckBox *purgeBeforeAcquire;

    QLineEdit *samplesPerPacketEntry;
    QLineEdit *contextPacketsPerBlobEntry;
    QLineEdit *dataPacketsPerBlobEntry;

    // Context packet
    QComboBox *contextSelect;
    QLabel *contextPktSizeLabel;
    ContextPacketLabels *contextPacketLabels;

    // Data packet
    QComboBox *dataSelect;
    QPushButton *streamBtn;
    QLabel *samplesPerPacketLabel;
    QLabel *dataPktSizeLabel;
    DataPacketLabels *dataPacketLabels;
    QLabel *peakFreqLabel;
    QLabel *peakAmpLabel;

    QDialog *dataDialog;

    QGroupBox *settingsBox;
    QGroupBox *captureBox;
    QGroupBox *contextBox;
    QGroupBox *dataBox;

    std::deque<VRTUserContextPkt> parsedContextPackets;
    std::deque<VRTUserDataPkt> parsedDataPackets;
};

#endif // CENTRAL_H
