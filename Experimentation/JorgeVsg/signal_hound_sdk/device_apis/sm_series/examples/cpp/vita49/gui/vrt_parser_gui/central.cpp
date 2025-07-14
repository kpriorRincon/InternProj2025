#include "central.h"

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

#include <cmath>
#include <limits>

#include <QFormLayout>
#include <QVBoxLayout>
#include <QDialog>

#include <QDebug>

#define CENTER      3e9
#define DECIMATION  2
#define BANDWIDTH   20e6
#define REFLEVEL    -20
#define PURGE       smFalse

#define SAMPLES_PER_PACKET          32768
#define CONTEXT_PACKETS_PER_BLOB    10
#define DATA_PACKETS_PER_BLOB       100
#define PACKET_COUNT                1

Central::Central(QWidget *parent) :
    QWidget(parent),
    device(-1),
    streaming(false)
{
    QVBoxLayout *overallBox = new QVBoxLayout;

    connectUSBBtn = new QPushButton("Connect USB SM Series Device", this);
    connect(connectUSBBtn, SIGNAL(clicked()), this, SLOT(connectUSBDevice()));
    overallBox->addWidget(connectUSBBtn);

    connectNetworkedBtn = new QPushButton("Connect Networked SM Series Device", this);
    connect(connectNetworkedBtn, SIGNAL(clicked()), this, SLOT(connectNetworkedDevice()));
    overallBox->addWidget(connectNetworkedBtn);

    // Settings
    settingsBox = new QGroupBox(tr("Settings"));
    settingsBox->setAlignment(Qt::AlignHCenter);
    settingsBox->setEnabled(false);
    QVBoxLayout *settingsBoxLayout = new QVBoxLayout;

    centerEntry = new QLineEdit(this);
    centerEntry->setText(QString::number(CENTER));
//    connect(centerEntry, SIGNAL(editingFinished()), this, SLOT(configureDevice()));

    decimationEntry = new QLineEdit(this);
    decimationEntry->setText(QString::number(DECIMATION));
//    connect(decimationEntry, SIGNAL(editingFinished()), this, SLOT(configureDevice()));

    bandwidthEntry = new QLineEdit(this);
    bandwidthEntry->setText(QString::number(BANDWIDTH));
//    connect(bandwidthEntry, SIGNAL(editingFinished()), this, SLOT(configureDevice()));

    reflevelEntry = new QLineEdit(this);
    reflevelEntry->setText(QString::number(REFLEVEL));
//    connect(reflevelEntry, SIGNAL(editingFinished()), this, SLOT(configureDevice()));

    purgeBeforeAcquire = new QCheckBox(this);
    purgeBeforeAcquire->setChecked(PURGE);

    samplesPerPacketEntry = new QLineEdit(this);
    samplesPerPacketEntry->setText(QString::number(SAMPLES_PER_PACKET));
    connect(samplesPerPacketEntry, SIGNAL(editingFinished()), this, SLOT(setSamplesPerPacket()));

    contextPacketsPerBlobEntry = new QLineEdit(this);
    contextPacketsPerBlobEntry->setText(QString::number(CONTEXT_PACKETS_PER_BLOB));

    dataPacketsPerBlobEntry = new QLineEdit(this);
    dataPacketsPerBlobEntry->setText(QString::number(DATA_PACKETS_PER_BLOB));

    QFormLayout *settingsForm = new QFormLayout;
    settingsForm->addRow(tr("&Center Frequency:"), centerEntry);
    settingsForm->addRow(tr("&Decimation:"), decimationEntry);
    settingsForm->addRow(tr("&Bandwidth:"), bandwidthEntry);
    settingsForm->addRow(tr("&Reference Level:"), reflevelEntry);
    settingsForm->addRow(tr("&Purge Before Acquire:"), purgeBeforeAcquire);
    settingsForm->addRow(tr("&Samples/Packet (128, 65535 - 6):"), samplesPerPacketEntry);
    settingsForm->addRow(tr("&Context Packets/Blob:"), contextPacketsPerBlobEntry);
    settingsForm->addRow(tr("&Data Packets/Blob:"), dataPacketsPerBlobEntry);

    settingsBoxLayout->addLayout(settingsForm);

    QPushButton *configureBtn = new QPushButton("Configure", this);
    connect(configureBtn, SIGNAL(clicked()), this, SLOT(configureDevice()));
    settingsBoxLayout->addWidget(configureBtn);

    settingsBox->setLayout(settingsBoxLayout);
    overallBox->addWidget(settingsBox);

    // Capture
    captureBox = new QGroupBox(tr("Capture"));
    captureBox->setAlignment(Qt::AlignHCenter);
    QVBoxLayout *captureBoxLayout = new QVBoxLayout;
    captureBox->setLayout(captureBoxLayout);

    QPushButton *captureBtn = new QPushButton("Get Blob", this);
    connect(captureBtn, SIGNAL(clicked()), this, SLOT(getBlob()));

    captureBoxLayout->addWidget(captureBtn);

    captureBox->setEnabled(false);
    overallBox->addWidget(captureBox);
    settingsBoxLayout->addWidget(configureBtn);

    // Context and Data Box
    QWidget *packetsContainer = new QWidget(this);
    QHBoxLayout *packetsLayout = new QHBoxLayout;
    packetsContainer->setLayout(packetsLayout);

    // Context Packet
    contextBox = new QGroupBox(tr("Context Packets"));
    contextBox->setAlignment(Qt::AlignHCenter);
    QVBoxLayout *contextBoxLayout = new QVBoxLayout;

    contextSelect = new QComboBox(this);
    connect(contextSelect, SIGNAL(currentIndexChanged(int)), this, SLOT(contextIndexChanged(int)));

    contextPktSizeLabel = new QLabel(this);
    contextPacketLabels = new ContextPacketLabels(this);

    QFormLayout *contextForm = new QFormLayout;
    contextForm->addRow(tr("&Packet Number:"), contextSelect);
    contextForm->addRow(tr("&Packet Size (words):"), contextPktSizeLabel);
    // Header
    contextForm->addRow(tr("&packetType:"), contextPacketLabels->metadataLabels->headerLabels->packetType);
    contextForm->addRow(tr("&packetCount:"), contextPacketLabels->metadataLabels->headerLabels->packetCount);
    contextForm->addRow(tr("&packetSize:"), contextPacketLabels->metadataLabels->headerLabels->packetSize);
    // Meta
    contextForm->addRow(tr("streamIdent:"), contextPacketLabels->metadataLabels->streamIdent);
    contextForm->addRow(tr("seconds:"), contextPacketLabels->metadataLabels->seconds);
    contextForm->addRow(tr("picoUpper:"), contextPacketLabels->metadataLabels->picoUpper);
    contextForm->addRow(tr("picoLower:"), contextPacketLabels->metadataLabels->picoLower);
    // Fields
    contextForm->addRow(tr("fieldChanged:"), contextPacketLabels->fieldChanged);
    contextForm->addRow(tr("bandwidth:"), contextPacketLabels->bandwidth);
    contextForm->addRow(tr("rfFreq:"), contextPacketLabels->rfFreq);
    contextForm->addRow(tr("reflevel:"), contextPacketLabels->reflevel);
    contextForm->addRow(tr("atten:"), contextPacketLabels->atten);
    contextForm->addRow(tr("sampleRate:"), contextPacketLabels->sampleRate);
    contextForm->addRow(tr("temperature:"), contextPacketLabels->temperature);
    contextForm->addRow(tr("devUid:"), contextPacketLabels->devUid);
    contextForm->addRow(tr("devModel:"), contextPacketLabels->devModel);
    contextBoxLayout->addLayout(contextForm);

    QPushButton *getContextPacketBtn = new QPushButton("Get Context Packet", this);
    connect(getContextPacketBtn, SIGNAL(clicked()), this, SLOT(getContextPacket()));
    contextBoxLayout->addWidget(getContextPacketBtn);

    contextBox->setEnabled(false);
    contextBox->setLayout(contextBoxLayout);
    packetsLayout->addWidget(contextBox);

    // Data Packets
    dataBox = new QGroupBox(tr("Data Packets"));
    dataBox->setAlignment(Qt::AlignHCenter);
    QVBoxLayout *dataBoxLayout = new QVBoxLayout;

    dataSelect = new QComboBox(this);
    connect(dataSelect, SIGNAL(currentIndexChanged(int)), this, SLOT(dataIndexChanged(int)));

    samplesPerPacketLabel = new QLabel(this);
    dataPktSizeLabel = new QLabel(this);
    dataPacketLabels = new DataPacketLabels(this);
    peakFreqLabel = new QLabel(this);
    peakAmpLabel = new QLabel(this);

    QFormLayout *dataForm = new QFormLayout;
    dataForm->addRow(tr("&Packet Number:"), dataSelect);
    dataForm->addRow(tr("&Samples/Packet:"), samplesPerPacketLabel);
    dataForm->addRow(tr("&Packet Size (words):"), dataPktSizeLabel);
    // Header
    dataForm->addRow(tr("&packetType:"), dataPacketLabels->metadataLabels->headerLabels->packetType);
    dataForm->addRow(tr("&packetCount:"), dataPacketLabels->metadataLabels->headerLabels->packetCount);
    dataForm->addRow(tr("&packetSize:"), dataPacketLabels->metadataLabels->headerLabels->packetSize);
    // Meta
    dataForm->addRow(tr("streamIdent:"), dataPacketLabels->metadataLabels->streamIdent);
    dataForm->addRow(tr("seconds:"), dataPacketLabels->metadataLabels->seconds);
    dataForm->addRow(tr("picoUpper:"), dataPacketLabels->metadataLabels->picoUpper);
    dataForm->addRow(tr("picoLower:"), dataPacketLabels->metadataLabels->picoLower);
    // Trailer
    dataForm->addRow(tr("isCalibratedTime:"), dataPacketLabels->trailerLabels->isCalibratedTime);
    dataForm->addRow(tr("isValidData:"), dataPacketLabels->trailerLabels->isValidData);
    dataForm->addRow(tr("isReferenceLock:"), dataPacketLabels->trailerLabels->isReferenceLock);
    dataForm->addRow(tr("isOverRange:"), dataPacketLabels->trailerLabels->isOverRange);
    dataForm->addRow(tr("isSampleLoss:"), dataPacketLabels->trailerLabels->isSampleLoss);
    // Data stats
    dataForm->addRow(tr("Peak Amplitude:"), peakAmpLabel);

    dataBoxLayout->addLayout(dataForm);

    QPushButton *getDataPacketBtn = new QPushButton("Get Data Packet", this);
    connect(getDataPacketBtn, SIGNAL(clicked()), this, SLOT(getDataPacket()));
    dataBoxLayout->addWidget(getDataPacketBtn);

    streamBtn = new QPushButton("Stream Data Packets", this);
    connect(streamBtn, SIGNAL(clicked()), this, SLOT(streamOrStop()));
    dataBoxLayout->addWidget(streamBtn);

    dataBox->setEnabled(false);
    dataBox->setLayout(dataBoxLayout);
    packetsLayout->addWidget(dataBox);

    overallBox->addWidget(packetsContainer);
    setLayout(overallBox);

    connect(this, SIGNAL(packetParsed()), this, SLOT(updateDataPacketLabels()));
}

Central::~Central() {
    smCloseDevice(device);

    if(contextPacketLabels) delete contextPacketLabels;
    if(dataPacketLabels) delete dataPacketLabels;
}

void Central::GetPacketSizes()
{
    GetContextPacketSize();
    GetDataPacketSize();
}

void Central::GetContextPacketSize()
{
    uint32_t wordCount = 0;

    SmStatus status = smGetVrtContextPktSize(device, &wordCount);
    if(status != smNoError) {
        contextBox->setEnabled(false);
        dataBox->setEnabled(false);
        qDebug() << "Could not get context packet size.";
        return;
    }

    contextPktSizeLabel->setText(QString::number(wordCount));
}

void Central::GetDataPacketSize()
{
    uint16_t samplesPerPacketActual = 0;
    uint32_t wordCount = 0;

    SmStatus status = smGetVrtPacketSize(device, &samplesPerPacketActual, &wordCount);
    if(status != smNoError) {
        contextBox->setEnabled(false);
        dataBox->setEnabled(false);
        qDebug() << "Could not get data packet size.";
        return;
    }

    samplesPerPacketLabel->setText(QString::number(samplesPerPacketActual));
    dataPktSizeLabel->setText(QString::number(wordCount));
}

int Central::ParseBlob(VRTParser &parser, uint32_t *words, uint32_t wordCount,
                       std::deque<VRTUserContextPkt> &parsedContextPackets,
                       std::deque<VRTUserDataPkt> &parsedDataPackets)
{
    uint32_t *curr = words;
    while(curr - words < wordCount) {
        curr += ParsePacket(parser, curr, parsedContextPackets, parsedDataPackets);
    }
    return curr - words;
}

int Central::ParsePacket(VRTParser &parser, uint32_t *pkt,
                         std::deque<VRTUserContextPkt> &parsedContextPackets,
                         std::deque<VRTUserDataPkt> &parsedDataPackets)
{
    SmVRTPacketType packetType;
    uint32_t packetSize;
    parser.Peek(pkt, &packetType, &packetSize);

    switch(packetType) {
    case smVRTDataPacket:
        parsedDataPackets.push_back(VRTUserDataPkt());
        return parser.ParseDataPacket(pkt, packetSize, parsedDataPackets.back());
    case smVRTContextPacket:
        parsedContextPackets.push_back(VRTUserContextPkt());
        return parser.ParseContextPacket(pkt, packetSize, parsedContextPackets.back());
    default:
        // Memory pointed to is not a valid SM200A VRT packet
        return -1;
    }
}

void Central::connectUSBDevice()
{
    SmStatus status = smOpenDevice(&device);
    if(status != smNoError) {
        qDebug() << "Could not open USB SM Series device.";
        return;
    }

    connectUSBBtn->setEnabled(false);
    connectNetworkedBtn->setEnabled(false);
    settingsBox->setEnabled(true);
}

void Central::connectNetworkedDevice()
{
    SmStatus status = smOpenNetworkedDevice(&device, SM_ADDR_ANY, SM_DEFAULT_ADDR, SM_DEFAULT_PORT);
    if(status != smNoError) {
        qDebug() << "Could not open networked SM Series device.";
        return;
    }

    connectUSBBtn->setEnabled(false);
    connectNetworkedBtn->setEnabled(false);
    settingsBox->setEnabled(true);
}

void Central::configureDevice()
{
    double center = centerEntry->text().toDouble();
    int decimation = decimationEntry->text().toInt();
    double bandwidth = bandwidthEntry->text().toDouble();
    double reflevel = reflevelEntry->text().toDouble();

    smSetIQCenterFreq(device, center);
    smSetIQSampleRate(device, decimation);
    smSetIQBandwidth(device, smTrue, bandwidth);
    smSetRefLevel(device, reflevel);

    SmStatus status = smConfigure(device, smModeIQStreaming); // VRT mode
    if(status != smNoError) {
        contextBox->setEnabled(false);
        dataBox->setEnabled(false);
        qDebug() << "Could not open SM200A.";
        return;
    }

    setSamplesPerPacket();
    parser.reflevel = reflevel;

    smSetVrtStreamID(device, 13);

    captureBox->setEnabled(true);
    contextBox->setEnabled(true);
    dataBox->setEnabled(true);
}

void Central::setSamplesPerPacket()
{
    uint16_t samplesPerPacket = samplesPerPacketEntry->text().toUInt();
    smSetVrtPacketSize(device, samplesPerPacket);
    GetPacketSizes();
}

void Central::getContextPacket()
{
    uint32_t wordCount = 0;
    SmStatus status = smGetVrtContextPktSize(device, &wordCount);
    if(status != smNoError) {
        qDebug() << "Could not get context packet size.";
        return;
    }

    uint32_t *words = new uint32_t[wordCount];
    uint32_t actualWordCount;
    status = smGetVrtContextPkt(device, words, &actualWordCount);
    if(status != smNoError) {
        qDebug() << "Could not get context packet.";
        return;
    }
    VRTUserContextPkt parsed;
    parser.ParseContextPacket(words, wordCount, parsed);

    parsedContextPackets.push_back(parsed);
    contextSelect->addItem(QString::number(parsedContextPackets.size()));
    contextSelect->setCurrentIndex(parsedContextPackets.size()-1);

    // Header
    contextPacketLabels->metadataLabels->headerLabels->packetType->setText(QString::number(parsed.prologue.header.packetType));
    contextPacketLabels->metadataLabels->headerLabels->packetCount->setText(QString::number(parsed.prologue.header.packetCount));
    contextPacketLabels->metadataLabels->headerLabels->packetSize->setText(QString::number(parsed.prologue.header.packetSize));
    // Metadata
    contextPacketLabels->metadataLabels->streamIdent->setText(QString::number(parsed.prologue.streamIdent));
    contextPacketLabels->metadataLabels->seconds->setText(QString::number(parsed.prologue.seconds));
    contextPacketLabels->metadataLabels->picoUpper->setText(QString::number(parsed.prologue.picoUpper));
    contextPacketLabels->metadataLabels->picoLower->setText(QString::number(parsed.prologue.picoLower));
    // Fields
    contextPacketLabels->fieldChanged->setText(QString::number(parsed.fieldChanged));
    contextPacketLabels->bandwidth->setText(QString::number(parsed.bandwidth));
    contextPacketLabels->rfFreq->setText(QString::number(parsed.rfFreq));
    contextPacketLabels->reflevel->setText(QString::number(parsed.reflevel));
    contextPacketLabels->atten->setText(QString::number(parsed.atten));
    contextPacketLabels->sampleRate->setText(QString::number(parsed.sampleRate));
    contextPacketLabels->temperature->setText(QString::number(parsed.temperature));
    contextPacketLabels->devUid->setText(QString::number(parsed.devUid));
    contextPacketLabels->devModel->setText(QString::number(parsed.devModel));

    reflevel = parsed.reflevel;

    if(words) delete[] words;
}

void Central::getDataPacket()
{
    int samplesPerPacket = samplesPerPacketEntry->text().toInt();

    uint32_t wordCount = 0;
    uint16_t samplesPerPacketReturn = 0;
    SmStatus status = smGetVrtPacketSize(device, &samplesPerPacketReturn, &wordCount);
    if(status != smNoError) {
        qDebug() << "Could not get data packet size.";
        return;
    }

    uint32_t *words = new uint32_t[wordCount];
    uint32_t actualWordCount;
    status = smGetVrtPackets(device, words, &actualWordCount, PACKET_COUNT, PURGE);
    if(status != smNoError) {
        qDebug() << "Could not get data packets.";
        return;
    }

    VRTUserDataPkt parsedLocal;
    parser.ParseDataPacket(words, wordCount, parsedLocal);

    parsedPacketLock.lock();
    parsedDataPackets.push_back(parsedLocal);
    parsedPacketLock.unlock();

    emit packetParsed();

    if(words) delete[] words;
}

void Central::getBlob()
{
    int contextPacketCount = contextPacketsPerBlobEntry->text().toInt();
    int dataPacketCount = dataPacketsPerBlobEntry->text().toInt();

    int samplesPerPacket = samplesPerPacketEntry->text().toInt();

    // Get context size
    uint32_t contextWordCount = 0;
    SmStatus status = smGetVrtContextPktSize(device, &contextWordCount);
    if(status != smNoError) {
        qDebug() << "Could not get context packet size.";
        return;
    }

    // Get data size
    uint32_t dataWordCount = 0;
    uint16_t samplesPerPacketReturn = 0;
    status = smGetVrtPacketSize(device, &samplesPerPacketReturn, &dataWordCount);
    if(status != smNoError) {
        qDebug() << "Could not get data packet size.";
        return;
    }

    uint32_t wordCount = contextWordCount * contextPacketCount +
                         dataWordCount * dataPacketCount;
    uint32_t *words = new uint32_t[wordCount];
    uint32_t *curr = words;

    int dataPacketsPerContextPacket = dataPacketCount / contextPacketCount;

    for(int i = 0; i < contextPacketCount; i++) {
        // Get context packet
        uint32_t actualContextWordCount;
        status = smGetVrtContextPkt(device, curr, &actualContextWordCount);
        if(status != smNoError) {
            qDebug() << "Could not get context packet.";
            return;
        }
        if(actualContextWordCount != contextWordCount) {
            qDebug() << "Context packet is not the expected size.";
            return;
        }
        curr += contextWordCount;

        // Get data packets
        uint32_t actualDataWordCount;
        status = smGetVrtPackets(device, curr, &actualDataWordCount, dataPacketsPerContextPacket, PURGE);
        if(status != smNoError) {
            qDebug() << "Could not get data packets.";
            return;
        }
        if(actualDataWordCount != dataWordCount * dataPacketsPerContextPacket) {
            qDebug() << "Data packet is not the expected size.";
            return;
        }
        curr += actualDataWordCount;
    }

    // Send to parser
    parsedContextPackets.clear();
    parsedDataPackets.clear();
    ParseBlob(parser, words, wordCount, parsedContextPackets, parsedDataPackets);

    contextSelect->clear();
    for(int i = 0; i < parsedContextPackets.size(); i++) {
        contextSelect->addItem(QString::number(i+1));
    }

    dataSelect->clear();
    for(int i = 0; i < parsedDataPackets.size(); i++) {
        dataSelect->addItem(QString::number(i+1));
    }

    if(words) delete[] words;
}

void Central::streamOrStop()
{
    if(streaming) {
        stopStreaming();
    } else {
        startStreaming();
    }
}

void Central::startStreaming()
{
    streaming = true;
    streamBtn->setText("Stop Streaming");
    streamingThread = std::thread(&Central::streamDataPacketsInBackground, this);
}

void Central::stopStreaming()
{
    streaming = false;
    streamBtn->setText("Stream Data Packets");
    if(streamingThread.joinable()) streamingThread.join();
}

void Central::streamDataPacketsInBackground()
{
    while(streaming) {
        getDataPacket();
#ifdef _WIN32
        Sleep(100);
#else
        usleep(100e3);
#endif

    }
}

void Central::updateDataPacketLabels()
{
    parsedPacketLock.lock();
    dataSelect->addItem(QString::number(parsedDataPackets.size()));
    parsedPacketLock.unlock();

    dataSelect->setCurrentIndex(parsedDataPackets.size()-1);
}

void Central::contextIndexChanged(int index)
{
    if(index < 0 || index > parsedContextPackets.size()-1) {
        return;
    }
    VRTUserContextPkt parsedLocal = parsedContextPackets[index];

    // Header
    contextPacketLabels->metadataLabels->headerLabels->packetType->setText(QString::number(parsedLocal.prologue.header.packetType));
    contextPacketLabels->metadataLabels->headerLabels->packetCount->setText(QString::number(parsedLocal.prologue.header.packetCount));
    contextPacketLabels->metadataLabels->headerLabels->packetSize->setText(QString::number(parsedLocal.prologue.header.packetSize));
    // Metadata
    contextPacketLabels->metadataLabels->streamIdent->setText(QString::number(parsedLocal.prologue.streamIdent));
    contextPacketLabels->metadataLabels->seconds->setText(QString::number(parsedLocal.prologue.seconds));
    contextPacketLabels->metadataLabels->picoUpper->setText(QString::number(parsedLocal.prologue.picoUpper));
    contextPacketLabels->metadataLabels->picoLower->setText(QString::number(parsedLocal.prologue.picoLower));
    // Fields
    contextPacketLabels->fieldChanged->setText(QString::number(parsedLocal.fieldChanged));
    contextPacketLabels->bandwidth->setText(QString::number(parsedLocal.bandwidth));
    contextPacketLabels->rfFreq->setText(QString::number(parsedLocal.rfFreq));
    contextPacketLabels->reflevel->setText(QString::number(parsedLocal.reflevel));
    contextPacketLabels->atten->setText(QString::number(parsedLocal.atten));
    contextPacketLabels->sampleRate->setText(QString::number(parsedLocal.sampleRate));
    contextPacketLabels->temperature->setText(QString::number(parsedLocal.temperature));
    contextPacketLabels->devUid->setText(QString::number(parsedLocal.devUid));
    contextPacketLabels->devModel->setText(QString::number(parsedLocal.devModel));

    reflevel = parsedLocal.reflevel;
}

void Central::dataIndexChanged(int index)
{
    if(index < 0 || index > parsedDataPackets.size()-1) {
        return;
    }

    VRTUserDataPkt parsedLocal = parsedDataPackets[index];

    // Header
    dataPacketLabels->metadataLabels->headerLabels->packetType->setText(QString::number(parsedLocal.prologue.header.packetType));
    dataPacketLabels->metadataLabels->headerLabels->packetCount->setText(QString::number(parsedLocal.prologue.header.packetCount));
    dataPacketLabels->metadataLabels->headerLabels->packetSize->setText(QString::number(parsedLocal.prologue.header.packetSize));
    // Metadata
    dataPacketLabels->metadataLabels->streamIdent->setText(QString::number(parsedLocal.prologue.streamIdent));
    dataPacketLabels->metadataLabels->seconds->setText(QString::number(parsedLocal.prologue.seconds));
    dataPacketLabels->metadataLabels->picoUpper->setText(QString::number(parsedLocal.prologue.picoUpper));
    dataPacketLabels->metadataLabels->picoLower->setText(QString::number(parsedLocal.prologue.picoLower));

    // Trailer
    if(parsedLocal.trailer.isCalibratedTime.enabled) {
        dataPacketLabels->trailerLabels->isCalibratedTime->setText(
                    QString::number(parsedLocal.trailer.isCalibratedTime.indicator));
    } else dataPacketLabels->trailerLabels->isCalibratedTime->setText("");
    if(parsedLocal.trailer.isValidData.enabled) {
        dataPacketLabels->trailerLabels->isValidData->setText(
                    QString::number(parsedLocal.trailer.isValidData.indicator));
    } else dataPacketLabels->trailerLabels->isValidData->setText("");
    if(parsedLocal.trailer.isReferenceLock.enabled) {
        dataPacketLabels->trailerLabels->isReferenceLock->setText(
                    QString::number(parsedLocal.trailer.isReferenceLock.indicator));
    } else dataPacketLabels->trailerLabels->isReferenceLock->setText("");
    if(parsedLocal.trailer.isOverRange.enabled) {
        dataPacketLabels->trailerLabels->isOverRange->setText(
                    QString::number(parsedLocal.trailer.isOverRange.indicator));
    } else dataPacketLabels->trailerLabels->isOverRange->setText("");
    if(parsedLocal.trailer.isSampleLoss.enabled) {
        dataPacketLabels->trailerLabels->isSampleLoss->setText(
                    QString::number(parsedLocal.trailer.isSampleLoss.indicator));
    } else dataPacketLabels->trailerLabels->isSampleLoss->setText("");

    // Data
    double peakFreq = 0;
    double peakAmp = -1e3;
    for(int i = 0; i < parsedLocal.data.size()-1; i += 2) {
        double amp = 10.0 * log10(parsedLocal.data[i] * parsedLocal.data[i] + parsedLocal.data[i+1] * parsedLocal.data[i+1]);
        if(amp > peakAmp) peakAmp = amp;
    }
    peakAmpLabel->setText(QString::number(peakAmp));
}
