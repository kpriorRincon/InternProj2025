#include "mainwindow.h"

#include <QVBoxLayout>
#include <QVariant>
#include <QStyle>
#include <QMenuBar>
#include <QIntValidator>
#include <QFormLayout>
#include <QSettings>
#include <QStandardPaths>
#include <QDir>
#include <QGroupBox>
#include <QMessageBox>
#include <QSerialPortInfo>

QString GetSettingsFilename()
{
    QString path = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation);
    path += "/SignalHound/";
    QDir().mkdir(path);
    path += "rfs/";
    QDir().mkdir(path);

    return path + "preferences.ini";
}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    setWindowTitle("Signal Hound RFS RF Switch");
    setMinimumWidth(400);

    // Connecting
    comPortSelect = new QComboBox();

    enumerateBtn = new QPushButton("Enumerate COM Devices");
    connect(enumerateBtn, SIGNAL(clicked()), this, SLOT(populateCOMPortSelect()));

    QFormLayout *loSerialPort = new QFormLayout();
    loSerialPort->addRow("COM Device", comPortSelect);

    connectBtn = new QPushButton("Connect");
    connect(connectBtn, SIGNAL(clicked()), this, SLOT(handleConnectBtn()));

    connectionLabel = new QLabel();
    connectionLabel->setAlignment(Qt::AlignCenter);

    QGroupBox *connectionBox = new QGroupBox();
    QVBoxLayout *loConnectionBox = new QVBoxLayout();
    loConnectionBox->addLayout(loSerialPort);
    loConnectionBox->addWidget(enumerateBtn);
    loConnectionBox->addWidget(connectBtn);
    loConnectionBox->addWidget(connectionLabel);
    connectionBox->setLayout(loConnectionBox);

    // Port buttons
    QVBoxLayout *loButtons = new QVBoxLayout();
    buttonGroup = new QButtonGroup();
    buttonGroup->setExclusive(true);
    connect(buttonGroup, SIGNAL(buttonClicked(QAbstractButton*)), this, SLOT(portButtonPressed(QAbstractButton*)));

    portButtons.resize(MAX_RF_PORTS);
    for(int i = 0; i < portButtons.size(); i++) {
        QPushButton *btn = new QPushButton(QString().sprintf("Port %d", i + 1));

        portButtons[i] = btn;
        loButtons->addWidget(btn);
        buttonGroup->addButton(btn, i + 1);

        btn->setStyleSheet("QPushButton {"
                             "background-color: rgb(220, 220, 220);"
                             "border-width: 1px;"
                             "border-radius: 5px;"
                             "min-width: 10em;"
                             "min-height: 2em;"
                             "padding: 1px;"
                           "}"

                           "QPushButton:hover {"
                             "border-style: solid;"
                             "border-color: rgb(170, 170, 170);"
                           "}"

                           "QPushButton:pressed {"
                             "background-color: rgb(200, 200, 200);"
                           "}"

                           "QPushButton[Active=true] {"
                             "background-color: rgb(172, 243, 174);"
                           "}");
    }

    // Layout
    QVBoxLayout *loMain = new QVBoxLayout();
    loMain->addWidget(connectionBox);
    loMain->addLayout(loButtons);

    QWidget *centralWidget = new QWidget();
    centralWidget->setLayout(loMain);

    setCentralWidget(centralWidget);

    // State
    LoadPreset();
    updateConnectionState();
    populateCOMPortSelect();

    // Connections
    connect(&rfsDevice, SIGNAL(connectionStateChanged()), this, SLOT(updateConnectionState()));
    connect(&rfsDevice, SIGNAL(portChanged()), this, SLOT(updateActivePortState()));

    connect(&rfsDevice, SIGNAL(failedToConnect()), this, SLOT(showFailedToConnectAlert()));
    connect(&rfsDevice, SIGNAL(unexpectedDisconnect()), this, SLOT(updateConnectionState()));
    connect(&rfsDevice, SIGNAL(unexpectedDisconnect()), this, SLOT(showDisconnectAlert()));
    connect(&rfsDevice, SIGNAL(invalidPort()), this, SLOT(showInvalidPortAlert()));
}

MainWindow::~MainWindow()
{
    SavePreset();

    rfsDevice.blockSignals(true);
}

void MainWindow::LoadPreset()
{
    QSettings s(GetSettingsFilename(), QSettings::IniFormat);

    int port = s.value("SerialPort", 1).toInt();

    for(int i = 0; i < comPortSelect->count(); i++) {
        if(comPortSelect->itemData(i).toInt() == port) {
            comPortSelect->setCurrentIndex(i);
        }
    }
}

void MainWindow::SavePreset()
{
    QSettings s(GetSettingsFilename(), QSettings::IniFormat);

    s.setValue("SerialPort", comPortSelect->currentData().toInt());
}

void MainWindow::SetPortButtonsEnabled(bool enabled)
{
    for(int i = 0; i < portButtons.size(); i++) {
        portButtons[i]->setEnabled(false);
//        portButtons[i]->hide();
    }

    for(int i = 0; i < rfsDevice.Ports(); i++) {
        portButtons[i]->setEnabled(enabled);
//        portButtons[i]->show();
    }
}

void MainWindow::ButtonStyleRefresh()
{
    // Style must be refreshed when a CSS property changes
    for(int i = 0; i < portButtons.size(); i++) {
        QPushButton *btn = portButtons[i];

        btn->style()->unpolish(btn);
        btn->style()->polish(btn);
        btn->update();
    }
}

void MainWindow::handleConnectBtn()
{
    rfsDevice.IsConnected() ? disconnectDevice()
                            : connectDevice();
}

void MainWindow::connectDevice()
{
    rfsDevice.Open(comPortSelect->currentData().toInt());
}

void MainWindow::disconnectDevice()
{
    rfsDevice.Close();
}

void MainWindow::populateCOMPortSelect()
{
    comPortSelect->clear();

    for(QSerialPortInfo portInfo : QSerialPortInfo::availablePorts()) {
        QString comPortStr = portInfo.portName() + ": "
                           + portInfo.description() + " | "
                           + portInfo.manufacturer();
        int portNum = portInfo.portName().replace("COM", "").toInt();
        comPortSelect->addItem(comPortStr, portNum);
    }

    connectBtn->setEnabled(true);
    comPortSelect->setEnabled(true);
}

void MainWindow::portButtonPressed(QAbstractButton *btn)
{
    rfsDevice.SetPort(buttonGroup->id(btn));
}

void MainWindow::updateConnectionState()
{
    bool connected = rfsDevice.IsConnected();

    if(connected) {
        connectionLabel->setText(QString().sprintf("%s\nSerial Number: %d",
                                                   rfsDevice.DeviceName().toStdString().c_str(),
                                                   rfsDevice.SerialNumber()));
        connectionLabel->setStyleSheet("color: green");
        connectBtn->setText("Disconnect");

    } else {
        connectionLabel->setText("No device connected");
        connectionLabel->setStyleSheet("color: red");
        connectBtn->setText("Connect");
    }

    updateActivePortState();

    SetPortButtonsEnabled(connected);
}

void MainWindow::updateActivePortState()
{
    int activePort = rfsDevice.ActivePort();
    int activePortButtonIdx = activePort - 1;

    // Set active property for CSS
    for(int i = 0; i < portButtons.size(); i++) {
        portButtons[i]->setProperty("Active", false);
    }

    if(rfsDevice.IsConnected()) {
        portButtons[activePortButtonIdx]->setProperty("Active", true);
    }

    ButtonStyleRefresh();
}

void MainWindow::showFailedToConnectAlert()
{
    QMessageBox::warning(this, "Connection Failure",
                         "Device failed to connect.\n\n"
                         "Make sure the correct serial port is selected.");
}

void MainWindow::showDisconnectAlert()
{
    QMessageBox::warning(this, "No Connection", "Device has disconnected unexpectedly.");
}

void MainWindow::showInvalidPortAlert()
{
    QMessageBox::warning(this, "Invalid Port",
                         QString().sprintf("Selected port is not valid.\n\n"
                                           "Please choose a port between 1 and %d.", rfsDevice.Ports()));
}
