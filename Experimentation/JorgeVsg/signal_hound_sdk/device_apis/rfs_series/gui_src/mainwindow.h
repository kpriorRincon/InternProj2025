#pragma once

#include "rfs_device.h"

#include <QMainWindow>
#include <QPushButton>
#include <QLineEdit>
#include <QDialog>
#include <QComboBox>
#include <QLabel>
#include <QButtonGroup>

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    void LoadPreset();
    void SavePreset();

    void SetConnectActionsEnabled(bool enabled);
    void SetPortButtonsEnabled(bool enabled);

    void ButtonStyleRefresh();

private:
    RFSDevice rfsDevice;

    QComboBox *comPortSelect;
    QPushButton *enumerateBtn;
    QPushButton *connectBtn;
    QLabel *connectionLabel;
    QButtonGroup *buttonGroup;
    std::vector<QPushButton*> portButtons;

private slots:
    void handleConnectBtn();
    void connectDevice();
    void disconnectDevice();

    void populateCOMPortSelect();
    void portButtonPressed(QAbstractButton*);

    void updateConnectionState();
    void updateActivePortState();

    void showFailedToConnectAlert();
    void showDisconnectAlert();
    void showInvalidPortAlert();
};
