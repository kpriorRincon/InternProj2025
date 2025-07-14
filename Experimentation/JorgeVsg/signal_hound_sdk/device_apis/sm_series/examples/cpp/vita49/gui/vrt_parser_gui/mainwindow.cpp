#include "mainwindow.h"

#include "central.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    setWindowTitle("Signal Hound SM Series VRT Parser");
    resize(500, 600);
    move(500, 50);

    Central *central = new Central(this);
    setCentralWidget(central);
}

MainWindow::~MainWindow()
{

}
