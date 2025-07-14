#-------------------------------------------------
#
# Project created by QtCreator 2018-03-09T13:39:35
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = VRTParser
TEMPLATE = app


SOURCES += main.cpp\
    mainwindow.cpp \
    central.cpp \
    vrt_parser.cpp \
    sh_vrt.cpp

HEADERS  += mainwindow.h \
    central.h \
    vrt_parser.h \
    sh_vrt.h

unix {
    LIBS += -L/usr/local/lib/
    LIBS += -lsm_api
}
