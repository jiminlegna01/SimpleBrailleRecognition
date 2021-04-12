TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

INCLUDEPATH += D:\sdk\opencv\install\include
LIBS += -LD:\sdk\opencv\install\x64\mingw\bin \
        -lopencv_world452

SOURCES += \
        main.cpp
