QT += core
QT -= gui

CONFIG += c++11

TARGET = Clasificador_Nuevo
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

INCLUDEPATH += C:/opencv/build/include
CONFIG(release,debug|release){
LIBS += C:/opencv/build/x86/vc12/lib/opencv_calib3d2413.lib \
C:/opencv/build/x86/vc12/lib/opencv_contrib2413.lib \
C:/opencv/build/x86/vc12/lib/opencv_core2413.lib \
C:/opencv/build/x86/vc12/lib/opencv_features2d2413.lib \
C:/opencv/build/x86/vc12/lib/opencv_flann2413.lib \
C:/opencv/build/x86/vc12/lib/opencv_gpu2413.lib \
C:/opencv/build/x86/vc12/lib/opencv_highgui2413.lib \
C:/opencv/build/x86/vc12/lib/opencv_imgproc2413.lib \
C:/opencv/build/x86/vc12/lib/opencv_legacy2413.lib \
C:/opencv/build/x86/vc12/lib/opencv_ml2413.lib \
C:/opencv/build/x86/vc12/lib/opencv_objdetect2413.lib \
C:/opencv/build/x86/vc12/lib/opencv_ts2413.lib \
C:/opencv/build/x86/vc12/lib/opencv_video2413.lib \
}
CONFIG(debug,debug|release){
LIBS += C:/opencv/build/x86/vc12/lib/opencv_calib3d2413d.lib \
C:/opencv/build/x86/vc12/lib/opencv_contrib2413d.lib \
C:/opencv/build/x86/vc12/lib/opencv_core2413d.lib \
C:/opencv/build/x86/vc12/lib/opencv_features2d2413d.lib \
C:/opencv/build/x86/vc12/lib/opencv_flann2413d.lib \
C:/opencv/build/x86/vc12/lib/opencv_gpu2413d.lib \
C:/opencv/build/x86/vc12/lib/opencv_highgui2413d.lib \
C:/opencv/build/x86/vc12/lib/opencv_imgproc2413d.lib \
C:/opencv/build/x86/vc12/lib/opencv_legacy2413d.lib \
C:/opencv/build/x86/vc12/lib/opencv_ml2413d.lib \
C:/opencv/build/x86/vc12/lib/opencv_objdetect2413d.lib \
C:/opencv/build/x86/vc12/lib/opencv_ts2413.lib \
C:/opencv/build/x86/vc12/lib/opencv_video2413d.lib \
}

SOURCES += main.cpp
