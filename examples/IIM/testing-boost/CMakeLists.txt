PROJECT(mysolver)
CMAKE_MINIMUM_REQUIRED(VERSION 3.15.0)

ADD_EXECUTABLE(boost2d testingboost.cpp)
ADD_EXECUTABLE(boost3d testingboost.cpp)

FIND_PACKAGE(IBAMR 0.11.0 REQUIRED)
TARGET_LINK_LIBRARIES(boost2d IBAMR::IBAMR2d)
TARGET_LINK_LIBRARIES(boost3d IBAMR::IBAMR3d)
# IBAMR saves the flags it used to compile - you can reuse them if you want to
SET(CMAKE_CXX_FLAGS ${IBAMR_CXX_FLAGS})

