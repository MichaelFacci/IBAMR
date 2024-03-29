PROJECT(mysolver)
CMAKE_MINIMUM_REQUIRED(VERSION 3.15.0)

ADD_EXECUTABLE(main2d example.cpp)

FIND_PACKAGE(IBAMR 0.11.0 REQUIRED)
TARGET_LINK_LIBRARIES(main2d IBAMR::IBAMR2d)
# IBAMR saves the flags it used to compile - you can reuse them if you want to
SET(CMAKE_CXX_FLAGS ${IBAMR_CXX_FLAGS})
