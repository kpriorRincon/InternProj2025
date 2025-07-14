# Install script for directory: /home/trevorwiseman/Documents/GitHub/InternProj2025/GNU_Radio/gr-Correlator/python/Correlator

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python3.10/dist-packages/gnuradio/Correlator" TYPE FILE FILES
    "/home/trevorwiseman/Documents/GitHub/InternProj2025/GNU_Radio/gr-Correlator/python/Correlator/__init__.py"
    "/home/trevorwiseman/Documents/GitHub/InternProj2025/GNU_Radio/gr-Correlator/python/Correlator/correlator.py"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python3.10/dist-packages/gnuradio/Correlator" TYPE FILE FILES
    "/home/trevorwiseman/Documents/GitHub/InternProj2025/GNU_Radio/gr-Correlator/build/python/Correlator/__init__.pyc"
    "/home/trevorwiseman/Documents/GitHub/InternProj2025/GNU_Radio/gr-Correlator/build/python/Correlator/correlator.pyc"
    "/home/trevorwiseman/Documents/GitHub/InternProj2025/GNU_Radio/gr-Correlator/build/python/Correlator/__init__.pyo"
    "/home/trevorwiseman/Documents/GitHub/InternProj2025/GNU_Radio/gr-Correlator/build/python/Correlator/correlator.pyo"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/trevorwiseman/Documents/GitHub/InternProj2025/GNU_Radio/gr-Correlator/build/python/Correlator/bindings/cmake_install.cmake")

endif()

