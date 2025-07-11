# Install script for directory: /home/trevor/Documents/InternProj2025/GNU_Radio/gr-customModule/python/customModule

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

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python3.12/dist-packages/gnuradio/customModule" TYPE FILE FILES
    "/home/trevor/Documents/InternProj2025/GNU_Radio/gr-customModule/python/customModule/__init__.py"
    "/home/trevor/Documents/InternProj2025/GNU_Radio/gr-customModule/python/customModule/QPSK_Modulator.py"
    "/home/trevor/Documents/InternProj2025/GNU_Radio/gr-customModule/python/customModule/QPSK_Demodulator.py"
    "/home/trevor/Documents/InternProj2025/GNU_Radio/gr-customModule/python/customModule/upsampler.py"
    "/home/trevor/Documents/InternProj2025/GNU_Radio/gr-customModule/python/customModule/downsampler.py"
    "/home/trevor/Documents/InternProj2025/GNU_Radio/gr-customModule/python/customModule/vec_to_var.py"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python3.12/dist-packages/gnuradio/customModule" TYPE FILE FILES
    "/home/trevor/Documents/InternProj2025/GNU_Radio/gr-customModule/build/python/customModule/__init__.pyc"
    "/home/trevor/Documents/InternProj2025/GNU_Radio/gr-customModule/build/python/customModule/QPSK_Modulator.pyc"
    "/home/trevor/Documents/InternProj2025/GNU_Radio/gr-customModule/build/python/customModule/QPSK_Demodulator.pyc"
    "/home/trevor/Documents/InternProj2025/GNU_Radio/gr-customModule/build/python/customModule/upsampler.pyc"
    "/home/trevor/Documents/InternProj2025/GNU_Radio/gr-customModule/build/python/customModule/downsampler.pyc"
    "/home/trevor/Documents/InternProj2025/GNU_Radio/gr-customModule/build/python/customModule/vec_to_var.pyc"
    "/home/trevor/Documents/InternProj2025/GNU_Radio/gr-customModule/build/python/customModule/__init__.pyo"
    "/home/trevor/Documents/InternProj2025/GNU_Radio/gr-customModule/build/python/customModule/QPSK_Modulator.pyo"
    "/home/trevor/Documents/InternProj2025/GNU_Radio/gr-customModule/build/python/customModule/QPSK_Demodulator.pyo"
    "/home/trevor/Documents/InternProj2025/GNU_Radio/gr-customModule/build/python/customModule/upsampler.pyo"
    "/home/trevor/Documents/InternProj2025/GNU_Radio/gr-customModule/build/python/customModule/downsampler.pyo"
    "/home/trevor/Documents/InternProj2025/GNU_Radio/gr-customModule/build/python/customModule/vec_to_var.pyo"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/trevor/Documents/InternProj2025/GNU_Radio/gr-customModule/build/python/customModule/bindings/cmake_install.cmake")

endif()

