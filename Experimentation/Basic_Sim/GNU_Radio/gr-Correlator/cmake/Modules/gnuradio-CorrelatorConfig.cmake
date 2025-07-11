find_package(PkgConfig)

PKG_CHECK_MODULES(PC_GR_CORRELATOR gnuradio-Correlator)

FIND_PATH(
    GR_CORRELATOR_INCLUDE_DIRS
    NAMES gnuradio/Correlator/api.h
    HINTS $ENV{CORRELATOR_DIR}/include
        ${PC_CORRELATOR_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    GR_CORRELATOR_LIBRARIES
    NAMES gnuradio-Correlator
    HINTS $ENV{CORRELATOR_DIR}/lib
        ${PC_CORRELATOR_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/gnuradio-CorrelatorTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(GR_CORRELATOR DEFAULT_MSG GR_CORRELATOR_LIBRARIES GR_CORRELATOR_INCLUDE_DIRS)
MARK_AS_ADVANCED(GR_CORRELATOR_LIBRARIES GR_CORRELATOR_INCLUDE_DIRS)
