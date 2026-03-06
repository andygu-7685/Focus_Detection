cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(FocusAlgo)

#------------------------------------------------------------------------------------------

include(GNUInstallDirs)
if(MSVC)
    add_compile_options(/W4)
    add_link_options(/W4)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin)
    # message("compiler: MSVC")
else()
    # message("compiler: NOT MSVC")
    # add_compile_options(-Wall -Wextra -pedantic -Werror -Wl,--fatal-warnings)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()


# place binaries and libraries according to GNU standards
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})


if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
endif()

#------------------------------------------------------------------------------------------

# build libraries as static in excutable
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)


include(FetchContent)


set(BUILD_LIST "core,imgproc,imgcodecs" CACHE STRING "" FORCE)

# 2. Disable heavy 3rd-party dependencies to speed up the build
set(WITH_TIFF OFF CACHE BOOL "" FORCE)
set(WITH_OPENEXR OFF CACHE BOOL "" FORCE)
set(WITH_JASPER OFF CACHE BOOL "" FORCE)
set(WITH_WEBP OFF CACHE BOOL "" FORCE)
set(WITH_OPENCL OFF CACHE BOOL "" FORCE)
set(WITH_IPP OFF CACHE BOOL "" FORCE)
set(WITH_ITT OFF CACHE BOOL "" FORCE)
set(WITH_QUIRC OFF CACHE BOOL "" FORCE)

# 1. Set options to skip the "heavy" parts of OpenCV during the build
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_PERF_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_python2 OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_python3 OFF CACHE BOOL "" FORCE)
set(BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_apps OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_dnn OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_stitching OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_video OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_videoio OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_gapi OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_ml OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_objdetect OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_photo OFF CACHE BOOL "" FORCE)

# 2. Declare OpenCV
FetchContent_Declare(opencv
    GIT_REPOSITORY https://github.com/opencv/opencv.git
    GIT_TAG 4.10.0  # It's better to use a specific version tag than 'master'
)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v3.0.2  # The latest stable version as of early 2026
)

# 3. Make it available
FetchContent_MakeAvailable(opencv pybind11)




#------------------------------------------------------------------------------------------
#   Grab all cpp files from includes folder
#------------------------------------------------------------------------------------------

set(INCLUDES_FOLDER includes)
FILE(GLOB_RECURSE SOURCE_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}   "${INCLUDES_FOLDER}/*.cpp" )

#------------------------------------------------------------------------------------------

# ADD_EXECUTABLE(main
#     main.cpp
#     ${SOURCE_FILES}
# )


# Link the core modules. 
# 'opencv_core' is mandatory; 'opencv_imgproc' and 'opencv_highgui' are for images/windows.
# target_link_libraries(main PRIVATE 
#     opencv_core 
#     opencv_imgproc 
#     opencv_highgui
#     opencv_imgcodecs
# )

# # This ensures the compiler can find the OpenCV headers
# target_include_directories(main PRIVATE 
#     ${CMAKE_BINARY_DIR}
#     ${opencv_SOURCE_DIR}/modules/core/include
#     ${opencv_INCLUDE_DIR}
#     ${opencv_SOURCE_DIR}/include
#     ${opencv_SOURCE_DIR}/modules/imgproc/include
#     ${opencv_SOURCE_DIR}/modules/imgcodecs/include
#     ${opencv_SOURCE_DIR}/modules/highgui/include
#     ${opencv_BINARY_DIR}
# )


pybind11_add_module(my_module main.cpp)

set_target_properties(my_module PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# 4. Link everything to BOTH targets
# Linking to 'opencv_core' automatically sets up the include paths for you!
target_link_libraries(my_module PRIVATE     
    opencv_core 
    opencv_imgproc 
    #opencv_highgui
    opencv_imgcodecs
    pybind11::module
)

# Apply the same include directories to the python module
target_include_directories(my_module PRIVATE 
    ${CMAKE_BINARY_DIR}
    ${pybind11_INCLUDE_DIRS}
    ${opencv_SOURCE_DIR}/modules/core/include
    ${opencv_INCLUDE_DIR}
    ${opencv_SOURCE_DIR}/include
    ${opencv_SOURCE_DIR}/modules/imgproc/include
    ${opencv_SOURCE_DIR}/modules/imgcodecs/include
    #${opencv_SOURCE_DIR}/modules/highgui/include
    ${opencv_BINARY_DIR}
)



#set min c++ standard
#TARGET_COMPILE_FEATURES(main PRIVATE cxx_std_17)
# install to bin folder
#INSTALL(TARGETS main)

#------------------------------------------------------------------------------------------
