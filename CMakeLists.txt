cmake_minimum_required(VERSION 3.5)

# Source path:
set(SOURCE_DIR ${CMAKE_HOME_DIRECTORY}/src)
# List of all files in src with name ending in '.cpp':
file(GLOB LIB_SOURCES ${SOURCE_DIR}/lib/*.cpp)
message( "CMAKE_CURRENT_SOURCE_DIR  " ${CMAKE_CURRENT_SOURCE_DIR} )
message( "SOURCE_DIR                " ${SOURCE_DIR} )


# TARGETS
# static library containing everything in src
#     first argument is the name of the target
#     second argument is the type of the library (STATIC of SHARED)
#     third argument is a list of source files
add_library(followmelib STATIC ${LIB_SOURCES})


set( CMAKE_CXX_STANDARD 14 )
set( CMAKE_CXX_STANDARD_REQUIRED TRUE )
SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

project( demo_tracking )

# list(APPEND CMAKE_MODULE_PATH ${CMAKE_HOME_DIRECTORY}/FindTBB.cmake)

# find_package(TBB COMPONENTS tbbmalloc tbbmalloc_proxy tbb_preview)

find_package( OpenCV REQUIRED )
find_package( realsense2 REQUIRED )
find_package( InferenceEngine REQUIRED)
# find_package( PCL 1.8 REQUIRED )
# find_package( OpenGL REQUIRED)

# find_package( PkgConfig REQUIRED)
# pkg_search_module(GLFW REQUIRED glfw3)

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${PROJECT_SOURCE_DIR} )
include_directories( ${InferenceEngine_INCLUDE_DIRS} )

add_executable( demo_tracking ${SOURCE_DIR}/main.cpp )
target_link_libraries( demo_tracking followmelib)
target_link_libraries( demo_tracking ${OpenCV_LIBS} )
target_link_libraries( demo_tracking ${realsense2_LIBRARY})
