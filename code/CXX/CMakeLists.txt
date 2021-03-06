# CMakeLists.txt for SBGAT_core
# Benjamin Bercovici, 03/10/2017
# ORCCA
# University of Colorado 

get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(EXE_NAME ${dirName} CACHE STRING "Name of executable to be created.")

if (${CMAKE_MINOR_VERSION} EQUAL 0 AND NOT APPLE) # if true, running on a CCAR computer
	set(CMAKE_C_COMPILER /usr/local/gcc/bin/gcc CACHE STRING "C Compiler" FORCE)
	set(CMAKE_CXX_COMPILER /usr/local/gcc/bin/g++ CACHE STRING "C++ Compiler" FORCE)
else()
	message("This is not a CCAR computer")
endif()

# Check what version of the compiler is installed
execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion
                OUTPUT_VARIABLE GCC_VERSION)
message("GCC version is " ${GCC_VERSION})

if (GCC_VERSION VERSION_LESS 4.7) # if true, either running on a CCAR computer or a Mac (Clang has messed-up version IDs)
	if (NOT APPLE) # If NOT APPLE, then running on a CCAR computer
    	set(CMAKE_CXX_FLAGS "-std=c++0x")
    else() # Else running on a MAC. Will attempt to switch compiler to get the OMP-friendly GCC 6.3.0_1 from Homebrew
    	if(EXISTS /usr/local/Cellar/gcc/6.3.0_1/bin/) 
        	set(CMAKE_C_COMPILER /usr/local/Cellar/gcc/6.3.0_1/bin/gcc-6 CACHE STRING "C Compiler" FORCE)
			set(CMAKE_CXX_COMPILER /usr/local/Cellar/gcc/6.3.0_1/bin/g++-6 CACHE STRING "C++ Compiler" FORCE)
		endif()
    endif()
else() # Running on a proper Linux machine
  	set(CMAKE_CXX_FLAGS "-std=c++11")
endif()


project(${EXE_NAME})

# Specify the version used
if (${CMAKE_MAJOR_VERSION} LESS 3)
	message(FATAL_ERROR " You are running an outdated version of CMake")
endif()

cmake_minimum_required(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.0)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/source/cmake)

# Compiler flags
add_definitions(-Wall -O2)

# Include directories
include_directories(include)
include_directories(external/include)


# Find Boost
set(BOOST_ROOT /usr/local/Cellar/boost/1.63.0/)
find_package(Boost COMPONENTS filesystem system) 
if (Boost_FOUND)
	include_directories(${Boost_INCLUDE_DIRS}) 
else() # This corresponds to a personal setting. The first statement should be sufficient to find BOOST provided it is installed
	include_directories(/usr/local/Cellar/boost/1.60.0_2/include) 
	set(Boost_LIBRARY_DIRS "/usr/local/Cellar/boost/1.60.0_2/lib")
endif()

# Find Armadillo 
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

# Find OpenMP
# This is not mandatory. If not found, the pragmas will simply be ignored (may have to comment out #include "omp.h")
find_package(OpenMP)
if(OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

message("CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})

# Assimp
# find_package(assimp)
# include_directories(${ASSIMP_INCLUDE_DIRS})


# include_directories(~/include/assimp)
# if (APPLE)
# 	set(ASSIMP_LIBRARY  
# 		"/usr/local/lib/libassimp.3.3.1.dylib"
# 		"/usr/local/lib/libassimp.3.dylib"
# 		"/usr/local/lib/libassimp.dylib")
# elseif(UNIX AND NOT APPLE)
# 	set(ASSIMP_LIBRARY  
# 			"/usr/local/lib/libassimp.so.3.3.1"
# 			"/usr/local/lib/libassimp.so.3"
# 			"/usr/local/lib/libassimp.so")
# endif()

# Fortran compiler, required by Armadillo on Linux/Ubuntu
if(UNIX AND NOT APPLE AND ${CMAKE_MINOR_VERSION} GREATER 0)
	find_library(GFORTRAN_LIBRARY gfortran
	    PATHS /usr/lib/gcc/x86_64-linux-gnu/5/ /usr/lib/gcc/x86_64-redhat-linux/4.4.7/32/)
	list(APPEND ARMADILLO_LIBRARIES "${GFORTRAN_LIBRARY}")
endif()

# Add source files in root directory
add_executable(${EXE_NAME}
source/main.cpp
source/partB.cpp
include/partB.hpp
)



# Linking
set(library_dependencies
${ARMADILLO_LIBRARIES}
${Boost_LIBRARIES})
target_link_libraries(${EXE_NAME} ${library_dependencies})

