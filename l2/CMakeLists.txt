cmake_minimum_required (VERSION 2.6)
project (hnsw_lib)

include_directories("${PROJECT_BINARY_DIR}")



set(SOURCE_EXE main.cpp)            

set(SOURCE_LIB peos.cpp)

add_library(peos STATIC ${SOURCE_LIB})


add_executable(PEOs ${SOURCE_EXE})
SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )
target_link_libraries(PEOs peos) 
