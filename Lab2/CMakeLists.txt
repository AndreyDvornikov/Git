cmake_minimum_required(VERSION 3.16)
project(test_blas LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenBLAS REQUIRED)
find_package(OpenMP)

add_executable(test_blas test_blas.cpp)
target_include_directories(test_blas PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(TARGET OpenBLAS::OpenBLAS)
    target_link_libraries(test_blas PRIVATE OpenBLAS::OpenBLAS)
else()
    target_include_directories(test_blas PRIVATE ${OpenBLAS_INCLUDE_DIRS})
    target_link_libraries(test_blas PRIVATE ${OpenBLAS_LIBRARIES})
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(test_blas PRIVATE OpenMP::OpenMP_CXX)
endif()

message(STATUS "OpenBLAS found: ${OpenBLAS_DIR}")
message(STATUS "OpenMP enabled: ${OpenMP_CXX_FOUND}")
