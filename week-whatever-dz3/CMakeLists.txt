cmake_minimum_required(VERSION 3.15)
project(week_whatever_dz3)

set(CMAKE_CXX_STANDARD 17)

add_executable(week_whatever_dz3 main.cpp)

find_package(Python3 COMPONENTS Development NumPy)
target_include_directories(week_whatever_dz3 PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(week_whatever_dz3 Python3::Python Python3::NumPy)
