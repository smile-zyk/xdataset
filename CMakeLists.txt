cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(xdataset)

find_package(Eigen3 3.3 CONFIG REQUIRED)

add_executable(xdataset
	main.cc
	cell_series.cc
)

target_link_libraries(xdataset PRIVATE Eigen3::Eigen)