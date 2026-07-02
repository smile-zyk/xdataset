cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(xdataset)

find_package(Eigen3 3.3 CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

enable_testing()

add_executable(cell_series_test
	tests/cell_series_test.cc
	cell_series.cc
	variable_data.cc
)

add_executable(variable_data_test
	tests/variable_data_test.cc
	cell_series.cc
	variable_data.cc
)

target_include_directories(cell_series_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(variable_data_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(cell_series_test PRIVATE Eigen3::Eigen GTest::gtest_main)
target_link_libraries(variable_data_test PRIVATE Eigen3::Eigen GTest::gtest_main)

gtest_discover_tests(cell_series_test)
gtest_discover_tests(variable_data_test)