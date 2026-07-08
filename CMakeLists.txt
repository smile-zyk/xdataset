cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(xdataset)

find_package(Eigen3 3.3 CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)
find_package(Boost 1.72 CONFIG REQUIRED)
find_package(tsl-ordered-map CONFIG REQUIRED)
include(GoogleTest)

enable_testing()

add_executable(cell_series_test
	tests/cell_series_test.cc
	cell_series.cc
	variable_spec.cc
	variable_data.cc
	block.cc
)

target_include_directories(cell_series_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(cell_series_test PRIVATE Eigen3::Eigen GTest::gtest_main Boost::headers tsl::ordered_map)

gtest_discover_tests(cell_series_test)