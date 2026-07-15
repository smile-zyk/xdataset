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

add_executable(xdataset_test
	tests/cell_series_test.cc
	tests/multi_dimension_test.cc
	tests/block_test.cc
	tests/variable_test.cc
	src/cell_series.cc
	src/dimension_spec.cc
	src/multi_dimension_spec.cc
	src/multi_index_selector.cc
	src/variable_descriptor.cc
	src/variable.cc
	src/block.cc
	src/table_data.cc
)

add_executable(xdataset_playground
	apps/xdataset_playground/main.cc
	src/cell_series.cc
	src/dimension_spec.cc
	src/multi_dimension_spec.cc
	src/multi_index_selector.cc
	src/variable_descriptor.cc
	src/variable.cc
	src/block.cc
	src/table_data.cc
)

target_include_directories(xdataset_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/xdataset ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(xdataset_playground PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/xdataset ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(xdataset_test PRIVATE Eigen3::Eigen GTest::gtest_main Boost::headers tsl::ordered_map)
target_link_libraries(xdataset_playground PRIVATE Eigen3::Eigen Boost::headers tsl::ordered_map)

gtest_discover_tests(xdataset_test)