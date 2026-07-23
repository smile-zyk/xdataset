cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(xdataset)

find_package(Eigen3 3.3 CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)
find_package(Boost 1.72 CONFIG REQUIRED)
find_package(tsl-ordered-map CONFIG REQUIRED)
find_package(llnl-units CONFIG REQUIRED)
include(GoogleTest)

enable_testing()

set(XDATASET_SOURCES
	src/dimension_spec.cc
	src/multi_dimension_spec.cc
	src/multi_index_selector.cc
	src/data_array.cc
	src/block.cc
	src/data_frame.cc
	src/data_series.cc
	src/arithmetic.cc
	src/measurement.cc
	src/unit.cc
	src/dataset.cc
)

add_library(xdataset SHARED ${XDATASET_SOURCES})
target_compile_definitions(xdataset PRIVATE XDATASET_BUILD_DLL)

add_executable(xdataset_test
	tests/data_series_test.cc
	tests/measurement_test.cc
	tests/arithmetic_test.cc
	tests/multi_dimension_test.cc
	tests/block_test.cc
	tests/data_array_test.cc
	tests/data_frame_test.cc
	tests/unit_test.cc
	tests/dataset_test.cc
)

add_executable(xdataset_playground
	apps/xdataset_playground/main.cc
)

target_include_directories(xdataset PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/xdataset ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(xdataset_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/xdataset ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(xdataset_playground PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/xdataset ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(xdataset PUBLIC Eigen3::Eigen Boost::headers tsl::ordered_map llnl-units::units)
target_link_libraries(xdataset_test PRIVATE xdataset GTest::gtest_main)
target_link_libraries(xdataset_playground PRIVATE xdataset)

gtest_discover_tests(xdataset_test)