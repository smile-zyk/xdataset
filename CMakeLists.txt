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

# HDF5: when HDF5_ROOT is set (preset/env), use manual import; otherwise try vcpkg find_package
if(HDF5_ROOT OR DEFINED ENV{HDF5_ROOT})
    if(NOT HDF5_ROOT)
        set(HDF5_ROOT "$ENV{HDF5_ROOT}")
    endif()
    add_library(hdf5_imported SHARED IMPORTED)
    set_target_properties(hdf5_imported PROPERTIES
        IMPORTED_LOCATION "${HDF5_ROOT}/bin/libhdf5-320.dll"
        IMPORTED_IMPLIB "${HDF5_ROOT}/lib/libhdf5.dll.a"
        INTERFACE_INCLUDE_DIRECTORIES "${HDF5_ROOT}/include"
    )
    set(HDF5_TARGET hdf5_imported)
    message(STATUS "HDF5: using system package at ${HDF5_ROOT}")
else()
    find_package(hdf5 CONFIG REQUIRED)
    set(HDF5_TARGET hdf5::hdf5)
    message(STATUS "HDF5: using vcpkg package")
endif()

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
	src/value.cc
	src/operation.cc
	src/measurement.cc
	src/unit.cc
	src/dataset.cc
	src/hdf5_io.cc
	src/touchstone_io.cc
	src/data_array_io.cc
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
	tests/hdf5_io_test.cc
	tests/touchstone_io_test.cc
	tests/value_test.cc
	tests/operation_test.cc
)

add_executable(xdataset_playground
	apps/xdataset_playground/main.cc
)

target_include_directories(xdataset PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/xdataset ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(xdataset_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/xdataset ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(xdataset_playground PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/xdataset ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(xdataset PUBLIC Eigen3::Eigen Boost::headers tsl::ordered_map llnl-units::units ${HDF5_TARGET})
target_link_libraries(xdataset_test PRIVATE xdataset GTest::gtest_main)
target_link_libraries(xdataset_playground PRIVATE xdataset)

gtest_discover_tests(xdataset_test)