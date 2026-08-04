cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(xdataset)

option(BUILD_XDATASET_TEST "Build xdataset unit tests" ON)

find_package(Eigen3 3.3 CONFIG REQUIRED)
find_package(Boost 1.72 CONFIG REQUIRED)
find_package(tsl-ordered-map CONFIG REQUIRED)
find_package(llnl-units CONFIG REQUIRED)

# HDF5: prefer Homebrew/system HDF5 via HDF5_ROOT hint, fall back to CMake's FindHDF5
if(HDF5_ROOT)
    set(HDF5_ROOT_HINT HINTS "${HDF5_ROOT}")
elseif(DEFINED ENV{HDF5_ROOT})
    set(HDF5_ROOT_HINT HINTS "$ENV{HDF5_ROOT}")
elseif(APPLE)
    # Auto-detect Homebrew HDF5 on macOS
    set(HDF5_ROOT_HINT HINTS "/opt/homebrew/opt/hdf5" "/usr/local/opt/hdf5")
endif()

find_package(HDF5 REQUIRED COMPONENTS C)
set(HDF5_TARGET hdf5::hdf5)
message(STATUS "HDF5: found at ${HDF5_ROOT} (version ${HDF5_VERSION})")

set(XDATASET_SOURCES
	src/dimension_spec.cc
	src/multi_dimension_spec.cc
	src/multi_index_selector.cc
	src/data_array.cc
	src/block.cc
	src/data_frame.cc
	src/data_series.cc
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

# GCC 15 needs -O1 for Eigen header-only template inlining, regardless of Eigen version
target_compile_options(xdataset PRIVATE $<$<CXX_COMPILER_ID:GNU>:$<$<CONFIG:Debug>:-O1>>)

if(BUILD_XDATASET_TEST)
    find_package(GTest CONFIG REQUIRED)
    include(GoogleTest)
    enable_testing()

    add_executable(xdataset_test
        tests/data_series_test.cc
        tests/measurement_test.cc
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

    target_include_directories(xdataset_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/xdataset ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(xdataset_test PRIVATE xdataset GTest::gtest_main)
    gtest_discover_tests(xdataset_test)
endif()

add_executable(xdataset_playground
    apps/xdataset_playground/main.cc
)

target_include_directories(xdataset PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/xdataset ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(xdataset_playground PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/xdataset ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(xdataset PUBLIC Eigen3::Eigen Boost::headers tsl::ordered_map llnl-units::units ${HDF5_TARGET})
target_link_libraries(xdataset_playground PRIVATE xdataset)

# =============================================================================
# Python bindings (pybind11)
# =============================================================================
#
# The pybind11 CMake package is located through the *Python module* pybind11:
#   <python> -m pybind11 --cmakedir
#
# On Windows the interpreter defaults to the MSYS2 MinGW Python so that the
# resulting .pyd stays ABI-compatible with the MinGW-built xdataset.dll.
# Override with:  -DPython3_EXECUTABLE=<path-to-python>
# =============================================================================

option(BUILD_XDATASET_PYTHON "Build the Python bindings (pybind11)" ON)

if(BUILD_XDATASET_PYTHON)
    if(WIN32 AND NOT DEFINED Python3_EXECUTABLE)
        foreach(_py_candidate
                "C:/msys64/mingw64/bin/python.exe"
                "C:/msys2/mingw64/bin/python.exe")
            if(EXISTS "${_py_candidate}")
                set(Python3_EXECUTABLE "${_py_candidate}" CACHE FILEPATH
                    "Python interpreter used for the xdataset Python bindings")
                break()
            endif()
        endforeach()
    endif()

    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
    message(STATUS "Python bindings interpreter: ${Python3_EXECUTABLE}")

    # Use the Python module pybind11 to resolve the CMake package location.
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _pybind11_cmakedir_result
        ERROR_QUIET)
    if(NOT _pybind11_cmakedir_result EQUAL 0 OR
       NOT EXISTS "${PYBIND11_CMAKE_DIR}/pybind11Config.cmake")
        message(WARNING
            "pybind11 Python module not found for ${Python3_EXECUTABLE}; "
            "skipping the Python bindings. Install it with: "
            "\"${Python3_EXECUTABLE}\" -m pip install pybind11")
    else()
        message(STATUS "pybind11 CMake dir: ${PYBIND11_CMAKE_DIR}")
        list(PREPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
        find_package(pybind11 CONFIG REQUIRED)

        # NumPy include dir — required by pybind11/numpy.h and pybind11/eigen.h.
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
            OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _numpy_include_result)
        if(NOT _numpy_include_result EQUAL 0 OR NOT EXISTS "${NUMPY_INCLUDE_DIR}")
            message(FATAL_ERROR
                "numpy is required to build the Python bindings. Install it with: "
                "\"${Python3_EXECUTABLE}\" -m pip install numpy")
        endif()

        # pybind11_add_module(xdataset_pybind python/xdataset_pybind.cc)

        # target_include_directories(xdataset_pybind PRIVATE
        #     ${CMAKE_CURRENT_SOURCE_DIR}/include/xdataset
        #     ${CMAKE_CURRENT_SOURCE_DIR}
        #     ${NUMPY_INCLUDE_DIR})

        # target_link_libraries(xdataset_pybind PRIVATE xdataset)

        # # Guarantee the importable name "xdataset.pyd" regardless of the
        # # ABI-tagged suffix pybind11 would otherwise generate.
        # # The bindings stay on C++11 (the same standard as the library).
        # set_target_properties(xdataset_pybind PROPERTIES
        #     CXX_STANDARD 11
        #     CXX_STANDARD_REQUIRED ON
        #     CXX_EXTENSIONS OFF
        #     OUTPUT_NAME xdataset
        #     PREFIX ""
        #     SUFFIX ".pyd")

        # # GCC 15 Debug needs -O1 for Eigen template inlining (same as xdataset).
        # target_compile_options(xdataset_pybind PRIVATE
        #     $<$<CXX_COMPILER_ID:GNU>:$<$<CONFIG:Debug>:-O1>>)
    endif()
endif()