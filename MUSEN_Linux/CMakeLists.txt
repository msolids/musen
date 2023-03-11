cmake_minimum_required(VERSION 3.17)

project(MUSEN C CXX CUDA)

# Verbose for debugging
# set(CMAKE_VERBOSE_MAKEFILE ON)

# Overall compiler settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON) # Find includes in corresponding build directories

# CUDA
find_package(CUDAToolkit REQUIRED)

# Matlab for MMUSEN
find_package(Matlab 9.5 QUIET)
if (NOT Matlab_FOUND)
	set(Matlab_INCLUDE_DIRS "")
endif()

# QT for GUI Musen
find_package(Qt5 COMPONENTS Core REQUIRED)
find_package(Qt5 COMPONENTS Gui REQUIRED)
find_package(Qt5 COMPONENTS OpenGL REQUIRED)
find_package(Qt5 COMPONENTS Widgets REQUIRED)
set(CMAKE_AUTOMOC ON) # Instruct CMake to run moc automatically when needed
set(CMAKE_AUTOUIC ON) # Create code from a list of Qt designer ui files
set(CMAKE_AUTORCC ON) # Build Qt resource files
#set(Qt5_USE_STATIC_LIBS ON)
#set(Qt5_USE_STATIC_RUNTIME ON)

# c++ settings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pthread -fPIC") # -Wall -Wno-unknown-pragmas -Wno-unused-variable -Wno-reorder -std=c++14

# cuda settings
if (CUDAToolkit_VERSION_MAJOR LESS 11)
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_75,code=sm_75 -DVERBOSE -Xcompiler -pthread -Xcompiler -fPIC -Wno-deprecated-gpu-targets")
else ()
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -DVERBOSE -Xcompiler -pthread -Xcompiler -fPIC -Wno-deprecated-gpu-targets")
endif ()

# Get all subdirectories for include
FILE(GLOB_RECURSE new_list RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h)
SET(PROJ_DIRS "")
FOREACH(file_path ${new_list})
	GET_FILENAME_COMPONENT(dir_path ${file_path} PATH)
	SET(PROJ_DIRS ${PROJ_DIRS} ${dir_path})
ENDFOREACH()
LIST(REMOVE_DUPLICATES PROJ_DIRS)

# For GUI Musen: Find all files and exclude CMusen/MMusen
file(GLOB_RECURSE proj_guimusen_SRC
	RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/*.cc
	${CMAKE_CURRENT_SOURCE_DIR}/*.h
	${CMAKE_CURRENT_SOURCE_DIR}/*.cu
	${CMAKE_CURRENT_SOURCE_DIR}/*.cuh
)
set (EXCLUDE_DIR "CMusen/")
foreach (TMP_PATH ${proj_guimusen_SRC})
	string (FIND ${TMP_PATH} ${EXCLUDE_DIR} EXCLUDE_DIR_FOUND)
	if (NOT ${EXCLUDE_DIR_FOUND} EQUAL -1)
		list (REMOVE_ITEM proj_guimusen_SRC ${TMP_PATH})
	endif ()
endforeach(TMP_PATH)
set (EXCLUDE_DIR "MMusen/")
foreach (TMP_PATH ${proj_guimusen_SRC})
	string (FIND ${TMP_PATH} ${EXCLUDE_DIR} EXCLUDE_DIR_FOUND)
	if (NOT ${EXCLUDE_DIR_FOUND} EQUAL -1)
		list (REMOVE_ITEM proj_guimusen_SRC ${TMP_PATH})
	endif ()
endforeach(TMP_PATH)
#MESSAGE( STATUS "GUIMusen files are: " ${proj_guimusen_SRC} )

# For CMusen: Find all files and exclude QT_GUI/MMusen
file(GLOB_RECURSE proj_cmusen_SRC
	RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/*.cc
	${CMAKE_CURRENT_SOURCE_DIR}/*.h
	${CMAKE_CURRENT_SOURCE_DIR}/*.cu
	${CMAKE_CURRENT_SOURCE_DIR}/*.cuh
)
set (EXCLUDE_DIR "MMusen/")
foreach (TMP_PATH ${proj_cmusen_SRC})
	string (FIND ${TMP_PATH} ${EXCLUDE_DIR} EXCLUDE_DIR_FOUND)
	if (NOT ${EXCLUDE_DIR_FOUND} EQUAL -1)
		list (REMOVE_ITEM proj_cmusen_SRC ${TMP_PATH})
	endif ()
endforeach(TMP_PATH)
set (EXCLUDE_DIR "MusenGUI/")
foreach (TMP_PATH ${proj_cmusen_SRC})
	string (FIND ${TMP_PATH} ${EXCLUDE_DIR} EXCLUDE_DIR_FOUND)
	if (NOT ${EXCLUDE_DIR_FOUND} EQUAL -1)
		list (REMOVE_ITEM proj_cmusen_SRC ${TMP_PATH})
	endif ()
endforeach(TMP_PATH)
set (EXCLUDE_DIR "QTDialogs/")
foreach (TMP_PATH ${proj_cmusen_SRC})
	string (FIND ${TMP_PATH} ${EXCLUDE_DIR} EXCLUDE_DIR_FOUND)
	if (NOT ${EXCLUDE_DIR_FOUND} EQUAL -1)
		list (REMOVE_ITEM proj_cmusen_SRC ${TMP_PATH})
	endif ()
endforeach(TMP_PATH)
#MESSAGE( STATUS "CMusen files are: " ${proj_cmusen_SRC} )

# For MMusen: Find all files and exclude QT_GUI/CMusen
file(GLOB_RECURSE proj_mmusen_SRC
	RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/*.cc
	${CMAKE_CURRENT_SOURCE_DIR}/*.h
	${CMAKE_CURRENT_SOURCE_DIR}/*.cu
	${CMAKE_CURRENT_SOURCE_DIR}/*.cuh
)
set (EXCLUDE_DIR "CMusen/")
foreach (TMP_PATH ${proj_mmusen_SRC})
	string (FIND ${TMP_PATH} ${EXCLUDE_DIR} EXCLUDE_DIR_FOUND)
	if (NOT ${EXCLUDE_DIR_FOUND} EQUAL -1)
		list (REMOVE_ITEM proj_mmusen_SRC ${TMP_PATH})
	endif ()
endforeach(TMP_PATH)
set (EXCLUDE_DIR "MusenGUI/")
foreach (TMP_PATH ${proj_mmusen_SRC})
	string (FIND ${TMP_PATH} ${EXCLUDE_DIR} EXCLUDE_DIR_FOUND)
	if (NOT ${EXCLUDE_DIR_FOUND} EQUAL -1)
		list (REMOVE_ITEM proj_mmusen_SRC ${TMP_PATH})
	endif ()
endforeach(TMP_PATH)
set (EXCLUDE_DIR "QTDialogs/")
foreach (TMP_PATH ${proj_mmusen_SRC})
	string (FIND ${TMP_PATH} ${EXCLUDE_DIR} EXCLUDE_DIR_FOUND)
	if (NOT ${EXCLUDE_DIR_FOUND} EQUAL -1)
		list (REMOVE_ITEM proj_mmusen_SRC ${TMP_PATH})
	endif ()
endforeach(TMP_PATH)
#MESSAGE( STATUS "MMusen files are: " ${proj_mmusen_SRC} )

# Protobuf
file(GLOB_RECURSE generated_SRC
	${CMAKE_CURRENT_BINARY_DIR}/GeneratedFiles/SimulationDescription.pb.cc
	${CMAKE_CURRENT_BINARY_DIR}/GeneratedFiles/SimulationDescription.pb.h
	${CMAKE_CURRENT_BINARY_DIR}/GeneratedFiles/AgglomeratesDatabase.pb.cc
	${CMAKE_CURRENT_BINARY_DIR}/GeneratedFiles/AgglomeratesDatabase.pb.h
	${CMAKE_CURRENT_BINARY_DIR}/GeneratedFiles/MaterialsDatabase.pb.cc
	${CMAKE_CURRENT_BINARY_DIR}/GeneratedFiles/MaterialsDatabase.pb.h
	${CMAKE_CURRENT_BINARY_DIR}/GeneratedFiles/GeometriesDatabase.pb.cc
	${CMAKE_CURRENT_BINARY_DIR}/GeneratedFiles/GeometriesDatabase.pb.h
)

add_definitions(-D_CRT_SECURE_NO_WARNINGS)
add_definitions(-D_SCL_SECURE_NO_WARNINGS)
add_definitions(-DREMOVE_OBSOLETE)
add_definitions(-D_FILE_OFFSET_BITS=64)
add_definitions(-DPATH_CONFIGURED)

include_directories(
	${PROJ_DIRS} 
	${CMAKE_CURRENT_BINARY_DIR} 
	"$ENV{MUSEN_PROTO_PATH}/include/"
	"$ENV{MUSEN_ZLIB_PATH}/include/"
	${CUDAToolkit_INCLUDE_DIRS}
	"$ENV{MUSEN_BOOST_PATH}/include/"
	"$ENV{MUSEN_SCOREP_PATH}/include/"
	${Matlab_INCLUDE_DIRS}
)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/GeneratedFiles)

add_custom_target(
	protobuf
	COMMAND $ENV{MUSEN_PROTO_PATH}/bin/protoc --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/Modules/SimResultsStorage/ --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/Databases/MaterialsDatabase/ --cpp_out=${CMAKE_CURRENT_BINARY_DIR}/GeneratedFiles ${CMAKE_CURRENT_SOURCE_DIR}/Modules/SimResultsStorage/SimulationDescription.proto 
	COMMAND $ENV{MUSEN_PROTO_PATH}/bin/protoc --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/Databases/AgglomerateDatabase/ --cpp_out=${CMAKE_CURRENT_BINARY_DIR}/GeneratedFiles ${CMAKE_CURRENT_SOURCE_DIR}/Databases/AgglomerateDatabase/AgglomeratesDatabase.proto 
	COMMAND $ENV{MUSEN_PROTO_PATH}/bin/protoc --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/Databases/GeometriesDatabase/ --cpp_out=${CMAKE_CURRENT_BINARY_DIR}/GeneratedFiles ${CMAKE_CURRENT_SOURCE_DIR}/Databases/GeometriesDatabase/GeometriesDatabase.proto 
	COMMAND $ENV{MUSEN_PROTO_PATH}/bin/protoc --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/Databases/MaterialsDatabase/ --cpp_out=${CMAKE_CURRENT_BINARY_DIR}/GeneratedFiles ${CMAKE_CURRENT_SOURCE_DIR}/Databases/MaterialsDatabase/MaterialsDatabase.proto 
)


###########################################################################
# GUI MUSEN: graphic user interface musen
###########################################################################
add_executable(musen ${proj_guimusen_SRC} ${generated_SRC} MusenGUI/musengui.qrc)
add_dependencies(musen protobuf)
target_link_libraries(musen
	${LIBS} 
	"$ENV{MUSEN_PROTO_PATH}/lib/libprotobuf.a" 
	"$ENV{MUSEN_ZLIB_PATH}/lib/libz.a"
	Qt5::Core
	Qt5::Gui
	Qt5::OpenGL
	Qt5::Widgets
	"$ENV{OPENGL_PATH}/libGL.so"
	"$ENV{OPENGL_PATH}/libGLU.so"
)
#convert the fpic option in Qt5::Core over to INTERFACE_POSITION_INDEPENDENT_CODE
if ( TARGET Qt5::Core )
	get_property( core_options TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS )
	string( REPLACE "-fPIC" "" new_core_options "${core_options}" )
	set_property( TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS ${new_core_options} )
	set_property( TARGET Qt5::Core PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE "ON" )
	set( CMAKE_CXX_COMPILE_OPTIONS_PIE "-fPIC" )
endif()


###########################################################################
# CMUSEN: command line musen
###########################################################################
add_executable(cmusen ${proj_cmusen_SRC} ${generated_SRC})
add_dependencies(cmusen protobuf)
target_link_libraries(cmusen
	${LIBS} 
	"$ENV{MUSEN_PROTO_PATH}/lib/libprotobuf.a" 
	"$ENV{MUSEN_ZLIB_PATH}/lib/libz.a"
)


###########################################################################
# MMUSEN: Matlab interface
###########################################################################
if (Matlab_FOUND)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++")
	add_library(mmusen SHARED ${proj_mmusen_SRC} ${generated_SRC}) # should be SHARED
	add_dependencies(mmusen protobuf)
	target_link_libraries(mmusen
		${LIBS} 
		"$ENV{MUSEN_PROTO_PATH}/lib/libprotobuf.so" 
		"$ENV{MUSEN_ZLIB_PATH}/lib/libz.so"
		"${Matlab_ROOT_DIR}/bin/glnxa64/libut.so"
		"${Matlab_ROOT_DIR}/bin/glnxa64/libMatlabDataArray.so"
		"${Matlab_ROOT_DIR}/bin/glnxa64/libmx.so"
		"${Matlab_ROOT_DIR}/bin/glnxa64/libmex.so"
		"${Matlab_ROOT_DIR}/bin/glnxa64/libmat.so"
	)
	set_target_properties(mmusen PROPERTIES SUFFIX .mexa64 PREFIX "")
endif()
