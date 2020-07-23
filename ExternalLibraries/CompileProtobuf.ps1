################################################################################
### Initializing

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$CURRENT_PATH = (Get-Item -Path ".\" -Verbose).FullName
if (-not (Get-Command Expand-7Zip -ErrorAction Ignore)) {
	Install-Package -Scope CurrentUser -Force 7Zip4PowerShell > $null
}

################################################################################
### Paths

$MAJOR_VERSION = "3"
$MIDDLE_VERSION = "9"
$MINOR_VERSION = "1"
$VERSION = "$MAJOR_VERSION.$MIDDLE_VERSION.$MINOR_VERSION"
$DOWNLOAD_ADDRESS = "https://github.com/protocolbuffers/protobuf/releases/download/v$VERSION/protobuf-cpp-$VERSION.tar.gz"
$NAME = "protobuf-$VERSION"
$TAR_NAME = "$NAME.tar"
$ZIP_NAME = "$TAR_NAME.gz"
$INSTALL_PATH = "$CURRENT_PATH\protobuf"
$SRC_PATH = "$CURRENT_PATH\$NAME"
$CMAKE_PATH = "$SRC_PATH\cmake"
$BUILD_PATH = "$CMAKE_PATH\build"

# libs
$ZLIB_INSTALL_PATH = "$CURRENT_PATH\zlib"

################################################################################
### Clear old

Remove-Item $INSTALL_PATH -Force -Recurse -ErrorAction Ignore
Remove-Item $BUILD_PATH -Force -Recurse -ErrorAction Ignore
Remove-Item $SRC_PATH -Force -Recurse -ErrorAction Ignore

################################################################################
### Download

Invoke-WebRequest $DOWNLOAD_ADDRESS -OutFile $ZIP_NAME
Expand-7Zip $ZIP_NAME . | Expand-7Zip $TAR_NAME .

################################################################################
### Build and install

# Build x64
New-Item $BUILD_PATH -ItemType directory
Set-Location $BUILD_PATH
cmake -G "Visual Studio 15 2017 Win64" $CMAKE_PATH `
	-DCMAKE_INSTALL_PREFIX:PATH=$INSTALL_PATH `
	-Dprotobuf_BUILD_EXAMPLES=OFF `
	-Dprotobuf_BUILD_TESTS=OFF `
	-Dprotobuf_WITH_ZLIB=ON `
	-Dprotobuf_MSVC_STATIC_RUNTIME=OFF `
	-DZLIB_INCLUDE_DIR="$ZLIB_INSTALL_PATH\include" `
	-DZLIB_LIBRARY="$ZLIB_INSTALL_PATH\lib\zlibstatic.lib" 
cmake --build . --target INSTALL --config Debug
cmake --build . --target INSTALL --config Release

################################################################################
### Clean installation directory

# define
$REM_ROOT_LIST = @(
	"cmake"
)
$REM_INCLUDE_LIST = @(
	"compiler",
	"util",
	"any.h",
	"any.pb.h",
	"any.proto",
	"api.pb.h",
	"api.proto",
	"descriptor.pb.h",
	"descriptor.proto",
	"descriptor_database.h",
	"duration.pb.h",
	"duration.proto",
	"dynamic_message.h",
	"empty.pb.h",
	"empty.proto",
	"extension_set_inl.h",
	"field_mask.pb.h",
	"field_mask.proto",
	"map_entry.h",
	"map_field.h",
	"map_field_inl.h",
	"reflection.h",
	"service.h",
	"source_context.pb.h",
	"source_context.proto",
	"struct.pb.h",
	"struct.proto",
	"text_format.h",
	"timestamp.pb.h",
	"timestamp.proto",
	"type.pb.h",
	"type.proto",
	"wrappers.pb.h",
	"wrappers.proto",
	"io\io_win32.h",
	"io\printer.h",
	"io\strtod.h",
	"io\tokenizer.h",
	"io\zero_copy_stream_impl.h",
	"stubs\bytestream.h",
	"stubs\status.h",
	"stubs\template_util.h"
)
$REM_LIB_LIST = @(
	"pkgconfig",
	"libprotobuf-lite.lib",
	"libprotobuf-lited.lib",
	"libprotoc.lib",
	"libprotocd.lib"
)

# gather
$REM_LIST = @("TEMP_TO_DEL")
foreach ($item in $REM_ROOT_LIST) {
	$REM_LIST += $INSTALL_PATH + '\' + $item
}
foreach ($item in $REM_INCLUDE_LIST) {
	$REM_LIST += $INSTALL_PATH + '\include\google\protobuf\' + $item
}
foreach ($item in $REM_LIB_LIST) {
	$REM_LIST += $INSTALL_PATH + '\lib\' + $item
}
# remove
foreach ($item in $REM_LIST) {
	Remove-Item "$item" -Force -Recurse -ErrorAction Ignore
}

################################################################################
### Clean work directory

Set-Location $CURRENT_PATH

Remove-Item $BUILD_PATH -Force -Recurse
Remove-Item $SRC_PATH -Force -Recurse
Remove-Item $ZIP_NAME -Force -Recurse
Remove-Item $TAR_NAME -Force -Recurse