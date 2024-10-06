################################################################################
### Initializing

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$CURRENT_PATH = (Get-Item -Path ".\" -Verbose).FullName

################################################################################
### Paths

$MAJOR_VERSION = "3"
$MIDDLE_VERSION = "21"
$MINOR_VERSION = "12"
$VERSION = "$MAJOR_VERSION.$MIDDLE_VERSION.$MINOR_VERSION"
$DOWNLOAD_ADDRESS = "https://github.com/protocolbuffers/protobuf/releases/download/v$MIDDLE_VERSION.$MINOR_VERSION/protobuf-cpp-$VERSION.tar.gz"
$NAME = "protobuf-$VERSION"
$ZIP_NAME = "$NAME.tar.gz"
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
tar -xf $ZIP_NAME

################################################################################
### Build and install

# Build x64
New-Item $BUILD_PATH -ItemType directory
Set-Location $BUILD_PATH
cmake -G "Visual Studio 17 2022" $CMAKE_PATH `
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
### Clean work directory

Set-Location $CURRENT_PATH

Remove-Item $BUILD_PATH -Force -Recurse
Remove-Item $SRC_PATH -Force -Recurse
Remove-Item $ZIP_NAME -Force -Recurse
