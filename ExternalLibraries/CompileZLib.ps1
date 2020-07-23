################################################################################
### Initializing

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$CURRENT_PATH = (Get-Item -Path ".\" -Verbose).FullName
if (-not (Get-Command Expand-7Zip -ErrorAction Ignore)) {
	Install-Package -Scope CurrentUser -Force 7Zip4PowerShell > $null
}

################################################################################
### Paths

$VERSION = "1.2.11"
$NAME = "zlib-$VERSION"
$TAR_NAME = "$NAME.tar"
$ZIP_NAME = "$TAR_NAME.gz"
$DOWNLOAD_ADDRESS = "http://www.zlib.net/$ZIP_NAME"
$INSTALL_PATH = "$CURRENT_PATH\zlib"
$SRC_PATH = "$CURRENT_PATH\$NAME"
$BUILD_PATH = "$SRC_PATH\build"

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
cmake -G "Visual Studio 15 2017 Win64" $SRC_PATH -DCMAKE_INSTALL_PREFIX:PATH=$INSTALL_PATH
cmake --build . --target INSTALL --config Release

################################################################################
### Clean installation directory

# define
$REM_ROOT_LIST = @(
	"bin",
	"share",
	"lib\zlib.lib"
)

# gather
$REM_LIST = @("TEMP_TO_DEL")
foreach ($item in $REM_ROOT_LIST) {
	$REM_LIST += $INSTALL_PATH + '\' + $item
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