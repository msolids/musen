:: Copyright (c) 2013-2023, MUSEN Development Team. All rights reserved.
:: This file is part of MUSEN framework http://msolids.net/musen.
:: See LICENSE file for license and warranty information.

@echo off

:: ===== compile solution =====
echo Compiling Release x64
devenv %2 /build "Release|x64"
echo Compiling Debug x64
devenv %2 /build "Debug|x64"

:: ===== create a file with additional varsion information =====
echo Getting version information
:: file name
set INI_FILE=data.ini
:: get name of the current git branch
set CURR_COMMIT_NAME=open
:: check if the file is available to write, and write data if it is
2>nul ( 
	>>%~dp0\%INI_FILE% (call )
) && (call :write_commit_name) || (echo %INI_FILE% file is locked and may not be updated)

:: ===== run installer compiler =====
echo Compiling installer
set QtInstallDir="%3"
..\Compiler\ISCC ..\Scripts\Main.iss "/dQtPath=%QtInstallDir%"

goto :eof

:: ===== functions =====
:: write commit name into ini file
:write_commit_name
(
	echo [Version]
	echo Branch=%CURR_COMMIT_NAME%
) > %~dp0\%INI_FILE%
goto :eof