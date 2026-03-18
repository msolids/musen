@echo off
:: Generates the Visual Studio solution for MUSEN ModelsCreator and opens it.
:: Prerequisites: CMake 3.23+, Visual Studio 2022, CUDA Toolkit.

setlocal

cd /d "%~dp0"

echo Generating Visual Studio solution...
cmake -B build -G "Visual Studio 17 2022"
if %errorlevel% neq 0 (
    echo.
    echo ERROR: CMake configuration failed.
    echo Make sure CMake 3.23+, Visual Studio 2022, and CUDA Toolkit are installed.
    pause
    exit /b 1
)

echo.
echo Opening solution in Visual Studio...
start "" "build\MUSEN_ModelsCreator.sln"
