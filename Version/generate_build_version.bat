@echo off
set DATE_TIME=
for /f "skip=1 delims=" %%x in ('WMIC OS GET LocalDateTime') do if not defined DATE_TIME set DATE_TIME=%%x

set DATE.YEAR=%DATE_TIME:~2,2%
set DATE.MONTH=%DATE_TIME:~4,2%
set DATE.DAY=%DATE_TIME:~6,2%
set DATE.HOUR=%DATE_TIME:~8,2%
set DATE.MINUTE=%DATE_TIME:~10,2%
set DATE.SECOND=%DATE_TIME:~12,2%

set DATE_TIME_F=%DATE.YEAR%%DATE.MONTH%%DATE.DAY%.%DATE.HOUR%%DATE.MINUTE%%DATE.SECOND%

for /f %%i in ('git rev-parse --short HEAD') do set CURR_COMMIT_HASH=%%i
for /f %%i in ('git rev-parse --abbrev-ref HEAD') do set CURR_COMMIT_NAME=%%i

:: write build time
2>nul (		REM check if file is available to write
	>>%~dp0\BuildTime.h (call )
) && (call :write_time) || (echo Time file is locked and may not be updated)
:: write build hash
2>nul (		REM check if file is available to write
	>>%~dp0\BuildHash.h (call )
) && (call :write_hash) || (echo Hash file is locked and may not be updated)

goto :eof

:write_time
(
	echo #pragma once
	echo const std::string _BUILD_TIME = "%DATE_TIME_F%";
) > %~dp0\BuildTime.h
goto :eof

:write_hash
(
	echo #pragma once
	echo const std::string _BUILD_HASH = "%CURR_COMMIT_HASH%.%CURR_COMMIT_NAME%";
) > %~dp0\BuildHash.h
goto :eof