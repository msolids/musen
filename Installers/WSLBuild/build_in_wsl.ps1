# Copyright (c) 2023, MUSEN Development Team. All rights reserved. This file is part of MUSEN framework http://msolids.net/musen. See LICENSE file for license and warranty information.

# read arguments
$solution_dir = $args[0]
$compile_cli  = $args[1]
$compile_gui  = $args[2]
$install_aux  = $args[3]
$wsl_distro   = $args[4]

# variables
$result_path      = "$($solution_dir)Installers\Installers\WSL"
$version_file     = "$($solution_dir)Version\MUSENVersion.h"
$current_path     = (Get-Item -Path ".\" -Verbose).FullName
$wsl_src_path     = "${PWD}/musen_wsl/"
$wsl_build_path   = "$($wsl_src_path)build"
$wsl_install_path = "$($wsl_src_path)install"

# switch to x64 powershell if it is x86 now
if ($env:PROCESSOR_ARCHITEW6432 -eq "AMD64") {
	if ($myInvocation.Line) {
		&"$env:WINDIR\sysnative\windowspowershell\v1.0\powershell.exe" -NonInteractive -NoProfile $myInvocation.Line }
	else {
		&"$env:WINDIR\sysnative\windowspowershell\v1.0\powershell.exe" -NonInteractive -NoProfile -file "$($myInvocation.InvocationName)" $args }
	exit $lastexitcode
}

# get version info
$version1 = ((Get-Content $version_file)[ 9] -replace '\s+', ' ' -split ' ')[2]
$version2 = ((Get-Content $version_file)[10] -replace '\s+', ' ' -split ' ')[2]
$version3 = ((Get-Content $version_file)[11] -replace '\s+', ' ' -split ' ')[2]
$branch_name = (git rev-parse --abbrev-ref HEAD)

# prepare output directory
$bin_path = "$($result_path)\v$($version1).$($version2).$($version3)_$($branch_name)"
Remove-Item -Force -Recurse -Path $bin_path -ErrorAction SilentlyContinue | Out-Null
New-Item -ItemType Directory -Force -Path $bin_path | Out-Null

# build arguments line with selected targets for compilation script
$targets_params = ""
if ($compile_cli -eq "false") {
	$targets_params = $targets_params + " -DBUILD_CLI=NO"
}
if ($compile_gui -eq "false") {
	$targets_params = $targets_params + " -DBUILD_GUI=NO"
}
if ($install_aux -eq "false") {
	$targets_params = $targets_params + " -DINSTALL_AUX_DATA=NO"
}

# run compilation in WSL
$solution_dir_wsl = $solution_dir -replace '\\','\\\\'
$bin_path_wsl = $bin_path -replace '\\','\\\\'
if ($wsl_distro -eq "default") {
	wsl                            -e bash -lic "./build_in_wsl.sh $solution_dir_wsl $bin_path_wsl '$targets_params'"
} else {
	wsl --distribution $wsl_distro -e bash -lic "./build_in_wsl.sh $solution_dir_wsl $bin_path_wsl '$targets_params'"
}
