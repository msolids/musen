# Copyright (c) 2023, MUSEN Development Team. All rights reserved.
# This file is part of MUSEN framework http://msolids.net/musen.
# See LICENSE file for license and warranty information.

### read arguments

$solution_dir   = $args[0]
$solution_path  = $args[1]
$qt_install_dir = $args[2]

### compile solution

Write-Host "Compiling Release x64"
devenv $solution_path /build "Release|x64"
Write-Host "Compiling Debug x64"
devenv $solution_path /build "Debug|x64"

### create a file with git branch

# check if git is installed and accessible
[bool] $is_git_installed = $false
try {
	git | Out-Null
	$is_git_installed = $true
}
catch [System.Management.Automation.CommandNotFoundException] {
    Write-Warning "Git not found. Branch name will not be used for installer."
}

# check if it is a git repository
[bool] $is_git_repo = $false
if ($is_git_installed -eq $true) {
	$rev_parse_output = git rev-parse --is-inside-work-tree
	if ($rev_parse_output -eq 'true') {
		$is_git_repo = $true
	}
	else {
		Write-Warning "Not a git repository. Branch name will not be used for installer."
	}
}

# get git branch name
$branch_name = ""
if ($is_git_repo -eq $true) {
	try { $branch_name = git symbolic-ref --short HEAD }
	catch {
		Write-Warning "Can not get branch name. Branch name will not be used for installer."
	}
}

# write branch name
try { 
	$ini_file = "$PSScriptRoot\data.ini"
	[io.file]::OpenWrite($ini_file).close() 
	Set-Content -Path $ini_file -Value "[Version]`nBranch=$branch_name" -Erroraction 'silentlycontinue'
}
catch { 
	Write-Warning "$ini_file file is locked and may not be updated." 
}

### run installer compiler

Write-Host "Compiling Installer"
$env:QtInstallDir="$qt_install_dir"
& $solution_dir\Installers\Compiler\ISCC.exe "$solution_dir\Installers\Scripts\Main.iss" "/dQtPath=$qt_install_dir"
