# Copyright (c) 2023, MUSEN Development Team. All rights reserved. This file is part of MUSEN framework http://msolids.net/musen. See LICENSE file for license and warranty information.

# get current date and time
$date_time = Get-Date -Format "yyMMdd.HHmmss"

# check if git is installed and accessible
[bool] $is_git_installed = $false
try {
	git | Out-Null
	$is_git_installed = $true
}
catch [System.Management.Automation.CommandNotFoundException] {
    Write-Warning "Git not found. No additional version information will be generated."
}

# check if it is a git repository
[bool] $is_git_repo = $false
if ($is_git_installed -eq $true) {
	$rev_parse_output = git rev-parse --is-inside-work-tree
	if ($rev_parse_output -eq 'true') {
		$is_git_repo = $true
	}
	else {
		Write-Warning "Not a git repository. No additional version information will be generated."
	}
}

# get hash of the current git commit and branch name
$hash_branch = ""
if ($is_git_repo -eq $true) {
	try { $hash = git rev-parse --short HEAD }
	catch {
		Write-Warning "Can not get commit hash. No additional version information will be generated"
	}
	try { $branch = git rev-parse --abbrev-ref HEAD }
	catch {
		Write-Warning "Can not get branch name. No additional version information will be generated"
	}
	if ($hash -And $branch) {
		$hash_branch = $hash + "." + $branch
	}
}

# get name of the current git branch
$branch = ""
if ($is_git_repo -eq $true) {
}

# write build time
try { 
	$time_file = "$PSScriptRoot\BuildTime.h"
	[io.file]::OpenWrite($time_file).close() 
	Set-Content -Path $time_file -Value "#pragma once`nconst std::string _BUILD_TIME = `"$date_time`";" -Erroraction 'silentlycontinue'
}
catch { 
	Write-Warning "Time file is locked and may not be updated." 
}

# write hash and branch
try { 
	$hash_file = "$PSScriptRoot\BuildHash.h"
	[io.file]::OpenWrite($hash_file).close() 
	Set-Content -Path $hash_file -Value "#pragma once`nconst std::string _BUILD_HASH = `"$hash_branch`";" -Erroraction 'silentlycontinue'
}
catch { 
	Write-Warning "Hash file is locked and may not be updated." 
}
