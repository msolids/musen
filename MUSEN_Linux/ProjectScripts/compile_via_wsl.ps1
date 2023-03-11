# read arguments
$remote_pc   = $args[0]
$compile_cli = $args[1]
$compile_gui = $args[2]
$compile_mat = $args[3]

# variables
$compiled_path = "..\compiled"
$version_file  = "..\..\Version\MUSENVersion.h"
$current_path  = (Get-Item -Path ".\" -Verbose).FullName

# switch to x64 powershell if it is x86 now
if ($env:PROCESSOR_ARCHITEW6432 -eq "AMD64") {
	if ($myInvocation.Line) {
		&"$env:WINDIR\sysnative\windowspowershell\v1.0\powershell.exe" -NonInteractive -NoProfile $myInvocation.Line }
	else {
		&"$env:WINDIR\sysnative\windowspowershell\v1.0\powershell.exe" -NonInteractive -NoProfile -file "$($myInvocation.InvocationName)" $args }
	exit $lastexitcode
}

#remove old files if exist
Write-Host "Remove old compiled files"
Remove-Item $compiled_path\cmusen           -ErrorAction Ignore
Remove-Item $compiled_path\musen_gui.tar.gz -ErrorAction Ignore
Remove-Item $compiled_path\mmusen.mexa64    -ErrorAction Ignore

# build arguments line with selected targets for compilation script
$targets_params = ""
if ($compile_cli -eq "true") {
	$targets_params = $targets_params + " --target=cli"
}
if ($compile_gui -eq "true") {
	$targets_params = $targets_params + " --target=gui"
}
if ($compile_mat -eq "true") {
	$targets_params = $targets_params + " --target=matlab"
}

# run compilation command via WSL
if ($remote_pc -eq "hpc5") {
	Write-Host "Run compilation on HPC5 via WSL"
	bash -c "cd ../ && ./compile_on_hpc5.sh $targets_params"
}
else {
	Write-Host "Run compilation locally wia WSL"
	bash -c "cd ../ && ./compile_on_host.sh $targets_params"
}

# check that any files were created
if (!(Test-Path $compiled_path\musen_gui.tar.gz) -and !(Test-Path $compiled_path\cmusen) -and !(Test-Path $compiled_path\mmusen.mexa64)) {
	Write-Warning "No files compiled!"
	exit 1
}

# get version info
$version1 = ((Get-Content $version_file)[ 9] -replace '\s+', ' ' -split ' ')[2]
$version2 = ((Get-Content $version_file)[10] -replace '\s+', ' ' -split ' ')[2]
$version3 = ((Get-Content $version_file)[11] -replace '\s+', ' ' -split ' ')[2]
$branch_name = (git rev-parse --abbrev-ref HEAD)

# create directory
$bin_path = "$($compiled_path)\v$($version1).$($version2).$($version3)_$($branch_name)"
New-Item -ItemType Directory -Force -Path $bin_path | Out-Null

# move compiled files 
Write-Host "Copy compiled files into " ([System.IO.Path]::GetFullPath($current_path + '\' + $bin_path))
Move-Item -Path $compiled_path\cmusen           -Destination $bin_path -Force -ErrorAction SilentlyContinue
Move-Item -Path $compiled_path\musen_gui.tar.gz -Destination $bin_path -Force -ErrorAction SilentlyContinue
Move-Item -Path $compiled_path\mmusen.mexa64    -Destination $bin_path -Force -ErrorAction SilentlyContinue
