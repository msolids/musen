; Copyright (c) 2013-2020, MUSEN Development Team.
; Copyright (c) 2026, DyssolTEC GmbH.
; All rights reserved. This file is part of MUSEN framework https://github.com/msolids/musen.
; See LICENSE file for license and warranty information.

#define ModelsCreatorDirSrc DataDir+"\ModelsCreator"
#define ModelsCreatorDir 	"{app}\ModelsCreator"
#define APIDir				"MUSEN_API"
#define TemplatesDir		"Models"
#define ExamplesDir			"ExampleModels"
#define DebugExeDir			"DebugRun"

#dim ModelTemplateDirs[4]
#define ModelTemplateDirs[0] "ModelPP"
#define ModelTemplateDirs[1] "ModelPW"
#define ModelTemplateDirs[2] "ModelSB"
#define ModelTemplateDirs[3] "ModelEF"

#dim ModelExampleDirs[4]
#define ModelExampleDirs[0] "ParticleParticle\HertzMindlin"
#define ModelExampleDirs[1] "ParticleWall\PWHertzMindlin"
#define ModelExampleDirs[2] "SolidBonds\BondModelElastic"
#define ModelExampleDirs[3] "ExternalForce\ViscousField"

#dim APIFiles[14]
#define APIFiles[00] "Modules\GeneralSources\AbstractDEMModel.h"
#define APIFiles[01] "Modules\GeneralSources\AbstractDEMModel.cpp"
#define APIFiles[02] "Modules\GeneralSources\BasicGPUFunctions.cuh"
#define APIFiles[03] "Modules\GeneralSources\BasicTypes.h"
#define APIFiles[04] "Modules\GeneralSources\DisableWarningHelper.h"
#define APIFiles[05] "Modules\GeneralSources\Quaternion.h"
#define APIFiles[06] "Modules\GeneralSources\Matrix3.h"
#define APIFiles[07] "Modules\GeneralSources\MUSENDefinitions.h"
#define APIFiles[08] "Modules\GeneralSources\MUSENHelperDefines.h"
#define APIFiles[09] "Modules\GeneralSources\Vector3.h"
#define APIFiles[10] "Modules\SimplifiedScene\SceneOptionalVariables.h"
#define APIFiles[11] "Modules\SimplifiedScene\SceneTypes.h"
#define APIFiles[12] "Modules\SimplifiedScene\SceneTypes.cpp"
#define APIFiles[13] "Modules\SimplifiedScene\SceneTypesGPU.h"

[Files]
; CMake project and helper script
Source: "{#ModelsCreatorDirSrc}\CMakeLists.txt";          DestDir: "{#ModelsCreatorDir}"; Flags: ignoreversion
Source: "{#ModelsCreatorDirSrc}\OpenInVisualStudio.bat";  DestDir: "{#ModelsCreatorDir}"; Flags: ignoreversion
Source: "{#SolutionDir}\cmake\CompilerSettings.cmake";    DestDir: "{#ModelsCreatorDir}\cmake"; Flags: ignoreversion
Source: "{#SolutionDir}\cmake\CudaSettings.cmake";        DestDir: "{#ModelsCreatorDir}\cmake"; Flags: ignoreversion
Source: "{#SolutionDir}\cmake\cccl_fix\CCCL4967.h";       DestDir: "{#ModelsCreatorDir}\cmake\cccl_fix"; Flags: ignoreversion

; Template models
#sub ModelTemplateDirs_entry
Source: "{#SolutionDir}\Models\Templates\{#ModelTemplateDirs[I]}\*.h";   DestDir: "{#ModelsCreatorDir}\{#TemplatesDir}\{#ModelTemplateDirs[I]}"; Flags: ignoreversion
Source: "{#SolutionDir}\Models\Templates\{#ModelTemplateDirs[I]}\*.cpp"; DestDir: "{#ModelsCreatorDir}\{#TemplatesDir}\{#ModelTemplateDirs[I]}"; Flags: ignoreversion
Source: "{#SolutionDir}\Models\Templates\{#ModelTemplateDirs[I]}\*.cuh"; DestDir: "{#ModelsCreatorDir}\{#TemplatesDir}\{#ModelTemplateDirs[I]}"; Flags: ignoreversion
Source: "{#SolutionDir}\Models\Templates\{#ModelTemplateDirs[I]}\*.cu";  DestDir: "{#ModelsCreatorDir}\{#TemplatesDir}\{#ModelTemplateDirs[I]}"; Flags: ignoreversion
#endsub
#for {I = 0; I < DimOf(ModelTemplateDirs); I++} ModelTemplateDirs_entry

; Example models
#sub ModelExampleDirs_entry
Source: "{#SolutionDir}\Models\{#ModelExampleDirs[I]}\*.h";   DestDir: "{#ModelsCreatorDir}\{#ExamplesDir}\{#ModelExampleDirs[I]}"; Flags: ignoreversion uninsremovereadonly overwritereadonly; Attribs: readonly
Source: "{#SolutionDir}\Models\{#ModelExampleDirs[I]}\*.cpp"; DestDir: "{#ModelsCreatorDir}\{#ExamplesDir}\{#ModelExampleDirs[I]}"; Flags: ignoreversion uninsremovereadonly overwritereadonly; Attribs: readonly
Source: "{#SolutionDir}\Models\{#ModelExampleDirs[I]}\*.cuh"; DestDir: "{#ModelsCreatorDir}\{#ExamplesDir}\{#ModelExampleDirs[I]}"; Flags: ignoreversion uninsremovereadonly overwritereadonly; Attribs: readonly
Source: "{#SolutionDir}\Models\{#ModelExampleDirs[I]}\*.cu";  DestDir: "{#ModelsCreatorDir}\{#ExamplesDir}\{#ModelExampleDirs[I]}"; Flags: ignoreversion uninsremovereadonly overwritereadonly; Attribs: readonly
#endsub
#for {I = 0; I < DimOf(ModelExampleDirs); I++} ModelExampleDirs_entry

; API files (read-only)
#sub APIFiles_entry
Source: "{#SolutionDir}\{#APIFiles[I]}"; DestDir: "{#ModelsCreatorDir}\{#APIDir}"; Flags: ignoreversion uninsremovereadonly overwritereadonly; Attribs: readonly
#endsub
#for {I = 0; I < DimOf(APIFiles); I++} APIFiles_entry

; Debug executables and Qt libraries
Source: "{#DebugDir}\{#MyAppExeName}"; DestDir: "{#ModelsCreatorDir}\{#DebugExeDir}"; Flags: ignoreversion
#sub QtLibsd_entry
Source: "{#QtPath}\bin\{#QtLibs[I]}d.dll"; DestDir: "{#ModelsCreatorDir}\{#DebugExeDir}"; Flags: ignoreversion
#endsub
#for {I = 0; I < DimOf(QtLibs); I++} QtLibsd_entry
#sub QtLibsPlatformsd_entry
Source: "{#QtPath}\plugins\{#QtPlatformsDir}\{#QtLibsPlatforms[I]}d.dll"; DestDir: "{#ModelsCreatorDir}\{#DebugExeDir}\{#QtPlatformsDir}"; Flags: ignoreversion
#endsub
#for {I = 0; I < DimOf(QtLibsPlatforms); I++} QtLibsPlatformsd_entry
#sub QtLibsImageFormatsd_entry
Source: "{#QtPath}\plugins\{#QtImageformatsDir}\{#QtLibsImageFormats[I]}d.dll"; DestDir: "{#ModelsCreatorDir}\{#DebugExeDir}\{#QtImageformatsDir}"; Flags: ignoreversion
#endsub
#for {I = 0; I < DimOf(QtLibsImageFormats); I++} QtLibsImageFormatsd_entry

[Dirs]
; Main directory
Name: "{#ModelsCreatorDir}"; Flags: uninsalwaysuninstall

; Directories for template models
Name: "{#ModelsCreatorDir}\{#TemplatesDir}"; Flags: uninsalwaysuninstall
#sub ModelTemplateDirs2_entry
Name: "{#ModelsCreatorDir}\{#TemplatesDir}\{#ModelTemplateDirs[I]}"; Flags: uninsalwaysuninstall
#endsub
#for {I = 0; I < DimOf(ModelTemplateDirs); I++} ModelTemplateDirs2_entry

; Directories for example models
Name: "{#ModelsCreatorDir}\{#ExamplesDir}"; Flags: uninsalwaysuninstall
#sub ModelExampleDirs2_entry
Name: "{#ModelsCreatorDir}\{#ExamplesDir}\{#ModelExampleDirs[I]}"; Flags: uninsalwaysuninstall
#endsub
#for {I = 0; I < DimOf(ModelExampleDirs); I++} ModelExampleDirs2_entry

Name: "{#ModelsCreatorDir}\cmake\cccl_fix"; Flags: uninsalwaysuninstall
Name: "{#ModelsCreatorDir}\cmake"; Flags: uninsalwaysuninstall
Name: "{#ModelsCreatorDir}\{#APIDir}"; Flags: uninsalwaysuninstall
Name: "{#ModelsCreatorDir}\{#DebugExeDir}"; Flags: uninsalwaysuninstall
Name: "{#ModelsCreatorDir}\{#DebugExeDir}\{#QtPlatformsDir}"; Flags: uninsalwaysuninstall
Name: "{#ModelsCreatorDir}\{#DebugExeDir}\{#QtImageformatsDir}"; Flags: uninsalwaysuninstall
