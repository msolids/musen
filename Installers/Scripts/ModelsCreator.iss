// ; Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
// ; This file is part of MUSEN framework http://msolids.net/musen.
// ; See LICENSE file for license and warranty information. 

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

#dim APIFiles[13]
#define APIFiles[00] "Modules\GeneralSources\AbstractDEMModel.h"; 	
#define APIFiles[01] "Modules\GeneralSources\AbstractDEMModel.cpp";	
#define APIFiles[02] "Modules\GeneralSources\BasicGPUFunctions.cuh";	
#define APIFiles[03] "Modules\GeneralSources\BasicTypes.h";			
#define APIFiles[04] "Modules\GeneralSources\DisableWarningHelper.h";			
#define APIFiles[05] "Modules\GeneralSources\Quaternion.h";			
#define APIFiles[06] "Modules\GeneralSources\Matrix3.h";					
#define APIFiles[07] "Modules\GeneralSources\MUSENDefinitions.h";					
#define APIFiles[08] "Modules\GeneralSources\Vector3.h";					
#define APIFiles[09] "Modules\SimplifiedScene\SceneOptionalVariables.h";					
#define APIFiles[10] "Modules\SimplifiedScene\SceneTypes.h";					
#define APIFiles[11] "Modules\SimplifiedScene\SceneTypes.cpp";				
#define APIFiles[12] "Modules\SimplifiedScene\SceneTypesGPU.h";				

[Files]
; Solution file
Source: "{#ModelsCreatorDirSrc}\MUSEN_ModelsCreator.sln"; 							DestDir: "{#ModelsCreatorDir}"; 		Flags: ignoreversion

; Template models
#sub ModelTemplateDirs_entry
Source: "{#SolutionDir}\Models\Templates\{#ModelTemplateDirs[I]}\*.h";		 DestDir: "{#ModelsCreatorDir}\{#TemplatesDir}\{#ModelTemplateDirs[I]}";	Flags: ignoreversion
Source: "{#SolutionDir}\Models\Templates\{#ModelTemplateDirs[I]}\*.cpp";	 DestDir: "{#ModelsCreatorDir}\{#TemplatesDir}\{#ModelTemplateDirs[I]}";	Flags: ignoreversion
Source: "{#SolutionDir}\Models\Templates\{#ModelTemplateDirs[I]}\*.cuh";	 DestDir: "{#ModelsCreatorDir}\{#TemplatesDir}\{#ModelTemplateDirs[I]}";	Flags: ignoreversion
Source: "{#SolutionDir}\Models\Templates\{#ModelTemplateDirs[I]}\*.cu";		 DestDir: "{#ModelsCreatorDir}\{#TemplatesDir}\{#ModelTemplateDirs[I]}";	Flags: ignoreversion
Source: "{#SolutionDir}\Models\Templates\{#ModelTemplateDirs[I]}\*.vcxproj"; DestDir: "{#ModelsCreatorDir}\{#TemplatesDir}\{#ModelTemplateDirs[I]}";	Flags: ignoreversion
#endsub
#for {I = 0; I < DimOf(ModelTemplateDirs); I++} ModelTemplateDirs_entry

; Example models
#sub ModelExampleDirs_entry
Source: "{#SolutionDir}\Models\{#ModelExampleDirs[I]}\*.h";		DestDir: "{#ModelsCreatorDir}\{#ExamplesDir}\{#ModelExampleDirs[I]}";	Flags: ignoreversion
Source: "{#SolutionDir}\Models\{#ModelExampleDirs[I]}\*.cpp";	DestDir: "{#ModelsCreatorDir}\{#ExamplesDir}\{#ModelExampleDirs[I]}";	Flags: ignoreversion
Source: "{#SolutionDir}\Models\{#ModelExampleDirs[I]}\*.cuh";	DestDir: "{#ModelsCreatorDir}\{#ExamplesDir}\{#ModelExampleDirs[I]}";	Flags: ignoreversion
Source: "{#SolutionDir}\Models\{#ModelExampleDirs[I]}\*.cu";	DestDir: "{#ModelsCreatorDir}\{#ExamplesDir}\{#ModelExampleDirs[I]}";	Flags: ignoreversion
#endsub
#for {I = 0; I < DimOf(ModelExampleDirs); I++} ModelExampleDirs_entry

; API files
#sub APIFiles_entry
Source: "{#SolutionDir}\{#APIFiles[I]}";				DestDir: "{#ModelsCreatorDir}\{#APIDir}"; Flags: ignoreversion uninsremovereadonly overwritereadonly; Attribs: readonly
#endsub
#for {I = 0; I < DimOf(APIFiles); I++} APIFiles_entry		
Source: "{#ModelsCreatorDirSrc}\{#APIDir}\*.vcxproj"; 					DestDir: "{#ModelsCreatorDir}\{#APIDir}"; Flags: ignoreversion
Source: "{#ModelsCreatorDirSrc}\{#APIDir}\*.vcxproj.user"; 				DestDir: "{#ModelsCreatorDir}\{#APIDir}"; Flags: ignoreversion
			
; Property sheets
Source: "{#SolutionDir}\PropertySheets\MusenCommonDebug.props";   DestDir: "{#ModelsCreatorDir}\PropertySheets"; 		Flags: ignoreversion
Source: "{#SolutionDir}\PropertySheets\MusenCommonRelease.props"; DestDir: "{#ModelsCreatorDir}\PropertySheets"; 		Flags: ignoreversion
Source: "{#SolutionDir}\PropertySheets\MusenCommonPath.props"; 	  DestDir: "{#ModelsCreatorDir}\PropertySheets"; 		Flags: ignoreversion

; Debud executables and Qt libraties
Source: "{#DebugDir}\{#MyAppExeName}"; 											DestDir: "{#ModelsCreatorDir}\{#DebugExeDir}";						Flags: ignoreversion
#sub QtLibsd_entry		
Source: "{#QtPath}\bin\{#QtLibs[I]}d.dll"; 										DestDir: "{#ModelsCreatorDir}\{#DebugExeDir}"; 						Flags: ignoreversion
#endsub	
#for {I = 0; I < DimOf(QtLibs); I++} QtLibsd_entry	
#sub QtLibsPlatformsd_entry	
Source: "{#QtPath}\plugins\{#QtPlatformsDir}\{#QtLibsPlatforms[I]}d.dll"; 		DestDir: "{#ModelsCreatorDir}\{#DebugExeDir}\{#QtPlatformsDir}"; 	Flags: ignoreversion
#endsub
#for {I = 0; I < DimOf(QtLibsPlatforms); I++} QtLibsPlatformsd_entry
#sub QtLibsImageFormatsd_entry
Source: "{#QtPath}\plugins\{#QtImageformatsDir}\{#QtLibsImageFormats[I]}d.dll";	DestDir: "{#ModelsCreatorDir}\{#DebugExeDir}\{#QtImageformatsDir}"; Flags: ignoreversion
#endsub
#for {I = 0; I < DimOf(QtLibsImageFormats); I++} QtLibsImageFormatsd_entry
#sub QtLibsStylesd_entry
Source: "{#QtPath}\plugins\{#QtStylesDir}\{#QtLibsStyles[I]}d.dll";				DestDir: "{#ModelsCreatorDir}\{#DebugExeDir}\{#QtStylesDir}"; 		Flags: ignoreversion
#endsub
#for {I = 0; I < DimOf(QtLibsStyles); I++} QtLibsStylesd_entry

[Dirs]
; Main directory
Name: "{#ModelsCreatorDir}"; 											Flags: uninsalwaysuninstall
		
; Directories for template models
Name: "{#ModelsCreatorDir}\{#TemplatesDir}"; 							Flags: uninsalwaysuninstall
#sub ModelTemplateDirs2_entry
Name: "{#ModelsCreatorDir}\{#TemplatesDir}\{#ModelTemplateDirs[I]}"; 	Flags: uninsalwaysuninstall
#endsub
#for {I = 0; I < DimOf(ModelTemplateDirs); I++} ModelTemplateDirs2_entry

; Directories for example models
Name: "{#ModelsCreatorDir}\{#ExamplesDir}"; 							Flags: uninsalwaysuninstall
#sub ModelExampleDirs2_entry
Name: "{#ModelsCreatorDir}\{#ExamplesDir}\{#ModelExampleDirs[I]}"; 		Flags: uninsalwaysuninstall
#endsub
#for {I = 0; I < DimOf(ModelExampleDirs); I++} ModelExampleDirs2_entry

Name: "{#ModelsCreatorDir}\{#APIDir}"; 									Flags: uninsalwaysuninstall

Name: "{#ModelsCreatorDir}\PropertySheets";								Flags: uninsalwaysuninstall

Name: "{#ModelsCreatorDir}\{#DebugExeDir}"; 							Flags: uninsalwaysuninstall
Name: "{#ModelsCreatorDir}\{#DebugExeDir}\{#QtPlatformsDir}"; 			Flags: uninsalwaysuninstall
Name: "{#ModelsCreatorDir}\{#DebugExeDir}\{#QtImageformatsDir}";		Flags: uninsalwaysuninstall
Name: "{#ModelsCreatorDir}\{#DebugExeDir}\{#QtStylesDir}";				Flags: uninsalwaysuninstall

[Code]
// Replaces all lines that contain Tag with Value
procedure ReplaceLine(const FileName, Tag, Value: string);
var
	I: Integer;
	FileLines: TStringList;
begin
	FileLines := TStringList.Create;
	try
		FileLines.LoadFromFile(FileName);
		for I:=0 to pred(FileLines.Count) do
		begin
			if Pos(Tag, FileLines[I]) <> 0 then
				FileLines[I] := Value;
		end;
	finally
		FileLines.SaveToFile(FileName);
		FileLines.Free;
	end;
end;

// Replace some settings in all *.vcxproj files of models templates
procedure UpdateTemplates();
var
	FilePath: string;
begin
#sub UpdateTemplatesFileEntry
	FilePath := ExpandConstant('{#ModelsCreatorDir}') + '\' + ExpandConstant('{#TemplatesDir}') + '\' + ExpandConstant('{#ModelTemplateDirs[I]}') + '\' + ExpandConstant('{#ModelTemplateDirs[I]}') + '.vcxproj';
	ReplaceLine(FilePath, 'GeneralSources.vcxproj', '    <ProjectReference Include="$(SolutionDir)MUSEN_API\MUSEN_API.vcxproj">');
#endsub
#for {I = 0; I < DimOf(ModelTemplateDirs); I++} UpdateTemplatesFileEntry
end;

// Update some files
procedure ModifyModelsCreatorFiles();
begin
	UpdateTemplates();
end;
