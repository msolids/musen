; Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
; This file is part of MUSEN framework http://msolids.net/musen.
; See LICENSE file for license and warranty information. 

#define public SolutionDir 	"..\.."
#define public ProjectDir 	"..\InstallerProject"
#define public DataDir 		SolutionDir+"\Installers\Data"
#define public ReleaseDir 	SolutionDir+"\x64\Release"
#define public DebugDir 	SolutionDir+"\x64\Debug"

#define MyAppName 			"MUSEN"
#define MyAppExeName 		MyAppName+".exe"
#define MyAppVersion 		GetStringFileInfo(ReleaseDir+'\'+MyAppExeName, "ProductVersion")
#define MyAppBranch 		ReadIni(ProjectDir+"\data.ini", "Version", "Branch", "unknown")
#define MyAppPublisher 		"MUSEN Development Team"
#define MyAppURL 			"https://msolids.net/musen/"
#define MyAppPublisherURL 	"https://msolids.net/musen/"
#define MyAppContact 		"dosta@tuhh.de"

#include "MainFiles.iss"
#include "ConfigINI.iss"
#include "QtLibs.iss"
#include "ModelsCreator.iss"
#include "Examples.iss"
#include "Documentation.iss"

[Setup]
AppId={{5391D901-224F-485B-AEC9-6463835BB0F6}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppPublisherURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={commonpf}\{#MyAppName}
DefaultGroupName={#MyAppName}
UninstallDisplayIcon={app}\{#MyAppExeName}
AppContact={#MyAppContact}
OutputBaseFilename={#MyAppName} {#MyAppVersion} {#MyAppBranch} Setup
OutputDir={#SolutionDir}\Installers\Installers
LicenseFile={#SolutionDir}\LICENSE
SetupIconFile={#SolutionDir}\MusenGUI\Resources\MUSEN_Ico.ico
WizardImageFile={#DataDir}\WizardImageFile.bmp
WizardSmallImageFile={#DataDir}\WizardSmallImageFile.bmp
ArchitecturesInstallIn64BitMode=x64
ArchitecturesAllowed=x64
DisableStartupPrompt=no
DisableWelcomePage=no
SolidCompression=yes
Compression=lzma
InternalCompressLevel=max
AllowNoIcons=yes
ChangesAssociations=yes
ShowLanguageDialog=auto
PrivilegesRequired=poweruser
UsedUserAreasWarning=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Icons]
Name: "{group}\{#MyAppName}"; 							Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:ProgramOnTheWeb,{#MyAppName}}"; 		Filename: "{#MyAppURL}"
Name: "{group}\{cm:UninstallProgram, {#MyAppName}}"; 	Filename: "{uninstallexe}"
Name: "{group}\{#MyAppName} Uninstall"; 				Filename: "{uninstallexe}"
Name: "{commondesktop}\{#MyAppName}"; 					Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Registry]
Root: "HKLM"; Subkey: "Software\Classes\.mdem"; 								ValueType: string; ValueData: "MUSEN simulation"; Flags: uninsdeletevalue
Root: "HKLM"; Subkey: "Software\Classes\MUSEN simulation"; 						ValueType: string; ValueData: "MUSEN simulation"; Flags: uninsdeletekey
Root: "HKLM"; Subkey: "Software\Classes\MUSEN simulation\DefaultIcon"; 			ValueType: string; ValueData: "{app}\{#MyAppExeName},0"
Root: "HKLM"; Subkey: "Software\Classes\MUSEN simulation\shell\open\command"; 	ValueType: string; ValueData: """{app}\{#MyAppExeName}"" ""%1"""

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: dirifempty; Name: "{app}"

[Code]
procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    ModifyModelsCreatorFiles();
  end;
end;