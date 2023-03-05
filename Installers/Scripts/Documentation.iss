// ; Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
// ; This file is part of MUSEN framework http://msolids.net/musen.
// ; See LICENSE file for license and warranty information. 

#define DocsDirSrc 	"Documentation"
#define DocsDirDst 	"Documentation"
#define GuideDir	"Users Guide"; 
#define CMHelpDir	"Models\Contact Models"; 
#define SBHelpDir	"Models\Solid Bond";
#define EFHelpDir	"Models\External Force";

[Files]
Source: "{#SolutionDir}\{#DocsDirSrc}\*.pdf"; 				DestDir: "{app}\{#DocsDirDst}\"; 				Flags: ignoreversion
Source: "{#SolutionDir}\{#DocsDirSrc}\{#GuideDir}\*.pdf"; 	DestDir: "{app}\{#DocsDirDst}\{#GuideDir}"; 	Flags: ignoreversion
Source: "{#SolutionDir}\{#DocsDirSrc}\{#CMHelpDir}\*.pdf"; 	DestDir: "{app}\{#DocsDirDst}\{#CMHelpDir}"; 	Flags: ignoreversion
Source: "{#SolutionDir}\{#DocsDirSrc}\{#SBHelpDir}\*.pdf"; 	DestDir: "{app}\{#DocsDirDst}\{#SBHelpDir}"; 	Flags: ignoreversion
Source: "{#SolutionDir}\{#DocsDirSrc}\{#EFHelpDir}\*.pdf"; 	DestDir: "{app}\{#DocsDirDst}\{#EFHelpDir}"; 	Flags: ignoreversion

[Dirs]
Name: "{app}\{#DocsDirDst}"; 				Flags: uninsalwaysuninstall
Name: "{app}\{#DocsDirDst}\{#GuideDir}"; 	Flags: uninsalwaysuninstall
Name: "{app}\{#DocsDirDst}\{#CMHelpDir}"; 	Flags: uninsalwaysuninstall
Name: "{app}\{#DocsDirDst}\{#SBHelpDir}"; 	Flags: uninsalwaysuninstall
Name: "{app}\{#DocsDirDst}\{#EFHelpDir}"; 	Flags: uninsalwaysuninstall
