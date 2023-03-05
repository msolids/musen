// ; Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
// ; This file is part of MUSEN framework http://msolids.net/musen.
// ; See LICENSE file for license and warranty information. 

#define MADBFileName 		"Agglomerates.madb"
#define MGDBFileName 		"Geometries.mgdb"
#define MMDBFileName 		"Materials.mdb"
#define DatabasesDirSrc 	DataDir+"\Databases"
#define DatabasesDirDst 	"{app}\Databases"

[Files]
Source: "{#ReleaseDir}\{#MyAppExeName}"; 						DestDir: "{app}"; 										Flags: ignoreversion
Source: "{#ReleaseDir}\CMusen.exe"; 	 						DestDir: "{app}"; 				DestName: "CMUSEN.exe";	Flags: ignoreversion
Source: "{#SolutionDir}\LICENSE"; 								DestDir: "{app}"; 										Flags: ignoreversion
Source: "{#SolutionDir}\MusenGUI\styles\musen_style1.qss"; 	    DestDir: "{app}\styles"; 								Flags: ignoreversion
Source: "{#DatabasesDirSrc}\{#MADBFileName}";					DestDir: "{#DatabasesDirDst}"; 							Flags: ignoreversion
Source: "{#DatabasesDirSrc}\{#MGDBFileName}";					DestDir: "{#DatabasesDirDst}"; 							Flags: ignoreversion
Source: "{#DatabasesDirSrc}\{#MMDBFileName}";					DestDir: "{#DatabasesDirDst}"; 							Flags: ignoreversion
Source: "{#DataDir}\Licenses\*"; 								DestDir: "{app}\Licenses"; 								Flags: ignoreversion
Source: "{#SolutionDir}\LICENSE"; 								DestDir: "{app}\Licenses"; 								Flags: ignoreversion

[Dirs]
Name: "{#DatabasesDirDst}"; 	Flags: uninsalwaysuninstall
Name: "{app}\Licenses"; 		Flags: uninsalwaysuninstall
Name: "{app}\styles"; 			Flags: uninsalwaysuninstall
