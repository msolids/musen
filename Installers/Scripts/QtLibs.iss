// ; Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
// ; This file is part of MUSEN framework http://msolids.net/musen.
// ; See LICENSE file for license and warranty information. 

#define QtPlatformsDir 		"platforms"
#define QtImageformatsDir 	"imageformats"
#define QtStylesDir 		"styles"

#dim QtLibs[4]
#define QtLibs[0] "Qt5Core"
#define QtLibs[1] "Qt5Gui"
#define QtLibs[2] "Qt5OpenGL"
#define QtLibs[3] "Qt5Widgets"

#dim QtLibsPlatforms[1]
#define QtLibsPlatforms[0] "qwindows"

#dim QtLibsImageFormats[1]
#define QtLibsImageFormats[0] "qjpeg"

#dim QtLibsStyles[1]
#define QtLibsStyles[0] "qwindowsvistastyle"

#define I

[Files]
#sub QtLibs_entry
Source: "{#QtPath}\bin\{#QtLibs[I]}.dll";  										DestDir: "{app}"; 						Flags: ignoreversion
#endsub
#for {I = 0; I < DimOf(QtLibs); I++} QtLibs_entry

#sub QtLibsPlatforms_entry
Source: "{#QtPath}\plugins\{#QtPlatformsDir}\{#QtLibsPlatforms[I]}.dll";  		DestDir: "{app}\{#QtPlatformsDir}"; 	Flags: ignoreversion
#endsub
#for {I = 0; I < DimOf(QtLibsPlatforms); I++} QtLibsPlatforms_entry

#sub QtLibsImageFormats_entry
Source: "{#QtPath}\plugins\{#QtImageformatsDir}\{#QtLibsImageFormats[I]}.dll"; 	DestDir: "{app}\{#QtImageformatsDir}"; 	Flags: ignoreversion
#endsub
#for {I = 0; I < DimOf(QtLibsImageFormats); I++} QtLibsImageFormats_entry

#sub QtLibsStyles_entry
Source: "{#QtPath}\plugins\{#QtStylesDir}\{#QtLibsStyles[I]}.dll"; 				DestDir: "{app}\{#QtStylesDir}"; 		Flags: ignoreversion
#endsub
#for {I = 0; I < DimOf(QtLibsStyles); I++} QtLibsStyles_entry

[Dirs]
Name: "{app}\{#QtPlatformsDir}"; 	Flags: uninsalwaysuninstall
Name: "{app}\{#QtImageformatsDir}"; Flags: uninsalwaysuninstall
Name: "{app}\{#QtStylesDir}"; 		Flags: uninsalwaysuninstall
