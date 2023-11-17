// ; Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
// ; This file is part of MUSEN framework http://msolids.net/musen.
// ; See LICENSE file for license and warranty information. 

#define ExamplesDir 		"Examples"
#define ExampleScenesDir 	"InitScenes"
#define ExampleScriptsDir 	"Scripts"

[Files]
Source: "{#DataDir}\{#ExamplesDir}\{#ExampleScenesDir}\*.mdem"; DestDir: "{app}\{#ExamplesDir}\{#ExampleScenesDir}";  Flags: ignoreversion
Source: "{#DataDir}\{#ExamplesDir}\{#ExampleScenesDir}\*.txt"; 	DestDir: "{app}\{#ExamplesDir}\{#ExampleScenesDir}";  Flags: ignoreversion
Source: "{#DataDir}\{#ExamplesDir}\{#ExampleScriptsDir}\*.txt"; DestDir: "{app}\{#ExamplesDir}\{#ExampleScriptsDir}"; Flags: ignoreversion
Source: "{#DataDir}\{#ExamplesDir}\{#ExampleScriptsDir}\*.bat"; DestDir: "{app}\{#ExamplesDir}\{#ExampleScriptsDir}"; Flags: ignoreversion

[Dirs]
Name: "{app}\{#ExamplesDir}"; 						Flags: uninsalwaysuninstall
Name: "{app}\{#ExamplesDir}\{#ExampleScenesDir}"; 	Flags: uninsalwaysuninstall
Name: "{app}\{#ExamplesDir}\{#ExampleScriptsDir}"; 	Flags: uninsalwaysuninstall
