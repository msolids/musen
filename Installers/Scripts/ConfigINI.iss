// ; Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
// ; This file is part of MUSEN framework http://msolids.net/musen.
// ; See LICENSE file for license and warranty information. 

#define INIFileName MyAppName+".ini"

[Files]
Source: "{#DataDir}\{#INIFileName}"; 	DestDir: "{autoappdata}\{#MyAppName}"; Flags: onlyifdoesntexist uninsneveruninstall

[Dirs]
Name: "{autoappdata}\{#MyAppName}"; Flags: uninsneveruninstall

[INI]
Filename: "{autoappdata}\{#MyAppName}\{#INIFileName}"; Section: "General"; Key: "AGGLOMERATES_DATABASE_PATH"; String: "{code:MakeRightSlashes|{#DatabasesDirDst}\MADBFileName}"; Flags: createkeyifdoesntexist
Filename: "{autoappdata}\{#MyAppName}\{#INIFileName}"; Section: "General"; Key: "GEOMETRIES_DATABASE_PATH";   String: "{code:MakeRightSlashes|{#DatabasesDirDst}\MGDBFileName}"; Flags: createkeyifdoesntexist
Filename: "{autoappdata}\{#MyAppName}\{#INIFileName}"; Section: "General"; Key: "MATERIALS_DATABASE_PATH";    String: "{code:MakeRightSlashes|{#DatabasesDirDst}\MMDBFileName}"; Flags: createkeyifdoesntexist

[Code]
function MakeRightSlashes(Value: string): string;
begin
  Result := Value;
  StringChangeEx(Result, '\', '/', True);
end;

