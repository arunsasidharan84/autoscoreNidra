; Inno Setup Script for autoscoreNidra
[Setup]
AppId={{C6D29A10-D24E-464A-A91B-6B8F01184F65}
AppName=autoscoreNidra
AppVersion=1.0.0
DefaultDirName={userappdata}\autoscoreNidra
DefaultGroupName=autoscoreNidra
OutputDir=..\dist
OutputBaseFilename=autoscoreNidra-Installer
SetupIconFile=runner\resources\app_icon.ico
Compression=lzma
SolidCompression=yes
WizardStyle=modern
DisableProgramGroupPage=yes
PrivilegesRequired=lowest

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\build\windows\x64\runner\Release\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\autoscoreNidra"; Filename: "{app}\autoscoreNidra.exe"
Name: "{autodesktop}\autoscoreNidra"; Filename: "{app}\autoscoreNidra.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\autoscoreNidra.exe"; Description: "{cm:LaunchProgram,autoscoreNidra}"; Flags: nowait postinstall skipifsilent
