' ==============================================================================
' Local Jarvis - Silent Background Startup Launcher (VBScript)
'
' Runs Local Jarvis silently in the background on Windows without opening
' any visible Command Prompt or PowerShell console window.
' The Jarvis system tray icon will appear in the Windows taskbar notification area.
'
' HOW TO AUTO-START ON WINDOWS LOGIN (shell:startup):
' ------------------------------------------------------------------------------
' 1. Press [Win + R] on your keyboard to open the Windows "Run" dialog.
' 2. Type "shell:startup" (without quotes) and press Enter.
'    This opens your personal Windows Startup folder:
'    C:\Users\<Username>\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup
' 3. Right-click on this "start_jarvis.vbs" file -> Select "Show more options" -> "Create shortcut".
' 4. Cut and paste (or drag) the created shortcut into the Startup folder.
' 5. That's it! Local Jarvis will now start automatically in the background
'    every time you log in to Windows, ready for hands-free "Hey Jarvis" queries.
'
' HOW TO SHUT DOWN:
' ------------------------------------------------------------------------------
' Right-click the Local Jarvis tray icon in your system tray and click "Quit Jarvis".
' ==============================================================================

Option Explicit

Dim objFSO, objShell, strScriptDir, strBatPath, strLogPath, strCommand

Set objFSO = CreateObject("Scripting.FileSystemObject")
Set objShell = CreateObject("WScript.Shell")

' Get project directory from script path
strScriptDir = objFSO.GetParentFolderName(WScript.ScriptFullName)
strBatPath = strScriptDir & "\start_jarvis.bat"
strLogPath = strScriptDir & "\jarvis_startup.log"

' Set working directory to project root
objShell.CurrentDirectory = strScriptDir

' Construct hidden command redirecting output to log file
strCommand = "%comspec% /c """"" & strBatPath & """ > """ & strLogPath & """ 2>&1"""

' 0 = Hide window (completely silent background execution), False = Do not wait for script to finish
objShell.Run strCommand, 0, False

Set objShell = Nothing
Set objFSO = Nothing
