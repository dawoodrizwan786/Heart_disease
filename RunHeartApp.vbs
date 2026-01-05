Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "cmd /c streamlit run app.py", 0
Set WshShell = Nothing
