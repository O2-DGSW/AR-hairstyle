@echo off
rem 예약 작업 heddy-supervise 가 run_hidden.vbs 로 창 없이 띄운다. 출력은 %TEMP%\heddy-supervise.out
cd /d "%~dp0.."
".venv\Scripts\python.exe" -u "server\supervise.py" > "%TEMP%\heddy-supervise.out" 2>&1
