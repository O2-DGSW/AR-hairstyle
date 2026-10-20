@echo off
rem 예약 작업 heddy-turn 이 ..\run_hidden.vbs 로 창 없이 띄운다. 출력은 %TEMP%\heddy-turn.log
rem 공인 IP 는 코드에 두지 않는다 - server/.env 의 HEDDY_TURN_PUBLIC_IP 에서 읽는다.
cd /d "%~dp0"
set "PUBIP="
for /f "usebackq tokens=1,* delims==" %%a in ("..\.env") do if "%%a"=="HEDDY_TURN_PUBLIC_IP" set "PUBIP=%%b"
if not defined PUBIP (
  echo HEDDY_TURN_PUBLIC_IP missing in ..\.env > "%TEMP%\heddy-turn.log"
  exit /b 1
)
"turn.exe" -public-ip %PUBIP% -env ..\.env > "%TEMP%\heddy-turn.log" 2>&1
