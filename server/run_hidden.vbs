' 인자로 받은 .cmd 파일을 콘솔 창 없이 실행한다 (예약 작업용).
' 예약 작업이 cmd.exe 를 직접 띄우면 바탕화면에 터미널 창이 남고, 그 창을 닫으면 서버가 같이 죽는다.
' 사용: wscript.exe run_hidden.vbs <파일.cmd>
CreateObject("WScript.Shell").Run """" & WScript.Arguments(0) & """", 0, False
