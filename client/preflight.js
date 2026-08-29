/* 웹캠이 안 켜질 때 원인을 페이지에서 바로 알려준다.
 *
 * 두 가지가 "조용히" 실패해서 증상이 똑같아 보인다.
 *   1) HTTPS 가 아니면 navigator.mediaDevices 가 아예 없다.
 *      버튼은 눌리는데 getUserMedia 에서 예외만 난다.
 *   2) CDN(import) 이 막히면 모듈이 통째로 안 올라간다.
 *      이 경우 [시작] 에 핸들러조차 안 붙어서 눌러도 아무 일이 없다.
 *
 * 콘솔을 열지 않고도 구분되도록 #status 에 적는다. 메인 스크립트보다
 * 먼저(단, DOM 뒤에서) 실행되어야 하므로 body 끝, 모듈 태그 앞에 둔다.
 */
(function () {
  var READY_MS = 9000;
  var box = document.getElementById("status");

  function warn(html) {
    if (!box) return;
    box.innerHTML = html;
    box.style.color = "#ff5f56";
  }

  // 1) 보안 컨텍스트. localhost 는 예외적으로 허용된다.
  if (!window.isSecureContext || !navigator.mediaDevices) {
    warn(
      "<b>이 주소에서는 웹캠을 쓸 수 없습니다</b> (" + location.origin + ")<br>" +
      "브라우저는 HTTPS 또는 localhost 에서만 카메라를 허용합니다. 평문 HTTP 라서 " +
      "navigator.mediaDevices 가 존재하지 않습니다.<br><br>" +
      "임시 우회 — Chrome 주소창에 <b>chrome://flags/#unsafely-treat-insecure-origin-as-secure</b> " +
      "를 열고 <b>" + location.origin + "</b> 을 추가한 뒤 브라우저를 재시작하세요."
    );
    return;
  }

  // 2) 메인 스크립트가 살아서 끝까지 갔는지. 실패는 error 이벤트로 오지만,
  //    CDN 이 응답 없이 매달리는 경우엔 이벤트가 안 오므로 시간으로도 본다.
  var main = document.querySelector('script[type="module"][src]') ||
             document.querySelector('script[src$="client.js"]');
  if (!main) return;

  main.addEventListener("error", function () {
    warn("<b>스크립트 로딩 실패</b>: " + main.getAttribute("src") +
         " 또는 그 의존 모듈을 가져오지 못했습니다. (three.js / MediaPipe 는 " +
         "cdn.jsdelivr.net 에서 받습니다)");
  });

  setTimeout(function () {
    if (window.__pageReady) return;
    warn("<b>스크립트가 " + READY_MS / 1000 + "초 안에 초기화되지 않았습니다.</b><br>" +
         "cdn.jsdelivr.net 접근이 막혔을 가능성이 큽니다. 콘솔의 첫 빨간 줄을 확인하세요.");
  }, READY_MS);
})();
