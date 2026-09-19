#!/usr/bin/env bash
# 자체서명 인증서 생성 (server.crt / server.key). 브라우저 웹캠(getUserMedia) 은 HTTPS 에서만 되므로
# 공인 IP/LAN 으로 접속할 땐 이게 필요하다. 사용: bash make_cert.sh [추가 IP 또는 도메인...]
# 유효기간 825일(애플 제한). 브라우저에서 한 번 "안전하지 않음 → 이동" 을 눌러야 한다.
set -e
cd "$(dirname "$0")"
SAN="DNS:localhost,IP:127.0.0.1"
for a in "$@"; do
  if [[ "$a" =~ ^[0-9.]+$ ]]; then SAN="$SAN,IP:$a"; else SAN="$SAN,DNS:$a"; fi
done
MSYS_NO_PATHCONV=1 openssl req -x509 -newkey rsa:2048 -sha256 -days 825 -nodes \
  -keyout server.key -out server.crt -subj "/CN=heddy" \
  -addext "subjectAltName=$SAN" -addext "basicConstraints=CA:FALSE"   -addext "keyUsage=digitalSignature,keyEncipherment" -addext "extendedKeyUsage=serverAuth" 2>/dev/null
# macOS/iOS 는 EKU serverAuth 가 없거나 825일을 넘는 인증서를 "이동" 버튼 없이 거부한다 (Apple 요건)
echo "SAN=$SAN"
openssl x509 -in server.crt -noout -dates
