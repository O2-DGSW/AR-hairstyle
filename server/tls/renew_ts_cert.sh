#!/usr/bin/env bash
# Tailscale 정식 인증서(Let's Encrypt) 받기/갱신 -> ts.crt / ts.key.
# iOS 네이티브 앱은 자체서명을 거부하므로 앱은 https://<머신>.<tailnet>.ts.net:8443 으로 붙는다 (폰에도 Tailscale 필요).
# 90일 만료라 두 달에 한 번쯤 돌리고 서버를 재시작한다 (인증서는 시작 때만 읽는다).
# 전제: tailnet 관리 콘솔 DNS -> HTTPS Certificates 가 켜져 있어야 한다.
# 사용: bash renew_ts_cert.sh [도메인]   (생략하면 이 머신의 MagicDNS 이름)
set -e
cd "$(dirname "$0")"
HOST="$1"
if [ -z "$HOST" ]; then
  HOST=$(tailscale status --json | grep -m1 '"DNSName"' | sed -E 's/.*"DNSName": *"([^"]+)\.".*/\1/')
fi
tailscale cert --cert-file ts.crt --key-file ts.key "$HOST"
openssl x509 -in ts.crt -noout -subject -enddate
