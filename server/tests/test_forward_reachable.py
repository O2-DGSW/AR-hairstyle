"""원격(학교 서버)으로 넘겨도 붙을 수 있는 offer 인지 (server._offer_reachable_from_remote).

시연용 폰이 LTE 에서 srflx 없이 host(192.0.0.2 CLAT) + IPv6 만 실어 보내, 학교로 넘어가면
매번 ICE failed(앱에선 타임아웃)였다.
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import _offer_reachable_from_remote                 # noqa: E402


def _body(*cands):
    sdp = "v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\n" + "".join(
        "a=candidate:1 1 udp 2122260223 %s 40019 typ %s\r\n" % c for c in cands)
    return json.dumps({"sdp": sdp, "type": "offer"}).encode()


class TestForwardReachable(unittest.TestCase):
    def test_lte_clat_without_srflx(self):
        # 실제 로그의 SM-S906N offer 모양
        self.assertFalse(_offer_reachable_from_remote(_body(
            ("192.0.0.2", "host"), ("2001:e60:bc19:4b0b:c88b:c0ee:35e4:a4ce", "host"))))

    def test_private_and_mdns_only(self):
        self.assertFalse(_offer_reachable_from_remote(_body(
            ("192.168.0.15", "host"), ("100.111.131.7", "host"),
            ("3f1c2a.local", "host"))))

    def test_srflx(self):
        self.assertTrue(_offer_reachable_from_remote(_body(
            ("192.0.0.4", "host"), ("125.140.125.203", "srflx"))))

    def test_relay(self):
        self.assertTrue(_offer_reachable_from_remote(_body(("119.201.91.129", "relay"))))

    def test_public_ipv4_host(self):
        self.assertTrue(_offer_reachable_from_remote(_body(("119.201.91.129", "host"))))

    def test_unreadable_body_forwards(self):
        self.assertTrue(_offer_reachable_from_remote(b"not json"))


if __name__ == "__main__":
    unittest.main()
