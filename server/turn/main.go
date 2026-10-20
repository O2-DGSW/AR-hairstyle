// heddy TURN 서버 (pion/turn).
//
// 왜 필요한가
// -----------
// 학교 GPU 서버는 도커(172.17.x) + 학교 NAT 뒤에 있어서, STUN 으로 공인 주소를 알아내도
// 외부 브라우저가 보낸 패킷이 들어가지 못한다(ICE failed). 학교 서버가 이 TURN 에서
// 릴레이 주소를 할당받아 answer 에 넣으면, 브라우저는 그 공인 주소(이 PC)로 보내고
// TURN 이 학교 서버로 넘겨준다. 학교 서버 -> 이 PC 3478 은 바깥으로 나가는 연결이라 열려 있다.
//
// 이 PC 는 랜카드에 공인 IP 가 직접 붙어 있어서 공유기 포트포워딩 없이 Windows 방화벽만
// 열면 된다(UDP/TCP 3478, UDP 릴레이 포트 범위).
//
// coturn 이 아닌 이유: Windows 빌드가 없어 WSL2 mirrored 네트워킹 + VM 상시 구동이 필요하다.
//
// 인증은 장기 자격(사용자/비밀번호) 하나. 값은 server/.env 의 HEDDY_TURN_USERNAME /
// HEDDY_TURN_CREDENTIAL - heddy 서버(aiortc)도 같은 키를 읽는다. 브라우저에는 주지 않는다:
// 릴레이 후보는 서버 쪽에만 있으면 되고, 브라우저는 그 공인 주소로 그냥 보내면 된다.
//
// 열린 릴레이가 이 PC 의 내부(루프백/LAN/Tailscale)로 들어가는 통로가 되지 않도록,
// 사설/루프백/링크로컬/CGNAT(100.64/10 = tailnet) 피어로의 퍼미션은 거절한다.
//
// 빌드: cd server/turn && go build -o turn.exe .
// 실행: turn.exe -public-ip <이 PC 공인 IPv4> [-env ..\.env]  (start_turn.cmd 는 .env 의 HEDDY_TURN_PUBLIC_IP 를 넘긴다)
package main

import (
	"bufio"
	"flag"
	"log"
	"net"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"

	"github.com/pion/turn/v4"
)

// readEnv 는 server/.env 의 KEY=VALUE 줄을 읽는다 (config.read_env_file 과 같은 규칙의 축소판).
func readEnv(path string) map[string]string {
	out := map[string]string{}
	f, err := os.Open(path)
	if err != nil {
		return out
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(strings.TrimPrefix(sc.Text(), "\xef\xbb\xbf"))
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		line = strings.TrimPrefix(line, "export ")
		k, v, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		v = strings.TrimSpace(v)
		if len(v) >= 2 && v[0] == v[len(v)-1] && (v[0] == '"' || v[0] == '\'') {
			v = v[1 : len(v)-1]
		}
		out[strings.TrimSpace(k)] = v
	}
	return out
}

var cgnat = &net.IPNet{IP: net.IPv4(100, 64, 0, 0), Mask: net.CIDRMask(10, 32)}

// publicPeer 는 릴레이로 보내도 되는 주소인지 본다.
func publicPeer(ip net.IP) bool {
	return !(ip.IsLoopback() || ip.IsPrivate() || ip.IsLinkLocalUnicast() ||
		ip.IsUnspecified() || ip.IsMulticast() || cgnat.Contains(ip))
}

func main() {
	publicIP := flag.String("public-ip", "", "릴레이 주소로 알려줄 이 PC 의 공인 IPv4 (필수)")
	port := flag.Int("port", 3478, "UDP/TCP 리슨 포트")
	minPort := flag.Int("min-port", 49160, "릴레이 포트 범위 시작 (방화벽과 맞출 것)")
	maxPort := flag.Int("max-port", 49259, "릴레이 포트 범위 끝")
	realm := flag.String("realm", "heddy", "TURN realm")
	envPath := flag.String("env", "..\\.env", "HEDDY_TURN_USERNAME / HEDDY_TURN_CREDENTIAL 을 읽을 파일")
	flag.Parse()

	env := readEnv(*envPath)
	user, pass := env["HEDDY_TURN_USERNAME"], env["HEDDY_TURN_CREDENTIAL"]
	if v := os.Getenv("HEDDY_TURN_USERNAME"); v != "" {
		user = v
	}
	if v := os.Getenv("HEDDY_TURN_CREDENTIAL"); v != "" {
		pass = v
	}
	relayIP := net.ParseIP(*publicIP).To4()
	if relayIP == nil || user == "" || pass == "" {
		log.Fatalf("-public-ip(IPv4) 와 HEDDY_TURN_USERNAME/HEDDY_TURN_CREDENTIAL(%s) 이 필요합니다", *envPath)
	}
	key := turn.GenerateAuthKey(user, *realm, pass)

	addr := net.JoinHostPort("0.0.0.0", strconv.Itoa(*port))
	udp, err := net.ListenPacket("udp4", addr)
	if err != nil {
		log.Fatalf("UDP %s: %v", addr, err)
	}
	tcp, err := net.Listen("tcp4", addr)
	if err != nil {
		log.Fatalf("TCP %s: %v", addr, err)
	}

	relay := func() turn.RelayAddressGenerator {
		return &turn.RelayAddressGeneratorPortRange{
			RelayAddress: relayIP,
			Address:      "0.0.0.0",
			MinPort:      uint16(*minPort),
			MaxPort:      uint16(*maxPort),
		}
	}
	permit := func(client net.Addr, peer net.IP) bool {
		if !publicPeer(peer) {
			log.Printf("거절: %s -> %s (내부 주소)", client, peer)
			return false
		}
		return true
	}

	s, err := turn.NewServer(turn.ServerConfig{
		Realm: *realm,
		AuthHandler: func(username, realm string, src net.Addr) ([]byte, bool) {
			if username != user {
				log.Printf("인증 실패: %q from %s", username, src)
				return nil, false
			}
			return key, true
		},
		PacketConnConfigs: []turn.PacketConnConfig{
			{PacketConn: udp, RelayAddressGenerator: relay(), PermissionHandler: permit},
		},
		ListenerConfigs: []turn.ListenerConfig{
			{Listener: tcp, RelayAddressGenerator: relay(), PermissionHandler: permit},
		},
	})
	if err != nil {
		log.Fatalf("TURN 서버 시작 실패: %v", err)
	}
	log.Printf("TURN 시작: %s (udp+tcp), 릴레이 %s:%d-%d, realm=%s, user=%s",
		addr, relayIP, *minPort, *maxPort, *realm, user)

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	<-sig
	if err := s.Close(); err != nil {
		log.Printf("종료 중 오류: %v", err)
	}
}
