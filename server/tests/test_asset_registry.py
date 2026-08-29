"""hair_asset.AssetRegistry 와 생성 에셋 영속화(save/load/prune).

레지스트리 계약:
  - 정적 에셋은 전 세션 공유 + 읽기 전용 (add() 가 절대 오염시키면 안 된다)
  - add() 로 넣은 것은 세션 스코프, max_session 초과 시 LRU 축출 + on_evict(name)
  - close() 는 세션 에셋을 전부 축출하고 여러 번 불러도 안전(idempotent)
  - default() 는 비어 있어도 예외 없이 None

파일 IO 는 전부 tempfile 안에서만 한다. server/assets, server/assets_generated 는
읽지도 쓰지도 않는다.
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

# `python -m unittest discover -s tests` 는 tests/ 를 패키지로 인식하지 않는다.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import hair_asset as ha


def fake_asset(name, yaw=None, bank=None, size=8, ref_skin=None, scale_adjust=1.0):
    """작은 RGBA 배열로 만든 가짜 에셋. 픽셀 내용은 위치를 알아볼 수 있게만 채운다."""
    # 잡음으로 채운다: PNG 가 압축되지 않아 prune_dir 의 크기 상한을 실제로 시험할 수 있다.
    rgba = np.random.default_rng(abs(hash(name)) % (2 ** 32)).integers(
        0, 256, (size, size, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    rgba[0, 0] = (1, 2, 3, 255)
    return ha.HairAsset(name, rgba, (2, 4), (6, 4), ref_skin=ref_skin,
                        yaw=yaw, bank=bank, scale_adjust=scale_adjust)


def static_dict(*names):
    return {n: fake_asset(n) for n in names}


class TestRegistryLookup(unittest.TestCase):
    def setUp(self):
        self.static = static_dict("procedural-bob", "long-wave")
        self.reg = ha.AssetRegistry(self.static)

    def test_get_finds_a_static_asset(self):
        self.assertIs(self.reg.get("long-wave"), self.static["long-wave"])

    def test_get_returns_none_for_an_unknown_name(self):
        # 왜 중요: 클라이언트가 임의 문자열을 보낸다. 예외가 나면 프레임 루프가 죽는다.
        self.assertIsNone(self.reg.get("nope"))

    def test_getitem_raises_key_error_for_an_unknown_name(self):
        # 왜 중요: dict 호환 경로는 dict 처럼 굴어야 한다(get 은 None, [] 는 KeyError).
        with self.assertRaises(KeyError):
            self.reg["nope"]

    def test_contains_covers_both_static_and_session(self):
        self.reg.add(fake_asset("gen-1"))
        self.assertIn("long-wave", self.reg)
        self.assertIn("gen-1", self.reg)
        self.assertNotIn("nope", self.reg)

    def test_default_returns_a_static_asset(self):
        got = self.reg.default()
        self.assertIsNotNone(got)
        self.assertIn(got.name, self.static)

    def test_default_returns_none_when_empty_instead_of_raising(self):
        # 왜 중요: 예전 코드가 next(iter(...)) 를 써서 에셋 디렉터리가 비면
        # StopIteration 이 프레임 루프 한가운데서 터졌다.
        self.assertIsNone(ha.AssetRegistry({}).default())
        self.assertIsNone(ha.AssetRegistry(None).default())

    def test_default_ignores_session_assets(self):
        # 왜 중요: 기본값은 공유해도 되는 정적 에셋이어야 한다. 세션 생성물이
        # 기본값이 되면 남의 얼굴에서 뽑은 헤어가 초기 상태로 뜬다.
        reg = ha.AssetRegistry({})
        reg.add(fake_asset("gen-1"))
        self.assertIsNone(reg.default())

    def test_names_lists_static_first_then_session(self):
        self.reg.add(fake_asset("gen-1"))
        names = self.reg.names()
        self.assertEqual(names[:2], ["procedural-bob", "long-wave"])
        self.assertIn("gen-1", names)

    def test_names_has_no_duplicates_when_a_session_asset_shadows_a_static_one(self):
        # 왜 중요: 같은 이름이 양쪽에 있으면 UI 목록에 두 번 뜬다.
        self.reg.add(fake_asset("long-wave"))
        names = self.reg.names()
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(names.count("long-wave"), 1)

    def test_a_session_asset_shadows_the_static_one_of_the_same_name(self):
        shadow = fake_asset("long-wave")
        self.reg.add(shadow)
        self.assertIs(self.reg.get("long-wave"), shadow)
        self.assertIs(self.static["long-wave"], self.static["long-wave"])  # 원본은 그대로

    def test_session_names_lists_only_added_assets(self):
        self.reg.add(fake_asset("gen-1"))
        self.reg.add(fake_asset("gen-2"))
        self.assertEqual(sorted(self.reg.session_names()), ["gen-1", "gen-2"])

    def test_banks_returns_sorted_unique_banks_across_both_scopes(self):
        static = {"s": fake_asset("s", yaw=0, bank="zed")}
        reg = ha.AssetRegistry(static)
        reg.add(fake_asset("g1", yaw=0, bank="alpha"))
        reg.add(fake_asset("g2", yaw=12, bank="alpha"))
        reg.add(fake_asset("g3"))                        # bank 없음
        self.assertEqual(reg.banks(), ["alpha", "zed"])

    def test_registry_is_dict_compatible_for_the_yaw_pickers(self):
        # 왜 중요: pick_by_yaw / list_banks 가 .values() 를 쓴다. 레지스트리를
        # 그대로 넘길 수 있어야 오프라인 스크립트와 시그니처가 갈리지 않는다.
        reg = ha.AssetRegistry({})
        for y in (-12, 0, 12):
            reg.add(fake_asset("b%+03d" % y, yaw=y, bank="bob"))
        self.assertEqual(ha.pick_by_yaw(reg, "bob", 5.0).yaw, 0)
        self.assertEqual(ha.list_banks(reg), ["bob"])
        self.assertEqual(len(reg), 3)
        self.assertEqual(sorted(reg.keys()), sorted(n for n, _ in reg.items()))
        self.assertEqual(len(list(iter(reg))), 3)
        self.assertEqual(len(reg.values()), 3)


class TestRegistryIsolation(unittest.TestCase):
    def test_add_never_mutates_the_shared_static_dict(self):
        # 왜 중요: 원래 사고 그 자체. 전역 dict 하나를 공유하면 A 가 만든 에셋
        # (A 의 얼굴/피부톤이 구워진 것)이 B 의 목록에 뜬다.
        static = static_dict("procedural-bob")
        before = dict(static)
        reg = ha.AssetRegistry(static)
        reg.add(fake_asset("A-face"))
        self.assertEqual(static, before)
        self.assertNotIn("A-face", static)

    def test_two_registries_sharing_static_do_not_see_each_others_session(self):
        # 왜 중요: 세션 격리의 관측 가능한 형태.
        static = static_dict("procedural-bob")
        a = ha.AssetRegistry(static)
        b = ha.AssetRegistry(static)
        a.add(fake_asset("A-face"))
        self.assertIn("A-face", a)
        self.assertNotIn("A-face", b)
        self.assertIsNone(b.get("A-face"))
        self.assertIn("procedural-bob", b)      # 정적 쪽은 그대로 공유

    def test_shadowing_a_static_name_does_not_leak_to_the_other_session(self):
        static = static_dict("procedural-bob")
        a = ha.AssetRegistry(static)
        b = ha.AssetRegistry(static)
        a.add(fake_asset("procedural-bob"))
        self.assertIsNot(a.get("procedural-bob"), b.get("procedural-bob"))
        self.assertIs(b.get("procedural-bob"), static["procedural-bob"])


class TestRegistryEviction(unittest.TestCase):
    def test_adding_beyond_max_session_evicts_the_least_recently_used(self):
        # 왜 중요: 상한이 없으면 GAN 생성물이 프로세스 수명 내내 쌓이기만 한다.
        evicted = []
        reg = ha.AssetRegistry({}, max_session=3, on_evict=evicted.append)
        for i in range(5):
            reg.add(fake_asset("g%d" % i))
        self.assertEqual(reg.session_names(), ["g2", "g3", "g4"])
        self.assertEqual(evicted, ["g0", "g1"])
        self.assertNotIn("g0", reg)

    def test_get_refreshes_recency_so_eviction_is_lru_not_fifo(self):
        # 왜 중요: FIFO 면 사용자가 계속 쓰고 있는 에셋이 축출된다.
        evicted = []
        reg = ha.AssetRegistry({}, max_session=3, on_evict=evicted.append)
        for i in range(3):
            reg.add(fake_asset("g%d" % i))
        reg.get("g0")                       # g0 를 최근 사용으로 올린다
        reg.add(fake_asset("g3"))
        self.assertEqual(evicted, ["g1"])
        self.assertIn("g0", reg)

    def test_re_adding_an_existing_name_refreshes_it_without_growing(self):
        # 왜 중요: 같은 이름을 다시 만들면(재생성) 개수가 늘면 안 된다.
        reg = ha.AssetRegistry({}, max_session=3)
        for i in range(3):
            reg.add(fake_asset("g%d" % i))
        reg.add(fake_asset("g0"))
        self.assertEqual(len(reg.session_names()), 3)
        self.assertEqual(reg.session_names()[-1], "g0")     # 가장 최근

    def test_eviction_hook_failures_do_not_abort_the_eviction(self):
        # 왜 중요: GPU 캐시 무효화가 던진다고 세션 정리가 멈추면 나머지가 통째로 남는다.
        seen = []

        def boom(name):
            seen.append(name)
            raise RuntimeError("GPU 캐시 무효화 실패")

        reg = ha.AssetRegistry({}, max_session=1, on_evict=boom)
        reg.add(fake_asset("g0"))
        reg.add(fake_asset("g1"))          # g0 축출 -> 훅이 던진다
        self.assertEqual(seen, ["g0"])
        self.assertEqual(reg.session_names(), ["g1"])

    def test_registry_works_without_an_evict_hook(self):
        reg = ha.AssetRegistry({}, max_session=1)
        reg.add(fake_asset("g0"))
        reg.add(fake_asset("g1"))
        self.assertEqual(reg.session_names(), ["g1"])

    def test_default_max_session_comes_from_config(self):
        # 왜 중요: 상한이 설정 한 곳에서만 정의되어야 한다(모듈 전역으로 되돌리지 않는다).
        from config import CONFIG
        reg = ha.AssetRegistry({})
        for i in range(CONFIG.session_asset_max + 3):
            reg.add(fake_asset("g%d" % i))
        self.assertEqual(len(reg.session_names()), CONFIG.session_asset_max)


class TestRegistryClose(unittest.TestCase):
    def test_close_evicts_every_session_asset(self):
        # 왜 중요: 세션이 끝나면 그 사용자의 생성물이 메모리/GPU 에서 사라져야 한다.
        evicted = []
        reg = ha.AssetRegistry(static_dict("procedural-bob"), on_evict=evicted.append)
        for i in range(3):
            reg.add(fake_asset("g%d" % i))
        reg.close()
        self.assertEqual(reg.session_names(), [])
        self.assertEqual(sorted(evicted), ["g0", "g1", "g2"])
        self.assertNotIn("g0", reg)

    def test_close_is_idempotent(self):
        # 왜 중요: 서버가 정상 종료 경로와 에러 경로 양쪽에서 부른다.
        evicted = []
        reg = ha.AssetRegistry({}, on_evict=evicted.append)
        reg.add(fake_asset("g0"))
        reg.close()
        reg.close()
        reg.close()
        self.assertEqual(evicted, ["g0"])       # 훅이 두 번 불리지 않는다

    def test_close_on_an_empty_registry_is_harmless(self):
        ha.AssetRegistry({}).close()

    def test_static_assets_survive_close(self):
        # 왜 중요: 정적 에셋은 다른 세션이 공유하고 있다. 여기서 지우면 남이 죽는다.
        static = static_dict("procedural-bob")
        reg = ha.AssetRegistry(static)
        reg.add(fake_asset("g0"))
        reg.close()
        self.assertIn("procedural-bob", static)
        self.assertIsNotNone(reg.get("procedural-bob"))


class TestAssetPersistence(unittest.TestCase):
    def setUp(self):
        # 실제 server/assets 나 assets_generated 는 절대 건드리지 않는다.
        self.dir = tempfile.mkdtemp(prefix="heddy-test-assets-")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_save_then_load_round_trips_every_field(self):
        # 왜 중요: 재시작 후 세션 에셋을 되살리는 경로. 앵커나 yaw 가 유실되면
        # 복원된 헤어가 엉뚱한 자리에 붙거나 뱅크에서 빠진다.
        a = fake_asset("gen-abc", yaw=-12.0, bank="live", ref_skin=[10.0, 20.0, 30.0],
                       scale_adjust=0.94)
        ha.save_asset(a, self.dir)
        loaded = ha.load_asset_dir(self.dir)

        self.assertEqual(list(loaded), ["gen-abc"])
        b = loaded["gen-abc"]
        np.testing.assert_allclose(b.eye_l, a.eye_l, atol=1e-4)
        np.testing.assert_allclose(b.eye_r, a.eye_r, atol=1e-4)
        self.assertAlmostEqual(b.yaw, -12.0, places=6)
        self.assertEqual(b.bank, "live")
        self.assertAlmostEqual(b.scale_adjust, 0.94, places=6)
        np.testing.assert_allclose(b.ref_skin, [10.0, 20.0, 30.0], atol=1e-4)
        np.testing.assert_array_equal(b.rgba, a.rgba)      # 알파 포함 무손실이어야 한다

    def test_save_preserves_the_alpha_channel(self):
        # 왜 중요: 알파가 날아가면 헤어가 사각형 판때기로 합성된다.
        a = fake_asset("gen-alpha")
        a.rgba[3:5, 3:5, 3] = 0
        ha.save_asset(a, self.dir)
        b = ha.load_asset_dir(self.dir)["gen-alpha"]
        self.assertEqual(b.rgba.shape[2], 4)
        self.assertEqual(int(b.rgba[3, 3, 3]), 0)
        self.assertEqual(int(b.rgba[0, 0, 3]), 255)

    def test_save_returns_the_png_path_and_writes_a_sidecar_json(self):
        png = ha.save_asset(fake_asset("gen-1"), self.dir)
        self.assertTrue(os.path.isfile(png))
        meta = os.path.splitext(png)[0] + ".json"
        self.assertTrue(os.path.isfile(meta))
        with open(meta, encoding="utf-8") as f:
            d = json.load(f)
        self.assertIn("eyeL", d)
        self.assertIn("eyeR", d)

    def test_save_creates_the_directory(self):
        sub = os.path.join(self.dir, "session-xyz")
        ha.save_asset(fake_asset("gen-1"), sub)
        self.assertTrue(os.path.isdir(sub))

    def test_save_sanitises_names_that_would_escape_the_directory(self):
        # 왜 중요: 에셋 이름이 클라이언트 문자열에서 파생될 수 있다. 그대로 경로에
        # 붙이면 ../ 로 디렉터리를 빠져나가 임의 위치에 파일을 쓴다.
        outside = os.path.dirname(self.dir)
        before = set(os.listdir(outside))
        for evil in ("../escaped", "..\\escaped", "a/b/c", "with space"):
            png = ha.save_asset(fake_asset(evil), self.dir)
            self.assertEqual(os.path.dirname(os.path.abspath(png)),
                             os.path.abspath(self.dir), evil)
        self.assertEqual(set(os.listdir(outside)) - before, set())

    def test_save_rejects_a_non_rgba_asset(self):
        # 왜 중요: 3채널을 저장하면 알파 없는 png 가 되어 복원 시 조용히 버려진다.
        bad = ha.HairAsset("bgr", np.zeros((8, 8, 3), np.uint8), (2, 4), (6, 4))
        with self.assertRaises(ValueError):
            ha.save_asset(bad, self.dir)

    def test_load_asset_dir_on_a_missing_directory_returns_empty(self):
        # 왜 중요: 첫 접속 세션은 디렉터리가 없다. 예외가 나면 세션이 못 뜬다.
        self.assertEqual(ha.load_asset_dir(os.path.join(self.dir, "nope")), {})

    def test_load_asset_dir_skips_a_truncated_sidecar(self):
        # 왜 중요: 저장 도중 프로세스가 죽으면 json 이 잘려 있다. 한 장 때문에
        # 세션 복원 전체를 포기하면 안 된다.
        ha.save_asset(fake_asset("good"), self.dir)
        png = ha.save_asset(fake_asset("broken"), self.dir)
        with open(os.path.splitext(png)[0] + ".json", "w", encoding="utf-8") as f:
            f.write('{"eyeL": [1, 2')          # 잘린 JSON
        loaded = ha.load_asset_dir(self.dir)
        self.assertIn("good", loaded)
        self.assertNotIn("broken", loaded)

    def test_load_asset_dir_ignores_non_png_files(self):
        with open(os.path.join(self.dir, "notes.txt"), "w", encoding="utf-8") as f:
            f.write("hello")
        ha.save_asset(fake_asset("gen-1"), self.dir)
        self.assertEqual(list(ha.load_asset_dir(self.dir)), ["gen-1"])

    def test_load_asset_dir_does_not_inject_the_procedural_default(self):
        # 왜 중요: 기본 에셋은 이미 레지스트리의 정적 쪽에 있다. 여기서도 끼워넣으면
        # 같은 이름이 양쪽에 생겨 목록에 중복으로 보인다.
        ha.save_asset(fake_asset("gen-1"), self.dir)
        self.assertEqual(list(ha.load_asset_dir(self.dir)), ["gen-1"])

    def test_load_asset_dir_does_not_recurse_into_subdirectories(self):
        # 왜 중요: 디렉터리 하나 = 세션 하나. 재귀하면 남의 세션 에셋을 읽는다.
        ha.save_asset(fake_asset("mine"), self.dir)
        ha.save_asset(fake_asset("theirs"), os.path.join(self.dir, "other-session"))
        self.assertEqual(list(ha.load_asset_dir(self.dir)), ["mine"])

    def test_saved_assets_can_be_restored_into_a_session_registry(self):
        # 왜 중요: 실제 사용 경로 전체(저장 -> 복원 -> 세션 스코프)를 한 번 밟는다.
        for i in range(3):
            ha.save_asset(fake_asset("gen-%d" % i, yaw=i * 12.0, bank="live"), self.dir)
        static = static_dict("procedural-bob")
        reg = ha.AssetRegistry(static)
        for a in ha.load_asset_dir(self.dir).values():
            reg.add(a)
        self.assertEqual(sorted(reg.session_names()), ["gen-0", "gen-1", "gen-2"])
        self.assertEqual(reg.banks(), ["live"])
        self.assertNotIn("gen-0", static)       # 정적 쪽은 여전히 깨끗하다


class TestPruneDir(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="heddy-test-prune-")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def make(self, session, name, mtime, size=8):
        sub = os.path.join(self.dir, session)
        png = ha.save_asset(fake_asset(name, size=size), sub)
        for p in (png, os.path.splitext(png)[0] + ".json"):
            os.utime(p, (mtime, mtime))
        return png

    @staticmethod
    def pair_bytes(png):
        return (os.path.getsize(png)
                + os.path.getsize(os.path.splitext(png)[0] + ".json"))

    def test_prune_removes_nothing_when_under_the_limit(self):
        self.make("s1", "a", 1_000_000)
        self.assertEqual(ha.prune_dir(self.dir, max_mb=1024), 0)
        self.assertEqual(list(ha.load_asset_dir(os.path.join(self.dir, "s1"))), ["a"])

    def test_prune_removes_oldest_first(self):
        # 왜 중요: 상한을 넘으면 오래된 것부터 지워야 한다. 최신부터 지우면
        # 방금 만든 헤어가 사라진다. 상한은 MB 단위 정수라(int(max_mb)) 실제로
        # 한 개만 남는 경계가 생기도록 에셋을 충분히 크게 만든다.
        old_png = self.make("s1", "old", 1_000_000, size=600)
        new_png = self.make("s1", "new", 2_000_000, size=600)
        mb = 1024 * 1024
        one, total = self.pair_bytes(new_png), 0
        total = self.pair_bytes(old_png) + one
        limit_mb = -(-one // mb)                  # ceil(MB) - 하나는 반드시 남는다
        self.assertLess(limit_mb * mb, total)     # 둘 다 남길 수는 없다
        self.assertEqual(ha.prune_dir(self.dir, max_mb=limit_mb), 1)
        self.assertFalse(os.path.exists(old_png))
        self.assertTrue(os.path.exists(new_png))
        self.assertEqual(list(ha.load_asset_dir(os.path.join(self.dir, "s1"))), ["new"])

    def test_prune_recurses_into_session_subdirectories(self):
        # 왜 중요: 생성 에셋은 세션별 하위 디렉터리에 쌓인다. 재귀하지 않으면
        # DataChannel 명령 하나로 디스크를 무한히 채울 수 있다.
        self.make("s1", "a", 1_000_000)
        self.make("s2", "b", 1_000_100)
        self.assertEqual(ha.prune_dir(self.dir, max_mb=0), 2)
        self.assertEqual(ha.load_asset_dir(os.path.join(self.dir, "s1")), {})
        self.assertEqual(ha.load_asset_dir(os.path.join(self.dir, "s2")), {})

    def test_prune_removes_the_json_sidecar_along_with_the_png(self):
        # 왜 중요: json 만 남으면 디스크가 계속 차오르고 목록에도 유령이 남는다.
        png = self.make("s1", "a", 1_000_000)
        ha.prune_dir(self.dir, max_mb=0)
        self.assertFalse(os.path.exists(png))
        self.assertFalse(os.path.exists(os.path.splitext(png)[0] + ".json"))

    def test_prune_cleans_up_emptied_session_directories(self):
        # 왜 중요: 안 지우면 세션 수만큼 빈 폴더가 영원히 쌓인다.
        self.make("s1", "a", 1_000_000)
        ha.prune_dir(self.dir, max_mb=0)
        self.assertFalse(os.path.isdir(os.path.join(self.dir, "s1")))
        self.assertTrue(os.path.isdir(self.dir))        # 루트는 남는다

    def test_prune_on_a_missing_directory_returns_zero(self):
        self.assertEqual(ha.prune_dir(os.path.join(self.dir, "nope"), max_mb=1), 0)

    def test_prune_on_an_empty_directory_returns_zero(self):
        self.assertEqual(ha.prune_dir(self.dir, max_mb=0), 0)

    def test_prune_limit_granularity_is_whole_megabytes(self):
        # API 의 실제 한계를 문서화한다: prune_dir 은 limit = int(max_mb) * 1MB 로
        # 계산하므로 1MB 미만의 상한을 표현할 수 없다. 0.5 를 넘기면 0 으로 내려가
        # "절반만 남겨라" 가 아니라 "전부 지워라" 가 된다. 소수 상한이 필요해지면
        # 소스를 고쳐야 하며, 그때 이 테스트가 알려준다.
        self.make("s1", "a", 1_000_000)
        self.make("s2", "b", 1_000_100)
        self.assertEqual(ha.prune_dir(self.dir, max_mb=0.9), 2)   # 0.9MB -> 0MB


if __name__ == "__main__":
    unittest.main()
