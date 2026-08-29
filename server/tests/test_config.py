"""config.load_config 의 환경변수 파싱과 frozen 계약.

실제 os.environ 은 건드리지 않는다 - load_config(env=dict) 로 격리한다.
"""
import dataclasses
import os
import sys
import unittest

# `python -m unittest discover -s tests` 는 tests/ 를 패키지로 인식하지 않는다.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg


class TestDefaults(unittest.TestCase):
    def test_empty_env_gives_the_declared_defaults(self):
        # 왜 중요: 환경변수를 안 주면 dataclass 기본값 그대로여야 한다.
        c = cfg.load_config(env={})
        d = cfg.Config()
        for f in dataclasses.fields(cfg.Config):
            self.assertEqual(getattr(c, f.name), getattr(d, f.name), f.name)

    def test_unrelated_env_vars_are_ignored(self):
        # 왜 중요: HEDDY_ 접두사가 없는 변수를 주워 담으면 남의 설정에 오염된다.
        c = cfg.load_config(env={"PORT": "1", "MAX_SESSIONS": "99", "HEDDYPORT": "2"})
        self.assertEqual(c.port, cfg.Config().port)
        self.assertEqual(c.max_sessions, cfg.Config().max_sessions)

    def test_unknown_heddy_variables_are_ignored(self):
        # 왜 중요: 오타 난 변수 때문에 서버가 못 뜨면 곤란하다(필드 기준으로만 읽는다).
        c = cfg.load_config(env={"HEDDY_NOT_A_FIELD": "x"})
        self.assertEqual(c.port, cfg.Config().port)

    def test_load_config_does_not_read_or_touch_os_environ_when_env_given(self):
        # 왜 중요: 테스트가 실제 프로세스 환경을 오염시키면 다른 테스트에 샌다.
        snapshot = dict(os.environ)
        cfg.load_config(env={"HEDDY_PORT": "9999"})
        self.assertEqual(dict(os.environ), snapshot)

    def test_module_level_config_singleton_exists(self):
        # 왜 중요: 모든 모듈이 `from config import CONFIG` 로 이 인스턴스를 공유한다.
        self.assertIsInstance(cfg.CONFIG, cfg.Config)


class TestCoercion(unittest.TestCase):
    def test_int_field_is_parsed_as_int(self):
        c = cfg.load_config(env={"HEDDY_PORT": "9999", "HEDDY_MAX_SESSIONS": "7"})
        self.assertEqual(c.port, 9999)
        self.assertIsInstance(c.port, int)
        self.assertEqual(c.max_sessions, 7)

    def test_int_field_accepts_negative_and_whitespace(self):
        c = cfg.load_config(env={"HEDDY_SHADOW_K": " -3 "})
        self.assertEqual(c.shadow_k, -3)

    def test_float_field_is_parsed_as_float(self):
        c = cfg.load_config(env={"HEDDY_SOFT_K": "2.5", "HEDDY_CAL_ALPHA": "1e-3"})
        self.assertIsInstance(c.soft_k, float)
        self.assertAlmostEqual(c.soft_k, 2.5)
        self.assertAlmostEqual(c.cal_alpha, 0.001)

    def test_float_field_accepts_an_integer_literal(self):
        # 왜 중요: HEDDY_SOFT_K=2 처럼 쓰는 게 자연스럽다. int 로 굳으면 나중에
        # 정수 나눗셈 같은 미묘한 차이가 생긴다.
        c = cfg.load_config(env={"HEDDY_SOFT_K": "2"})
        self.assertIsInstance(c.soft_k, float)
        self.assertAlmostEqual(c.soft_k, 2.0)

    def test_bool_field_accepts_the_documented_true_spellings(self):
        # 왜 중요: docstring 이 1/true/yes/on 을 참으로 본다고 약속한다(대소문자 무관).
        for raw in ("1", "true", "TRUE", "True", "yes", "YES", "on", "ON", " on "):
            c = cfg.load_config(env={"HEDDY_USE_CUDA_GRAPH": raw})
            self.assertIs(c.use_cuda_graph, True, raw)

    def test_bool_field_accepts_the_documented_false_spellings(self):
        # 왜 중요: 끄는 쪽도 명시적으로 쓸 수 있어야 한다.
        for raw in ("0", "false", "FALSE", "off", "no", "n", " f "):
            c = cfg.load_config(env={"HEDDY_USE_CUDA_GRAPH": raw})
            self.assertIs(c.use_cuda_graph, False, raw)

    def test_bool_typo_raises_instead_of_silently_disabling(self):
        # 왜 중요: 실제로 났던 문제. 예전 구현은 `raw in _TRUE` 한 줄이라 오타가
        # 조용히 False 가 됐다. bool 필드는 대부분 **기본이 꺼짐인 안전장치**를
        # 켜는 데 쓰이므로(allow_frame_recording 처럼), 오타 하나면 "켰다고
        # 생각했는데 안 켜진 채로 도는" 실패 모드가 되고 로그에도 안 남는다.
        # 이제 int/float/tuple 과 같이 시작할 때 죽는다.
        for raw in ("ture", "enabled", "2", ""):
            with self.assertRaises(ValueError, msg=raw):
                cfg.load_config(env={"HEDDY_ALLOW_FRAME_RECORDING": raw})

    def test_bool_error_message_names_the_variable_and_accepted_values(self):
        # 왜 중요: 오타를 냈을 때 무엇을 써야 하는지 메시지만 보고 알 수 있어야 한다.
        with self.assertRaises(ValueError) as ctx:
            cfg.load_config(env={"HEDDY_ALLOW_FRAME_RECORDING": "ture"})
        msg = str(ctx.exception)
        self.assertIn("HEDDY_ALLOW_FRAME_RECORDING", msg)
        self.assertIn("ture", msg)

    def test_str_field_is_passed_through_verbatim(self):
        # 왜 중요: 모델 리비전 해시 같은 값은 어떤 정규화도 하면 안 된다.
        c = cfg.load_config(env={"HEDDY_GAN_BACKEND": "thread",
                                 "HEDDY_MODEL_REVISION": "  abc123  "})
        self.assertEqual(c.gan_backend, "thread")
        self.assertEqual(c.model_revision, "  abc123  ")

    def test_tuple_field_is_split_on_commas_into_floats(self):
        c = cfg.load_config(env={"HEDDY_LIVE_TARGETS": "-30,0,30"})
        self.assertIsInstance(c.live_targets, tuple)
        self.assertEqual(c.live_targets, (-30.0, 0.0, 30.0))
        for v in c.live_targets:
            self.assertIsInstance(v, float)

    def test_tuple_field_ignores_blank_entries(self):
        # 왜 중요: "-30,0,30," 처럼 꼬리 쉼표가 붙기 쉽다.
        c = cfg.load_config(env={"HEDDY_LIVE_TARGETS": "-30, 0 ,30,"})
        self.assertEqual(c.live_targets, (-30.0, 0.0, 30.0))

    def test_tuple_field_accepts_a_single_value(self):
        c = cfg.load_config(env={"HEDDY_LIVE_TARGETS": "0"})
        self.assertEqual(c.live_targets, (0.0,))

    def test_multiple_overrides_apply_together(self):
        # 왜 중요: 실제 사용은 `HEDDY_MAX_SESSIONS=3 HEDDY_GAN_BACKEND=thread ...` 다.
        c = cfg.load_config(env={"HEDDY_MAX_SESSIONS": "3",
                                 "HEDDY_GAN_BACKEND": "thread",
                                 "HEDDY_METRICS_ENABLED": "0",
                                 "HEDDY_LIVE_TOL": "4.5"})
        self.assertEqual(c.max_sessions, 3)
        self.assertEqual(c.gan_backend, "thread")
        self.assertIs(c.metrics_enabled, False)
        self.assertAlmostEqual(c.live_tol, 4.5)
        self.assertEqual(c.port, cfg.Config().port)     # 나머지는 기본값 유지


class TestInvalidValues(unittest.TestCase):
    def test_non_numeric_int_raises_value_error(self):
        # 왜 중요: docstring 의 약속 - 오타 난 튜닝 값으로 몇 시간 실험하는 것보다
        # 시작할 때 죽는 쪽이 낫다.
        with self.assertRaises(ValueError):
            cfg.load_config(env={"HEDDY_PORT": "eighty"})

    def test_float_literal_in_an_int_field_raises_value_error(self):
        # 왜 중요: HEDDY_MAX_SESSIONS=2.5 는 조용히 2 가 되면 안 된다.
        with self.assertRaises(ValueError):
            cfg.load_config(env={"HEDDY_MAX_SESSIONS": "2.5"})

    def test_non_numeric_float_raises_value_error(self):
        with self.assertRaises(ValueError):
            cfg.load_config(env={"HEDDY_SOFT_K": "soft"})

    def test_bad_tuple_element_raises_value_error(self):
        with self.assertRaises(ValueError):
            cfg.load_config(env={"HEDDY_LIVE_TARGETS": "-30,abc,30"})

    def test_the_error_message_names_the_variable_and_the_value(self):
        # 왜 중요: 이 예외는 서버 기동 실패 로그의 유일한 단서다.
        with self.assertRaises(ValueError) as ctx:
            cfg.load_config(env={"HEDDY_PORT": "eighty"})
        msg = str(ctx.exception)
        self.assertIn("HEDDY_PORT", msg)
        self.assertIn("eighty", msg)


class TestFrozen(unittest.TestCase):
    def test_config_instances_are_immutable(self):
        # 왜 중요: 런타임에 값이 바뀌면 스레드 사이에서 값이 갈리고, CUDA 그래프
        # 캡처 크기 같은 것이 어긋난다(docstring 의 명시적 규칙).
        c = cfg.load_config(env={})
        with self.assertRaises(dataclasses.FrozenInstanceError):
            c.port = 1234
        with self.assertRaises(dataclasses.FrozenInstanceError):
            c.input_size = 256

    def test_the_shared_singleton_is_immutable_too(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            cfg.CONFIG.max_sessions = 99

    def test_new_attributes_cannot_be_added(self):
        # 왜 중요: 오타로 새 필드를 만들어 놓고 설정을 바꿨다고 착각하면 안 된다.
        c = cfg.load_config(env={})
        with self.assertRaises(dataclasses.FrozenInstanceError):
            c.definitely_not_a_field = 1

    def test_input_size_stays_pinned_for_the_cuda_graph(self):
        # 왜 중요: CUDA 그래프가 이 크기로 캡처된다. 조용히 바뀌면 재캡처 없이
        # 크기가 어긋나 터진다(로짓은 이것의 1/4 해상도).
        self.assertEqual(cfg.Config().input_size, 512)


if __name__ == "__main__":
    unittest.main()
