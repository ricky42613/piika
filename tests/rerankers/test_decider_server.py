import copy
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "rerankers"))
from decider_server import (
    DEFAULT_MODEL, BatchingDecider, ServiceUnavailable,
    create_server, decide_request, validate_request,
)


class FakeRequestError(ValueError):
    pass


class FakeDecider:
    def __init__(self):
        self.calls = []
        self.thread_ids = []
        self.error = None
        self.result = {
            "answers": {"relevance": {"type": "noul", "noul": 0.875}},
            "usage": {"input_tokens": 123, "output_tokens": 0, "sequences": 1},
        }

    def decide(self, request):
        self.calls.append(request)
        self.thread_ids.append(threading.get_ident())
        if self.error:
            raise self.error
        return [self.result for _ in request] if isinstance(request, list) else self.result


def request_body():
    return {
        "model": DEFAULT_MODEL,
        "state": "Charles Babbage designed the analytical engine.",
        "questions": {
            "relevance": {"type": "noul", "instructions": "Does this identify the designer?"},
        },
    }


class DecisionsTests(unittest.TestCase):
    def test_native_probability_and_usage_are_preserved_without_recalibration(self):
        decider = FakeDecider()
        payload = request_body()
        original = copy.deepcopy(payload)
        response = decide_request(decider, payload)
        self.assertEqual(decider.calls, [{
            "state": payload["state"], "questions": payload["questions"],
        }])
        self.assertEqual(response, {
            "model": DEFAULT_MODEL,
            "answers": {"relevance": {"type": "noul", "noul": 0.875}},
            "usage": {"input_tokens": 123, "output_tokens": 0},
        })
        self.assertEqual(payload, original)

    def test_question_names_and_multiline_text_are_forwarded(self):
        payload = request_body()
        payload["state"] = "第一行\n第二行"
        payload["questions"] = {"custom-id": {"type": "noul", "instructions": "Relevant\nto the engine?"}}
        decider = FakeDecider()
        decider.result["answers"] = {"custom-id": {"type": "noul", "noul": 0.1}}
        response = decide_request(decider, payload)
        self.assertEqual(decider.calls[0]["state"], payload["state"])
        self.assertIn("custom-id", response["answers"])

    def test_criteria_only_noul_is_supported(self):
        payload = request_body()
        payload["questions"] = {"q": {"type": "noul", "criteria": {
            "true": "Identifies the engine designer", "false": "Unrelated",
        }}}
        validate_request(payload, DEFAULT_MODEL)

    def test_invalid_or_unsupported_requests_fail_before_inference(self):
        payloads = [None, [], {},
                    {**request_body(), "model": "wrong"},
                    {**request_body(), "state": {}},
                    {**request_body(), "candidates": []},
                    {**request_body(), "questions": {}},
                    {**request_body(), "questions": {"a": {}, "b": {}}}]
        for question in [None, {"type": "choice"}, {"type": "noul"},
                         {"type": "noul", "instructions": 123},
                         {"type": "noul", "criteria": {"yes": "Y"}},
                         {"type": "noul", "criteria": {"true": 1}},
                         {"type": "noul", "instructions": "Q?", "extra": True}]:
            payloads.append({**request_body(), "questions": {"q": question}})
        decider = FakeDecider()
        for payload in payloads:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                decide_request(decider, payload)
        self.assertEqual(decider.calls, [])


class BatchingTests(unittest.TestCase):
    def batcher(self, decider, **kwargs):
        batcher = BatchingDecider(decider, **kwargs)
        self.addCleanup(batcher.close)
        return batcher

    def test_partial_batch_flushes_without_more_requests(self):
        decider = FakeDecider()
        batcher = self.batcher(decider, max_batch_size=8, max_wait_ms=10)
        self.assertEqual(batcher.submit({"state": "single"}).result(timeout=2), decider.result)
        self.assertEqual(decider.calls, [[{"state": "single"}]])

    def test_new_arrivals_do_not_reset_oldest_deadline(self):
        decider = FakeDecider()
        batcher = self.batcher(decider, max_batch_size=1000, max_wait_ms=80)
        first = batcher.submit({"state": "first"})
        stop = threading.Event()

        def produce():
            while not stop.wait(0.01):
                batcher.submit({"state": "later"})

        producer = threading.Thread(target=produce)
        producer.start()
        try:
            first.result(timeout=0.5)
            self.assertTrue(producer.is_alive())
        finally:
            stop.set()
            producer.join(timeout=1)

    def test_full_batch_flushes_immediately_and_preserves_mapping(self):
        class EchoDecider:
            def decide(self, requests):
                return [{"echo": request["state"]} for request in requests]

        batcher = self.batcher(EchoDecider(), max_batch_size=2, max_wait_ms=5000)
        first = batcher.submit({"state": "first"})
        second = batcher.submit({"state": "second"})
        self.assertEqual(first.result(timeout=1), {"echo": "first"})
        self.assertEqual(second.result(timeout=1), {"echo": "second"})

    def test_batch_size_limit_and_single_inference_worker(self):
        decider = FakeDecider()
        batcher = self.batcher(decider, max_batch_size=2, max_wait_ms=10)
        futures = [batcher.submit({"state": str(index)}) for index in range(5)]
        for future in futures:
            future.result(timeout=2)
        self.assertEqual(sum(len(batch) for batch in decider.calls), 5)
        self.assertTrue(all(len(batch) <= 2 for batch in decider.calls))
        self.assertEqual(len(set(decider.thread_ids)), 1)
        self.assertNotEqual(decider.thread_ids[0], threading.get_ident())

    def test_queue_overflow_and_shutdown_release_waiters(self):
        started = threading.Event()
        release = threading.Event()

        class BlockingDecider:
            def decide(self, requests):
                started.set()
                if not release.wait(3):
                    raise RuntimeError("test inference was not released")
                return [{} for _ in requests]

        batcher = self.batcher(BlockingDecider(), max_batch_size=1, max_queue_size=1)
        running = batcher.submit({})
        self.assertTrue(started.wait(1))
        queued = batcher.submit({})
        with self.assertRaises(ServiceUnavailable):
            batcher.submit({})
        closer = threading.Thread(target=batcher.close)
        closer.start()
        try:
            with self.assertRaises(ServiceUnavailable):
                queued.result(timeout=1)
        finally:
            release.set()
            closer.join(timeout=3)
        running.result(timeout=1)
        self.assertFalse(closer.is_alive())
        with self.assertRaises(ServiceUnavailable):
            batcher.submit({})

    def test_canceled_queued_requests_are_not_inferred(self):
        decider = FakeDecider()
        batcher = self.batcher(decider, max_batch_size=2, max_wait_ms=5000)
        canceled = batcher.submit({"state": "canceled"})
        self.assertTrue(canceled.cancel())
        batcher.submit({"state": "live"}).result(timeout=1)
        self.assertEqual(decider.calls, [[{"state": "live"}]])

    def test_inference_failure_reaches_whole_batch_and_worker_recovers(self):
        decider = FakeDecider()
        decider.error = RuntimeError("GPU failure")
        batcher = self.batcher(decider, max_batch_size=2, max_wait_ms=10)
        futures = [batcher.submit({}), batcher.submit({})]
        for future in futures:
            with self.assertRaisesRegex(RuntimeError, "GPU failure"):
                future.result(timeout=2)
        decider.error = None
        batcher.submit({}).result(timeout=2)

    def test_bad_request_does_not_fail_other_members(self):
        class SelectiveDecider:
            def decide(self, requests):
                if any(request["state"] == "bad" for request in requests):
                    raise FakeRequestError("too long")
                return [{"ok": True} for _ in requests]

        batcher = self.batcher(SelectiveDecider(), max_batch_size=2, max_wait_ms=5000,
                              request_errors=(FakeRequestError,))
        bad = batcher.submit({"state": "bad"})
        good = batcher.submit({"state": "good"})
        with self.assertRaises(FakeRequestError):
            bad.result(timeout=1)
        self.assertEqual(good.result(timeout=1), {"ok": True})

    def test_invalid_batch_result_count_fails_all_waiters(self):
        class InvalidDecider:
            def decide(self, requests):
                return []

        batcher = self.batcher(InvalidDecider(), max_wait_ms=0)
        with self.assertRaisesRegex(RuntimeError, "batch result count"):
            batcher.submit({}).result(timeout=1)

    def test_batch_configuration_validation(self):
        for config in [{"max_batch_size": 0}, {"max_wait_ms": -1},
                       {"max_wait_ms": float("nan")}, {"max_queue_size": 0}]:
            with self.subTest(config=config), self.assertRaises(ValueError):
                BatchingDecider(FakeDecider(), **config)


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.decider = FakeDecider()
        self.server = create_server(self.decider, port=0, request_errors=(FakeRequestError,))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def request(self, method, path, body=None):
        connection = HTTPConnection(*self.server.server_address, timeout=5)
        try:
            connection.request(method, path, body=body,
                               headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def test_health_does_not_invoke_model(self):
        self.assertEqual(self.request("GET", "/health"),
                         (200, {"status": "ready", "model": DEFAULT_MODEL}))
        self.assertEqual(self.decider.calls, [])

    def test_official_request_and_response_shape(self):
        status, response = self.request("POST", "/v1/decisions", json.dumps(request_body()))
        self.assertEqual(status, 200)
        self.assertEqual(response, {
            "model": DEFAULT_MODEL,
            "answers": {"relevance": {"type": "noul", "noul": 0.875}},
            "usage": {"input_tokens": 123, "output_tokens": 0},
        })
        self.assertEqual(len(self.decider.calls), 1)

    def test_concurrent_http_requests_are_batched(self):
        barrier = threading.Barrier(2)

        def send(_index):
            barrier.wait(timeout=2)
            return self.request("POST", "/v1/decisions", json.dumps(request_body()))

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(send, range(2)))
        self.assertTrue(all(status == 200 for status, _ in results))
        self.assertEqual(sum(len(batch) for batch in self.decider.calls), 2)
        self.assertTrue(all(isinstance(batch, list) for batch in self.decider.calls))

    def test_invalid_requests_return_400_without_inference(self):
        for body in ["not json", "[]", "{}"]:
            with self.subTest(body=body):
                self.assertEqual(self.request("POST", "/v1/decisions", body)[0], 400)
        self.assertEqual(self.decider.calls, [])

    def test_old_rerank_route_is_removed(self):
        self.assertEqual(self.request("POST", "/rerank", "{}")[0], 404)
        self.assertEqual(self.request("GET", "/unknown")[0], 404)

    def test_sdk_request_errors_return_400(self):
        self.decider.error = FakeRequestError("Input exceeds model context")
        status, response = self.request("POST", "/v1/decisions", json.dumps(request_body()))
        self.assertEqual(status, 400)
        self.assertEqual(response, {"error": "Input exceeds model context"})

    def test_inference_errors_return_500(self):
        self.decider.error = RuntimeError("Internal GPU error")
        with self.assertLogs(level="ERROR"):
            status, response = self.request("POST", "/v1/decisions", json.dumps(request_body()))
        self.assertEqual(status, 500)
        self.assertEqual(response, {"error": "Native Decider inference failed"})


class LauncherTests(unittest.TestCase):
    script = Path(__file__).resolve().parents[2] / "scripts" / "rerankers" / "decider_server.sh"

    def test_help_works_from_another_directory_without_loading_sdk(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                ["bash", str(self.script), "--help"], cwd=directory,
                env={**os.environ, "PIIKA_DECIDER_PYTHON": sys.executable},
                capture_output=True, text=True, check=True,
            )
        self.assertIn("--max-batch-size", result.stdout)
        self.assertIn("--served-model-name", result.stdout)

    def test_preserves_arguments_working_directory_and_exit_status(self):
        with tempfile.TemporaryDirectory() as directory:
            python = Path(directory) / "fake python"
            python.write_text('#!/bin/bash\nprintf "%s\\n" "$PWD" "$@"\nexit 7\n')
            python.chmod(0o700)
            arguments = ["--model", "models/my checkpoint", "--max-batch-size", "4"]
            result = subprocess.run(
                ["bash", str(self.script), *arguments], cwd=directory,
                env={**os.environ, "PIIKA_DECIDER_PYTHON": str(python)},
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 7)
            lines = result.stdout.splitlines()
            self.assertEqual(Path(lines[0]).resolve(), Path(directory).resolve())
            self.assertEqual(lines[1:], ["-u", "-B", str(self.script.with_suffix(".py")), *arguments])

    def test_missing_model_returns_argument_error(self):
        result = subprocess.run(
            ["bash", str(self.script)],
            env={**os.environ, "PIIKA_DECIDER_PYTHON": sys.executable},
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--model", result.stderr)


if __name__ == "__main__":
    unittest.main()
