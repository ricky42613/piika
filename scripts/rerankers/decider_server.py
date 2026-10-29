"""Native Decider service with a text-only subset of the Decisions API.

Supports one noul question per request. Uses the SDK's calibrated probability,
not raw logits. Candidate bookkeeping and ranking belong to the caller.
"""

import argparse
from collections import deque
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
import threading
import time
from typing import Any, Protocol

DEFAULT_MODEL = "pplx-decider-flash-v1.2"


class Decider(Protocol):
    def decide(self, request: dict[str, Any] | list[dict[str, Any]]) -> Any: ...


class ServiceUnavailable(RuntimeError):
    pass


class BatchingDecider:
    """HTTP threads enqueue work; exactly one worker owns model inference.

    The collection deadline is anchored to the first request's enqueue time.
    It does not bound time spent waiting for an earlier GPU batch to finish.
    """

    def __init__(self, decider, max_batch_size=8, max_wait_ms=10, max_queue_size=256,
                 request_errors=()):
        if isinstance(max_batch_size, bool) or not isinstance(max_batch_size, int) or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")
        if not math.isfinite(max_wait_ms) or max_wait_ms < 0:
            raise ValueError("max_wait_ms must be finite and nonnegative")
        if isinstance(max_queue_size, bool) or not isinstance(max_queue_size, int) or max_queue_size < 1:
            raise ValueError("max_queue_size must be a positive integer")
        self.decider = decider
        self.max_batch_size = max_batch_size
        self.max_wait_seconds = max_wait_ms / 1000
        self.max_queue_size = max_queue_size
        self.request_errors = request_errors
        self.queue = deque()
        self.condition = threading.Condition()
        self.closed = False
        self.worker = threading.Thread(target=self._run, name="decider-inference", daemon=True)
        self.worker.start()

    def submit(self, request):
        future = Future()
        with self.condition:
            if self.closed:
                raise ServiceUnavailable("Decider is shutting down")
            if len(self.queue) >= self.max_queue_size:
                raise ServiceUnavailable("Decider request queue is full")
            self.queue.append((time.monotonic(), request, future))
            self.condition.notify()
        return future

    def decide(self, request):
        return self.submit(request).result()

    def close(self):
        with self.condition:
            self.closed = True
            while self.queue:
                _, _, future = self.queue.popleft()
                if future.set_running_or_notify_cancel():
                    future.set_exception(ServiceUnavailable("Decider is shutting down"))
            self.condition.notify_all()
        # Running inference cannot safely be interrupted; let it finish.
        self.worker.join()

    def _infer(self, requests):
        results = self.decider.decide(requests)
        if not isinstance(results, list) or len(results) != len(requests):
            raise RuntimeError("Decider returned an unexpected batch result count")
        return results

    def _run(self):
        while True:
            with self.condition:
                while not self.queue and not self.closed:
                    self.condition.wait()
                if self.closed:
                    return
                deadline = self.queue[0][0] + self.max_wait_seconds
                while len(self.queue) < self.max_batch_size and not self.closed:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self.condition.wait(remaining)
                if self.closed:
                    return
                batch = [self.queue.popleft() for _ in range(min(len(self.queue), self.max_batch_size))]
            batch = [(request, future) for _, request, future in batch
                     if future.set_running_or_notify_cancel()]
            if not batch:
                continue
            try:
                results = self._infer([request for request, _ in batch])
            except self.request_errors:
                # A too-long/invalid request must not fail unrelated batch members.
                for request, future in batch:
                    try:
                        future.set_result(self._infer([request])[0])
                    except Exception as error:
                        future.set_exception(error)
            except Exception as error:
                for _, future in batch:
                    future.set_exception(error)
            else:
                for (_, future), result in zip(batch, results, strict=True):
                    future.set_result(result)


def validate_request(payload, model: str):
    if not isinstance(payload, dict):
        raise ValueError("Request must be an object")
    if set(payload) - {"model", "state", "questions"}:
        raise ValueError("Supported fields are model, state, and questions")
    if payload.get("model") != model:
        raise ValueError(f"model must be {model}")
    if not isinstance(payload.get("state"), str):
        raise ValueError("This service currently supports text state only")
    questions = payload.get("questions")
    if not isinstance(questions, dict) or len(questions) != 1:
        raise ValueError("This service requires exactly one question per request")
    for name, question in questions.items():
        if not name.strip() or not isinstance(question, dict):
            raise ValueError("Each named question must be an object with a non-empty name")
        if question.get("type") != "noul":
            raise ValueError("This service currently supports noul questions only")
        if set(question) - {"type", "instructions", "criteria"}:
            raise ValueError("Unsupported question fields")
        instructions = question.get("instructions")
        if instructions is not None and not isinstance(instructions, str):
            raise ValueError("instructions must be a string or null")
        criteria = question.get("criteria")
        if criteria is not None:
            if not isinstance(criteria, dict) or set(criteria) - {"true", "false"}:
                raise ValueError("noul criteria may contain only true and false")
            if any(value is not None and not isinstance(value, str) for value in criteria.values()):
                raise ValueError("noul criteria descriptions must be strings or null")
        has_instructions = bool(instructions and instructions.strip())
        has_criteria = any(value and value.strip() for value in (criteria or {}).values())
        if not has_instructions and not has_criteria:
            raise ValueError("noul requires instructions or criteria")


def decide_request(decider: Decider, payload: dict[str, Any], model: str = DEFAULT_MODEL):
    validate_request(payload, model)
    result = decider.decide({"state": payload["state"], "questions": payload["questions"]})
    return {
        "model": model,
        "answers": result["answers"],
        "usage": {
            "input_tokens": result["usage"]["input_tokens"],
            "output_tokens": result["usage"]["output_tokens"],
        },
    }


def create_server(
    decider: Decider,
    host: str = "127.0.0.1",
    port: int = 8002,
    model: str = DEFAULT_MODEL,
    request_errors: tuple[type[Exception], ...] = (),
    max_batch_size: int = 8,
    max_wait_ms: float = 10,
    max_queue_size: int = 256,
) -> ThreadingHTTPServer:
    """Concurrent HTTP intake with serial batched inference; trusted local use only."""
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def respond(self, status, payload):
            body = json.dumps(payload, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/health":
                self.respond(404, {"error": "Not found"})
                return
            self.respond(200, {"status": "ready", "model": model})

        def do_POST(self):
            if self.path != "/v1/decisions":
                self.respond(404, {"error": "Not found"})
                return
            try:
                if self.headers.get("Transfer-Encoding"):
                    raise ValueError("Transfer-Encoding is not supported")
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > 8 * 1024 * 1024:
                    raise ValueError("Request body must be between 1 byte and 8 MiB")
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("Incomplete request body")
                payload = json.loads(raw)
                validate_request(payload, model)
            except (ValueError, UnicodeError, TimeoutError) as error:
                self.respond(400, {"error": str(error)})
                return
            try:
                response = decide_request(batcher, payload, model)
                json.dumps(response, allow_nan=False)
            except ServiceUnavailable as error:
                self.respond(503, {"error": str(error)})
                return
            except request_errors as error:
                self.respond(400, {"error": str(error)})
                return
            except Exception:
                logging.exception("Native Decider inference failed")
                self.respond(500, {"error": "Native Decider inference failed"})
                return
            self.respond(200, response)

    class Server(ThreadingHTTPServer):
        def server_close(self):
            batcher.close()
            super().server_close()

    batcher = BatchingDecider(decider, max_batch_size, max_wait_ms, max_queue_size, request_errors)
    try:
        return Server((host, port), Handler)
    except Exception:
        batcher.close()
        raise


def main():
    parser = argparse.ArgumentParser(description="Serve the native Decider Decisions API subset")
    parser.add_argument("--model", required=True, help="Local checkpoint path or Hugging Face repo ID")
    parser.add_argument("--served-model-name", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--max-wait-ms", type=float, default=10)
    parser.add_argument("--max-queue-size", type=int, default=256)
    args = parser.parse_args()
    if args.max_length < 1:
        parser.error("--max-length must be positive")
    if args.max_batch_size < 1 or args.max_queue_size < 1:
        parser.error("--max-batch-size and --max-queue-size must be positive")
    if not math.isfinite(args.max_wait_ms) or args.max_wait_ms < 0:
        parser.error("--max-wait-ms must be finite and nonnegative")
    if not args.served_model_name.strip():
        parser.error("--served-model-name must be non-empty")
    logging.basicConfig(level=logging.INFO)
    # Lazy import keeps --help and fake-model tests independent of GPU dependencies.
    from pplx_decider import Decider as NativeDecider, LengthError, RequestError

    decider = NativeDecider.from_pretrained(
        args.model, device=args.device, fast=False, max_length=args.max_length,
    )
    with create_server(decider, args.host, args.port, args.served_model_name,
                       request_errors=(RequestError, LengthError),
                       max_batch_size=args.max_batch_size, max_wait_ms=args.max_wait_ms,
                       max_queue_size=args.max_queue_size) as server:
        logging.info("Native Decider ready at http://%s:%s/v1/decisions", *server.server_address)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
