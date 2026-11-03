#!/usr/bin/env python3
"""Local HTTP scorer for the official pplx-decider-flash-v1.2 checkpoint.

This is deliberately NOT generic vLLM: the checkpoint needs noncausal full
attention and its per-marker decision readout. Bind to loopback only.
"""
from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = HERE / 'model'
MODEL_ID = 'pplx-decider-flash-v1.2'
PORT = int(os.environ.get('PPLX_DECIDER_PORT', '18286'))
MAX_BODY = 2 * 1024 * 1024
MAX_BATCH = 32
sys.path.insert(0, str(MODEL.resolve() / 'source' / 'src'))
from pplx_decider import Decider

started = time.monotonic()
engine = Decider.from_pretrained(MODEL.resolve(), device='cuda', fast=False,
                                  max_length=8192, batch_tokens=4096, cpu_threads=4)
lock = threading.Lock()
print(f'READY {MODEL_ID} port={PORT} load_seconds={time.monotonic()-started:.2f} path={MODEL.resolve()}',flush=True)


def probability(margin: float, temperature: float) -> float:
    x = margin / temperature
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def request_score(payload: dict):
    if payload.get('model', MODEL_ID) != MODEL_ID:
        raise ValueError(f'model must be {MODEL_ID}')
    query = payload.get('queries')
    docs = payload.get('documents')
    if not isinstance(query, str) or not query.strip() or not isinstance(docs, list) or not 1 <= len(docs) <= MAX_BATCH:
        raise ValueError('queries must be one nonempty string; documents must contain 1..32 strings')
    if any(not isinstance(doc, str) or not doc.strip() for doc in docs):
        raise ValueError('each document must be a nonempty string')
    instruction = payload.get('instruction', 'Does the candidate document provide evidence that helps answer the search query?')
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('instruction must be a nonempty string')
    requests = [{'state': {'query': query, 'candidate_passage': doc},
                 'questions': {'relevant': {'type': 'noul', 'instructions': instruction}}} for doc in docs]
    with lock:
        logits = engine.logits(requests)
    data = []
    for i, row in enumerate(logits):
        false, true = row['relevant']
        margin = true - false
        prob = probability(margin, engine.temperature)
        data.append({'index': i, 'score': prob, 'margin': margin, 'probability': prob,
                     'false_logit': false, 'true_logit': true})
    return {'model': MODEL_ID, 'data': data, 'temperature': engine.temperature,
            'score_semantics': 'calibrated P(true); use margin for stable sorting'}


def request_decide(payload: dict):
    if payload.get('model', MODEL_ID) != MODEL_ID:
        raise ValueError(f'model must be {MODEL_ID}')
    requests = payload.get('requests')
    if requests is None:
        request = {k: payload[k] for k in ('state', 'questions') if k in payload}
        if 'images' in payload:
            request['images'] = payload['images']
        requests = [request]
        single = True
    else:
        single = False
        if not isinstance(requests, list) or not 1 <= len(requests) <= MAX_BATCH:
            raise ValueError('requests must contain 1..32 requests')
    with lock:
        response = engine.decide(requests)
    return {'model': MODEL_ID, 'response': response[0] if single else response}


class Handler(BaseHTTPRequestHandler):
    server_version = 'pplx-decider-flash-v1.2/1.0'

    def send_json(self, status: int, value: dict):
        data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == '/health':
            self.send_json(200, {'status': 'ok', 'model': MODEL_ID, 'backend': 'pplx_decider_reference'})
        elif self.path == '/v1/models':
            self.send_json(200, {'object': 'list', 'data': [{'id': MODEL_ID, 'object': 'model',
                             'owned_by': 'pplx_decider', 'root': str(MODEL.resolve())}]})
        else:
            self.send_json(404, {'error': 'unknown route'})

    def do_POST(self):
        size = self.headers.get('Content-Length')
        try:
            if size is None or not size.isdigit() or not 0 < int(size) <= MAX_BODY:
                raise ValueError('Content-Length must be 1..2097152 bytes')
            payload = json.loads(self.rfile.read(int(size)))
            if not isinstance(payload, dict):
                raise ValueError('expected a JSON object')
            if self.path == '/score':
                result = request_score(payload)
            elif self.path == '/v1/decide':
                result = request_decide(payload)
            else:
                self.send_json(404, {'error': 'unknown route'})
                return
            self.send_json(200, result)
        except (ValueError, KeyError, TypeError) as exc:
            self.send_json(400, {'error': str(exc)})
        except Exception as exc:
            print('REQUEST_FAILURE',repr(exc),flush=True)
            self.send_json(500, {'error': type(exc).__name__})


if __name__ == '__main__':
    ThreadingHTTPServer(('127.0.0.1', PORT), Handler).serve_forever(poll_interval=0.2)
