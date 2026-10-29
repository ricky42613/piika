import assert from "node:assert/strict";
import test from "node:test";

import { prepareRerankCandidates } from "../../src/pi-search/reranking/document_loader";
import type { SearchBackendReadDocumentFoundResponse } from "../../src/pi-search/searcher/contract/types";

function page(
  docid: string,
  text: string,
  extra: Partial<SearchBackendReadDocumentFoundResponse> = {},
): SearchBackendReadDocumentFoundResponse {
  return { found: true, docid, text, offset: 1, limit: 200, truncated: false, ...extra };
}

void test("document preparation deduplicates before reads and preserves first hit's score and order", async () => {
  const signal = new AbortController().signal;
  const calls: string[] = [];
  const hits = Object.freeze([
    Object.freeze({ docid: "a", score: 10, title: "Search title", snippet: "Not scoring text" }),
    Object.freeze({ docid: "b", score: 8, title: "Fallback title" }),
    Object.freeze({ docid: "a", score: 20 }),
  ]);
  const result = await prepareRerankCandidates(
    {
      readDocument: async (request, receivedSignal) => {
        assert.deepEqual(request, { docid: request.docid });
        assert.strictEqual(receivedSignal, signal);
        calls.push(request.docid);
        return page(request.docid, `Document ${request.docid}`, {
          title: request.docid === "a" ? "Document title" : null,
        });
      },
    },
    hits,
    signal,
  );
  assert.deepEqual(calls, ["a", "b"]);
  assert.deepEqual(result, [
    { docid: "a", text: "Document a", title: "Document title", retrievalScore: 10 },
    { docid: "b", text: "Document b", title: "Fallback title", retrievalScore: 8 },
  ]);
  assert.equal(hits.length, 3);
});

void test("document preparation takes the first 4000 characters verbatim", async () => {
  const text = "  中文\n\t".repeat(1000);
  let calls = 0;
  const result = await prepareRerankCandidates(
    {
      readDocument: async () => {
        calls += 1;
        return page("a", text, { truncated: true, nextOffset: 201 });
      },
    },
    [{ docid: "a" }],
  );
  assert.equal(result[0].text, text.slice(0, 4000));
  assert.equal(result[0].text.length, 4000);
  assert.equal(calls, 1);
});

void test("document preparation reads once even when the backend returns a continuation", async () => {
  let calls = 0;
  const result = await prepareRerankCandidates(
    {
      readDocument: async (request) => {
        calls += 1;
        assert.deepEqual(request, { docid: "a" });
        return page("a", "a".repeat(2500), { truncated: true, nextOffset: 201 });
      },
    },
    [{ docid: "a" }],
  );
  assert.equal(calls, 1);
  assert.equal(result[0].text, "a".repeat(2500));
});

void test("short and empty documents stay short without falling back to snippets", async () => {
  const result = await prepareRerankCandidates(
    {
      readDocument: async (request) =>
        page(request.docid, request.docid === "a" ? "  這是一段文字。\nSecond line  " : ""),
    },
    [{ docid: "a" }, { docid: "b", snippet: "must not use this" }],
  );
  assert.equal(result[0].text, "  這是一段文字。\nSecond line  ");
  assert.equal(result[1].text, "");
});

void test("empty hit lists do not call readDocument", async () => {
  assert.deepEqual(
    await prepareRerankCandidates(
      {
        readDocument: async () => {
          throw new Error("must not read");
        },
      },
      [],
    ),
    [],
  );
});

void test("missing documents and backend failures are not silently replaced with snippets", async () => {
  await assert.rejects(
    prepareRerankCandidates(
      {
        readDocument: async () => ({ found: false, docid: "a" }),
      },
      [{ docid: "a", snippet: "fallback" }],
    ),
    /document a not found/,
  );
  const error = new Error("backend unavailable");
  await assert.rejects(
    prepareRerankCandidates(
      {
        readDocument: async () => {
          throw error;
        },
      },
      [{ docid: "a" }],
    ),
    (actual) => actual === error,
  );
});

void test("mismatched docids are rejected", async () => {
  await assert.rejects(
    prepareRerankCandidates(
      {
        readDocument: async () => page("wrong", "text"),
      },
      [{ docid: "a" }],
    ),
    /Unexpected document/,
  );
});

void test("cancellation prevents initial and subsequent reads", async () => {
  const controller = new AbortController();
  let calls = 0;
  const backend = {
    readDocument: async () => {
      calls += 1;
      controller.abort();
      return page("a", "short", { truncated: true, nextOffset: 201 });
    },
  };
  await assert.rejects(prepareRerankCandidates(backend, [{ docid: "a" }], controller.signal), {
    name: "AbortError",
  });
  await assert.rejects(prepareRerankCandidates(backend, [{ docid: "a" }], controller.signal), {
    name: "AbortError",
  });
  assert.equal(calls, 1);
});
