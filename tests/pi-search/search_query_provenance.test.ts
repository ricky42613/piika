import assert from "node:assert/strict";
import type { EventEmitter } from "node:events";
import test from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { parsePiSearchExtensionConfig } from "../../src/pi-search/config";
import { registerPiSearchExtension } from "../../src/pi-search/extension";
import {
  CitedDirectSearchParamsSchema,
  DirectSearchParamsSchema,
} from "../../src/pi-search/protocol/schemas";

const backend = { kind: "mock", documents: [{ docid: "a", text: "engine", snippet: "engine" }] };

void test("provenance config accepts off and cite-docids but rejects unknown modes", () => {
  for (const mode of ["off", "cite-docids"]) {
    assert.equal(
      parsePiSearchExtensionConfig(JSON.stringify({ backend, searchQueryProvenance: mode }))
        .searchQueryProvenance,
      mode,
    );
  }
  assert.throws(() =>
    parsePiSearchExtensionConfig(JSON.stringify({ backend, searchQueryProvenance: "unknown" })),
  );
});

for (const mode of [undefined, "off", "cite-docids"] as const) {
  void test(`search provenance ${mode ?? "omitted"} selects schema and records only enabled citations`, async (t) => {
    const tools: Array<Parameters<ExtensionAPI["registerTool"]>[0]> = [];
    const handlers = new Map<string, () => void | Promise<void>>();
    const emitter: EventEmitter = process;
    const events = ["exit", "SIGINT", "SIGTERM"];
    const listeners = new Map(events.map((event) => [event, new Set(emitter.listeners(event))]));
    t.after(async () => {
      await handlers.get("session_shutdown")?.();
      for (const event of events)
        for (const listener of emitter.listeners(event)) {
          if (!listeners.get(event)!.has(listener))
            emitter.removeListener(event, listener as (...args: unknown[]) => void);
        }
    });
    registerPiSearchExtension(
      {
        registerTool: (tool: Parameters<ExtensionAPI["registerTool"]>[0]) => tools.push(tool),
        on: (name: string, handler: () => void) => handlers.set(name, handler),
      } as never,
      {
        toolInterface: "pyserini-rest-2tool",
        resolveConfig: () =>
          parsePiSearchExtensionConfig(JSON.stringify({ backend, searchQueryProvenance: mode })),
      },
    );
    const search = tools.find((tool) => tool.name === "search")!;
    assert.equal(tools.filter((tool) => tool.name === "search").length, 1);
    const enabled = mode === "cite-docids";
    assert.equal(
      search.parameters,
      enabled ? CitedDirectSearchParamsSchema : DirectSearchParamsSchema,
    );
    assert.deepEqual(
      Object.keys((enabled ? CitedDirectSearchParamsSchema : DirectSearchParamsSchema).properties),
      enabled ? ["reason", "cite", "query", "hits"] : ["reason", "query", "hits"],
    );
    assert.equal(
      search.promptGuidelines?.some((line) => line.includes("cite")),
      enabled,
    );
    const execute = (cite: unknown) =>
      search.execute(
        "call",
        { reason: "Use the document's clue", cite, query: "engine", hits: 1 },
        undefined,
        undefined,
        { cwd: "." } as never,
      );
    const result = await execute(["a", "a", "b"]);
    assert.deepEqual(
      (result.details as { cite?: string[] }).cite,
      enabled ? ["a", "b"] : undefined,
    );
    if (enabled) {
      assert.deepEqual(((await execute([])).details as { cite: string[] }).cite, []);
      for (const invalid of [undefined, "a", [""], [" "], [1]]) {
        await assert.rejects(() => execute(invalid), /cite must be an array/);
      }
    }
  });
}

void test("cite-docids is rejected for the three-tool interface", () => {
  assert.throws(
    () =>
      registerPiSearchExtension({} as ExtensionAPI, {
        toolInterface: "pi-serini-3tool",
        resolveConfig: () =>
          parsePiSearchExtensionConfig(
            JSON.stringify({ backend, searchQueryProvenance: "cite-docids" }),
          ),
      }),
    /requires pyserini-rest-2tool/,
  );
});
