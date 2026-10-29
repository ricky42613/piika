import assert from "node:assert/strict";
import type { EventEmitter } from "node:events";
import test from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

import { buildMockExtensionConfig, type PiSearchRerankingConfig } from "../../src/pi-search/config";
import {
  registerPiSearchExtension,
  type PiSearchExtensionOptions,
} from "../../src/pi-search/extension";

type RegisteredTool = Parameters<ExtensionAPI["registerTool"]>[0];
const scorer = {
  provider: "pplx-decider" as const,
  baseUrl: "http://localhost:8002/v1",
  model: "pplx-decider-flash-v1.2",
};
const customized: PiSearchRerankingConfig = {
  mode: "customized",
  candidateK: 50,
  scorer,
  generator: { provider: "qorrect", baseUrl: "http://localhost:8001/v1", model: "qorrect" },
};

function setup(
  reranking: PiSearchRerankingConfig | undefined,
  options: PiSearchExtensionOptions = {},
) {
  const tools = new Map<string, RegisteredTool>();
  const handlers = new Map<string, () => void | Promise<void>>();
  const emitter: EventEmitter = process;
  const events = ["exit", "SIGINT", "SIGTERM"] as const;
  const originalListeners = new Map(
    events.map((event) => [event, new Set(emitter.listeners(event))]),
  );
  registerPiSearchExtension(
    {
      on: (name: string, handler: () => void | Promise<void>) => handlers.set(name, handler),
      registerTool: (tool: RegisteredTool) => tools.set(tool.name, tool),
    } as never,
    {
      toolInterface: "pyserini-rest-2tool",
      resolveConfig: () => ({
        ...buildMockExtensionConfig({
          documents: [
            { docid: "a", text: "Engine A", snippet: "Preview A" },
            { docid: "b", text: "Engine B", snippet: "Preview B" },
          ],
        }),
        reranking,
      }),
      ...options,
    },
  );
  return {
    async search() {
      const result = await tools
        .get("search")!
        .execute(
          "call-1",
          { reason: "Find the designer", query: "engine", hits: 1 },
          undefined,
          undefined,
          { cwd: "." } as never,
        );
      return result.content
        .filter((item) => item.type === "text")
        .map((item) => item.text)
        .join("\n");
    },
    async cleanup() {
      await handlers.get("session_shutdown")?.();
      for (const event of events) {
        for (const listener of emitter.listeners(event)) {
          if (!originalListeners.get(event)!.has(listener))
            emitter.removeListener(event, listener as (...args: unknown[]) => void);
        }
      }
    },
  };
}

void test("extension initializes fixed reranking from config and executes the configured scorer", async () => {
  let calls = 0;
  const extension = setup(
    { mode: "fixed-template", candidateK: 50, scorer },
    {
      rerankingFetch: async (_url, init) => {
        calls += 1;
        const body = JSON.parse(String(init?.body));
        return Response.json({
          model: scorer.model,
          answers: {
            relevance: { type: "noul", noul: body.state === "Engine B" ? 0.9 : 0.1 },
          },
        });
      },
    },
  );
  try {
    const text = await extension.search();
    assert.match(text, /1\. docid=b/);
    assert.match(text, /rerank_score=0\.9000/);
    assert.doesNotMatch(text, /Rerank criterion/);
    assert.equal(calls, 2);
  } finally {
    await extension.cleanup();
  }
});

void test("customized extension uses explicit original question and current search reason", async () => {
  let generationCalls = 0;
  const question = "Does this document identify the engine designer?";
  const extension = setup(customized, {
    originalQuestion: "Who designed the engine?",
    rerankingFetch: async (url, init) => {
      const body = JSON.parse(String(init?.body));
      if (String(url).endsWith("/chat/completions")) {
        generationCalls += 1;
        assert.match(body.messages[0].content, /Original question:\nWho designed the engine\?/);
        assert.match(body.messages[0].content, /Reason for current search:\nFind the designer/);
        assert.match(body.messages[0].content, /Current search query:\nengine/);
        return Response.json({
          choices: [{ message: { content: question }, finish_reason: "stop" }],
        });
      }
      assert.equal(body.questions.relevance.instructions, question);
      return Response.json({
        model: scorer.model,
        answers: { relevance: { type: "noul", noul: 0.5 } },
      });
    },
  });
  try {
    const text = await extension.search();
    assert.ok(text.includes(`Rerank criterion: \n${question}`));
    assert.equal(generationCalls, 1);
  } finally {
    await extension.cleanup();
  }
});

void test("disabled extension performs ordinary search without scorer requests", async () => {
  const extension = setup(
    { mode: "off" },
    {
      rerankingFetch: async () => {
        throw new Error("must not call models");
      },
    },
  );
  try {
    assert.match(await extension.search(), /Pyserini REST ranking/);
  } finally {
    await extension.cleanup();
  }
});

void test("extension refuses reranking on 3-tool before registering tools", () => {
  assert.throws(
    () =>
      setup(
        { mode: "fixed-template", candidateK: 50, scorer },
        {
          toolInterface: "pi-serini-3tool",
        },
      ),
    /only.*2tool/,
  );
});

void test("customized extension rejects missing original question at initialization", () => {
  assert.throws(
    () => setup(customized, { originalQuestion: " " }),
    /requires PI_SEARCH_ORIGINAL_QUESTION/,
  );
});
