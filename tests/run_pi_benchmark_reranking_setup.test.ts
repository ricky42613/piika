import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { chmodSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import type { PiSearchRerankingConfig } from "../src/pi-search/config";

const scorer = {
  provider: "pplx-decider" as const,
  baseUrl: "http://127.0.0.1:8002/v1",
  model: "pplx-decider-flash-v1.2",
  concurrency: 4,
};
const cases: Array<[string, PiSearchRerankingConfig | undefined]> = [
  ["omitted", undefined],
  ["off", { mode: "off" }],
  ["fixed-template", { mode: "fixed-template", candidateK: 50, scorer }],
  [
    "customized",
    {
      mode: "customized",
      candidateK: 50,
      scorer,
      generator: {
        provider: "qorrect",
        baseUrl: "http://127.0.0.1:8001/v1",
        model: "qorrect",
        temperature: 0.3,
        tokenEnv: "QORRECT_TEST_API_KEY",
      },
    },
  ],
];

const testCases = cases.flatMap(([name, reranking]) => [
  { name, reranking, fromFile: false },
  ...(reranking ? [{ name, reranking, fromFile: true }] : []),
]);
for (const { name, reranking, fromFile } of testCases) {
  void test(`runner snapshots ${name} reranking via ${fromFile ? "single CLI file override" : "env"}`, (t) => {
    const root = mkdtempSync(join(tmpdir(), "piika-rerank-setup-"));
    t.after(() => rmSync(root, { recursive: true, force: true }));
    const queries = join(root, "queries.tsv");
    const qrels = join(root, "qrels.txt");
    const output = join(root, "run");
    const captured = join(root, "child-config.json");
    const fakePi = join(root, "fake-pi.mjs");
    const config = {
      backend: { kind: "mock", documents: [] },
      ...(reranking ? { reranking } : {}),
    };
    const configPath = join(root, "reranking.json");
    if (fromFile) writeFileSync(configPath, JSON.stringify(reranking));
    const envConfig = fromFile
      ? { ...config, reranking: reranking?.mode === "off" ? cases[2][1] : { mode: "off" } }
      : config;
    writeFileSync(queries, "1\tWho designed the analytical engine?\n");
    writeFileSync(qrels, "1 0 doc-1 1\n");
    writeFileSync(
      fakePi,
      `#!/usr/bin/env node
import { writeFileSync } from 'node:fs';
writeFileSync(process.env.RERANK_TEST_CAPTURE, process.env.PI_SEARCH_EXTENSION_CONFIG);
console.log(JSON.stringify({type:'session'}));
console.log(JSON.stringify({type:'message_end',message:{role:'assistant',content:[{type:'text',text:'Charles Babbage.'}]}}));
console.log(JSON.stringify({type:'agent_end'}));
`,
    );
    chmodSync(fakePi, 0o755);
    execFileSync(
      fromFile ? process.execPath : "npx",
      [
        ...(fromFile
          ? ["bin/piika.js", "run", "--mode", "single", "--reranking-config", configPath]
          : ["tsx", "src/orchestration/run_pi_benchmark.ts"]),
        "--benchmark",
        "benchmark-template",
        "--querySet",
        "dev",
        "--query",
        queries,
        "--qrels",
        qrels,
        "--outputDir",
        output,
        "--model",
        "openai-codex/gpt-5.4-mini",
        "--thinking",
        "medium",
        "--extension",
        "src/extensions/pi_search.ts",
        "--pi",
        fakePi,
        "--timeoutSeconds",
        "5",
        "--promptVariant",
        "plain_minimal",
      ],
      {
        cwd: process.cwd(),
        encoding: "utf8",
        env: {
          ...process.env,
          PI_SEARCH_EXTENSION_CONFIG: JSON.stringify(envConfig),
          PI_SEARCH_TOOL_INTERFACE: "pyserini-rest-2tool",
          RERANK_TEST_CAPTURE: captured,
          QORRECT_TEST_API_KEY: "test-secret-not-for-snapshot",
        },
      },
    );
    const setupText = readFileSync(join(output, "run_setup.json"), "utf8");
    const setup = JSON.parse(setupText);
    assert.deepEqual(setup.reranking, reranking ?? { mode: "off" });
    assert.deepEqual(JSON.parse(readFileSync(captured, "utf8")), config);
    assert.doesNotMatch(setupText, /test-secret-not-for-snapshot/);
    assert.equal(JSON.parse(readFileSync(join(output, "1.json"), "utf8")).status, "completed");
  });
}
