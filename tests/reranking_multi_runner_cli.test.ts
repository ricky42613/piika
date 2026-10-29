import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { chmodSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

const scorer = { provider: "pplx-decider", baseUrl: "http://127.0.0.1:8002/v1", model: "decider" };
const configs = [
  { mode: "off" },
  { mode: "fixed-template", candidateK: 50, scorer },
  {
    mode: "customized",
    candidateK: 50,
    scorer,
    generator: {
      provider: "qorrect",
      baseUrl: "http://127.0.0.1:8001/v1",
      model: "qorrect",
      tokenEnv: "TEST_QORRECT_TOKEN",
    },
  },
];

for (const mode of ["shared", "sharded"]) {
  for (const reranking of configs) {
    void test(`${mode} forwards ${reranking.mode} reranking to every query runner`, (t) => {
      const root = mkdtempSync(join(tmpdir(), `piika-${mode}-rerank-`));
      t.after(() => rmSync(root, { recursive: true, force: true }));
      const queries = join(root, "queries.tsv");
      const qrels = join(root, "qrels.txt");
      const configPath = join(root, "reranking.json");
      const output = join(root, "run");
      const fakePi = join(root, "fake-pi.mjs");
      writeFileSync(queries, "1\talpha\n2\tbeta\n");
      writeFileSync(qrels, "1 0 d1 1\n2 0 d2 1\n");
      writeFileSync(configPath, JSON.stringify(reranking));
      writeFileSync(
        fakePi,
        `#!/usr/bin/env node
import { writeFileSync } from 'node:fs';
writeFileSync(process.env.TEST_CAPTURE_PREFIX + process.env.PI_SEARCH_ORIGINAL_QUESTION + '.json', JSON.stringify({
  config: JSON.parse(process.env.PI_SEARCH_EXTENSION_CONFIG), toolInterface: process.env.PI_SEARCH_TOOL_INTERFACE,
}));
console.log(JSON.stringify({type:'session'}));
console.log(JSON.stringify({type:'message_end',message:{role:'assistant',content:[{type:'text',text:'Answer.'}]}}));
console.log(JSON.stringify({type:'agent_end'}));
`,
      );
      chmodSync(fakePi, 0o755);
      const backend = { kind: "pyserini-rest", baseUrl: "http://127.0.0.1:8081", index: "test" };
      const env = {
        ...process.env,
        BENCHMARK: "benchmark-template",
        QUERY_SET: "dev",
        QUERY_FILE: queries,
        QRELS_FILE: qrels,
        OUTPUT_DIR: output,
        LOG_DIR: join(root, "logs"),
        PI_BIN: fakePi,
        MODEL: "openai-codex/gpt-5.4-mini",
        PI_SEARCH_TOOL_INTERFACE: "pi-serini-3tool",
        PI_SEARCH_EXTENSION_CONFIG: JSON.stringify({
          backend,
          reranking: reranking.mode === "off" ? configs[1] : { mode: "off" },
        }),
        TEST_CAPTURE_PREFIX: join(root, "capture-"),
        TEST_QORRECT_TOKEN: "test-secret",
        PI_SERINI_DRY_RUN: "0",
        AUTO_SUMMARIZE_ON_MERGE: "0",
        AUTO_EVALUATE_ON_MERGE: "0",
      };
      const args = [
        "bin/piika.js",
        "run",
        "--mode",
        mode,
        "--benchmark",
        "benchmark-template",
        "--query-set",
        "dev",
        "--query-file",
        queries,
        "--qrels",
        qrels,
        "--output-dir",
        output,
        "--tool-interface",
        "pyserini-rest-2tool",
        "--reranking-config",
        configPath,
      ];
      const options = { cwd: process.cwd(), encoding: "utf8" as const, env, timeout: 30000 };
      if (mode === "shared") {
        // Exercise the exact child command without starting a real Java BM25 daemon.
        const dryRun = execFileSync(process.execPath, [...args, "--dry-run"], options);
        const commandLine = dryRun.split("\n").find((line) => line.startsWith("COMMAND_JSON="));
        assert.ok(commandLine);
        const [command, ...childArgs] = JSON.parse(
          commandLine.slice("COMMAND_JSON=".length),
        ) as string[];
        assert.deepEqual(childArgs.slice(-2), ["--reranking-config", configPath]);
        execFileSync(command, childArgs, {
          ...options,
          env: { ...env, PI_SEARCH_TOOL_INTERFACE: "pyserini-rest-2tool" },
        });
      } else {
        execFileSync(
          process.execPath,
          [...args, "--shard-count", "2", "--max-shard-attempts", "1"],
          options,
        );
      }
      for (const query of ["alpha", "beta"]) {
        const captured = JSON.parse(readFileSync(join(root, `capture-${query}.json`), "utf8"));
        assert.deepEqual(captured.config, { backend, reranking });
        assert.equal(captured.toolInterface, "pyserini-rest-2tool");
      }
      const setupPaths =
        mode === "shared"
          ? [join(output, "run_setup.json")]
          : [
              join(output, "merged", "run_setup.json"),
              join(output, "shard-runs", "shard_01", "run_setup.json"),
              join(output, "shard-runs", "shard_02", "run_setup.json"),
            ];
      for (const path of setupPaths) {
        const text = readFileSync(path, "utf8");
        assert.deepEqual(JSON.parse(text).reranking, reranking);
        assert.doesNotMatch(text, /test-secret/);
      }
      if (reranking.mode !== "off") {
        assert.throws(
          () =>
            execFileSync(
              process.execPath,
              [...args, "--tool-interface", "pi-serini-3tool", "--dry-run"],
              options,
            ),
          /requires pyserini-rest-2tool/,
        );
      }
    });
  }
}
