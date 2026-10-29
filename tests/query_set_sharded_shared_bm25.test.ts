import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, rmSync, existsSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import {
  mergeShardOutputs,
  readShardRerankingConfig,
} from "../src/orchestration/query_set_sharded_shared_bm25";

const reranking = {
  mode: "fixed-template",
  candidateK: 50,
  scorer: {
    provider: "pplx-decider",
    baseUrl: "http://localhost:8002/v1",
    model: "pplx-decider-flash-v1.2",
  },
};

void test("mergeShardOutputs synthesizes merged metadata instead of failing on shard-local metadata conflicts", () => {
  const root = mkdtempSync(join(tmpdir(), "sharded-merge-"));
  const queryPath = join(root, "queries.tsv");
  const qrelsPath = join(root, "qrels.txt");
  const indexPath = join(root, "index");
  const shardOutputRoot = join(root, "shard-runs");
  const mergedOutputDir = join(root, "merged");

  writeFileSync(queryPath, "q1\tfirst query\nq2\tsecond query\n", "utf8");
  writeFileSync(qrelsPath, "q1 0 d1 1\nq2 0 d2 1\n", "utf8");
  mkdirSync(indexPath, { recursive: true });

  for (const shardName of ["shard_01", "shard_02"]) {
    mkdirSync(join(shardOutputRoot, shardName, "raw-events"), { recursive: true });
    mkdirSync(join(shardOutputRoot, shardName, "stderr"), { recursive: true });
  }

  writeFileSync(join(shardOutputRoot, "shard_01", "q1.json"), '{"query_id":"q1"}\n', "utf8");
  writeFileSync(join(shardOutputRoot, "shard_02", "q2.json"), '{"query_id":"q2"}\n', "utf8");
  writeFileSync(join(shardOutputRoot, "shard_01", "raw-events", "q1.jsonl"), "{}\n", "utf8");
  writeFileSync(join(shardOutputRoot, "shard_02", "raw-events", "q2.jsonl"), "{}\n", "utf8");
  writeFileSync(join(shardOutputRoot, "shard_01", "stderr", "q1.log"), "stderr-1\n", "utf8");
  writeFileSync(join(shardOutputRoot, "shard_02", "stderr", "q2.log"), "stderr-2\n", "utf8");

  writeFileSync(
    join(shardOutputRoot, "shard_01", "benchmark_manifest_snapshot.json"),
    JSON.stringify({ query_path: join(root, "shard-queries", "shard_01.tsv") }, null, 2) + "\n",
    "utf8",
  );
  writeFileSync(
    join(shardOutputRoot, "shard_02", "benchmark_manifest_snapshot.json"),
    JSON.stringify({ query_path: join(root, "shard-queries", "shard_02.tsv") }, null, 2) + "\n",
    "utf8",
  );
  writeFileSync(
    join(shardOutputRoot, "shard_01", "run_setup.json"),
    JSON.stringify(
      { queryFile: join(root, "shard-queries", "shard_01.tsv"), totalQueries: "1", reranking },
      null,
      2,
    ) + "\n",
    "utf8",
  );
  writeFileSync(
    join(shardOutputRoot, "shard_02", "run_setup.json"),
    JSON.stringify(
      { queryFile: join(root, "shard-queries", "shard_02.tsv"), totalQueries: "1", reranking },
      null,
      2,
    ) + "\n",
    "utf8",
  );

  mergeShardOutputs(
    {
      benchmarkId: "benchmark-template",
      backendKind: "shared-bm25",
      querySetId: "dev",
      model: "openai-codex/gpt-5.4-mini",
      piSearchPromptVariant: "plain_minimal",
      outputMode: "answer",
      outputModes: ["answer"],
      toolInterface: "pyserini-rest-2tool",
      rankedListDepth: 1000,
      outputDir: root,
      timeoutSeconds: 300,
      thinking: "medium",
      piBin: "pi",
      extensionPath: "src/extensions/pi_search.ts",
      queryPath,
      qrelsPath,
      indexPath,
      shardCount: 2,
      host: "127.0.0.1",
      port: 12345,
      outputRoot: root,
      logDir: join(root, "logs"),
      bm25LogPath: join(root, "logs", "bm25.log"),
      shardQueryDir: join(root, "shard-queries"),
      shardOutputRoot,
      mergedOutputDir,
      controlDir: join(root, "control"),
      retryRequestPath: join(root, "control", "retry-request.json"),
      retryApprovalPath: join(root, "control", "retry-approval.json"),
      autoSummarizeOnMerge: false,
      autoEvaluateOnMerge: false,
      evaluateForce: false,
      evaluateLimit: 0,
      maxShardAttempts: 2,
      shardRetryMode: "manual",
      modelTag: "gpt54mini",
      runStamp: "20260416_000000",
      resolvedIndexPath: indexPath,
    },
    ["shard_01", "shard_02"],
    2,
  );

  assert.equal(readFileSync(join(mergedOutputDir, "q1.json"), "utf8"), '{"query_id":"q1"}\n');
  assert.equal(readFileSync(join(mergedOutputDir, "q2.json"), "utf8"), '{"query_id":"q2"}\n');
  assert.equal(readFileSync(join(mergedOutputDir, "raw-events", "q1.jsonl"), "utf8"), "{}\n");
  assert.equal(readFileSync(join(mergedOutputDir, "stderr", "q2.log"), "utf8"), "stderr-2\n");

  const runSetup = JSON.parse(readFileSync(join(mergedOutputDir, "run_setup.json"), "utf8")) as {
    queryFile: string;
    totalQueries: string;
    slice: string;
    reranking: unknown;
  };
  assert.equal(runSetup.queryFile, queryPath);
  assert.equal(runSetup.totalQueries, "2");
  assert.equal(runSetup.slice, "dev");
  assert.deepEqual(runSetup.reranking, reranking);

  const manifest = JSON.parse(
    readFileSync(join(mergedOutputDir, "benchmark_manifest_snapshot.json"), "utf8"),
  ) as {
    benchmark_id: string;
    query_set_id: string;
    query_path: string;
  };
  assert.equal(manifest.benchmark_id, "benchmark-template");
  assert.equal(manifest.query_set_id, "dev");
  assert.equal(manifest.query_path, queryPath);
});

void test("shard reranking metadata handles legacy and explicit off configs", (t) => {
  const root = mkdtempSync(join(tmpdir(), "sharded-rerank-off-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  for (const name of ["legacy", "off", "missing"]) mkdirSync(join(root, name));
  writeFileSync(join(root, "legacy", "run_setup.json"), "{}");
  writeFileSync(
    join(root, "off", "run_setup.json"),
    JSON.stringify({ reranking: { mode: "off" } }),
  );
  assert.deepEqual(readShardRerankingConfig(root, ["legacy", "off", "missing"]), { mode: "off" });
});

void test("shard reranking equality ignores JSON key order", (t) => {
  const root = mkdtempSync(join(tmpdir(), "sharded-rerank-order-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  for (const name of ["a", "b"]) mkdirSync(join(root, name));
  writeFileSync(join(root, "a", "run_setup.json"), JSON.stringify({ reranking }));
  writeFileSync(
    join(root, "b", "run_setup.json"),
    JSON.stringify({
      reranking: {
        scorer: reranking.scorer,
        candidateK: 50,
        mode: "fixed-template",
      },
    }),
  );
  assert.deepEqual(readShardRerankingConfig(root, ["a", "b"]), reranking);
});

void test("mismatched reranking configs fail before creating merged outputs", (t) => {
  const root = mkdtempSync(join(tmpdir(), "sharded-rerank-mismatch-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  for (const name of ["a", "b"]) mkdirSync(join(root, name));
  writeFileSync(join(root, "a", "run_setup.json"), JSON.stringify({ reranking }));
  const mergedOutputDir = join(root, "merged");
  for (const other of [{ mode: "off" }, { ...reranking, candidateK: 100 }]) {
    writeFileSync(join(root, "b", "run_setup.json"), JSON.stringify({ reranking: other }));
    const plan = { shardOutputRoot: root, mergedOutputDir } as Parameters<
      typeof mergeShardOutputs
    >[0];
    assert.throws(() => mergeShardOutputs(plan, ["a", "b"], 2), /different reranking configs/);
    assert.equal(existsSync(mergedOutputDir), false);
  }
});

void test("malformed shard reranking metadata is rejected", (t) => {
  const root = mkdtempSync(join(tmpdir(), "sharded-rerank-invalid-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  mkdirSync(join(root, "a"));
  for (const setup of [null, [], { reranking: null }, { reranking: { mode: "unknown" } }]) {
    writeFileSync(join(root, "a", "run_setup.json"), JSON.stringify(setup));
    assert.throws(() => readShardRerankingConfig(root, ["a"]), /Invalid/);
  }
});

void test("mergeShardOutputs writes merged ranked-list TREC run file", () => {
  const root = mkdtempSync(join(tmpdir(), "sharded-merge-ranked-list-"));
  const queryPath = join(root, "queries.tsv");
  const qrelsPath = join(root, "qrels.txt");
  const indexPath = join(root, "index");
  const shardOutputRoot = join(root, "shard-runs");
  const mergedOutputDir = join(root, "merged");

  writeFileSync(queryPath, "q1\tfirst query\nq2\tsecond query\n", "utf8");
  writeFileSync(qrelsPath, "q1 0 d1 1\nq2 0 d3 1\n", "utf8");
  mkdirSync(indexPath, { recursive: true });
  mkdirSync(join(shardOutputRoot, "shard_01"), { recursive: true });
  mkdirSync(join(shardOutputRoot, "shard_02"), { recursive: true });
  writeFileSync(
    join(shardOutputRoot, "shard_01", "q1.json"),
    JSON.stringify({ query_id: "q1", ranked_docids: ["d2", "d1"] }) + "\n",
    "utf8",
  );
  writeFileSync(
    join(shardOutputRoot, "shard_02", "q2.json"),
    JSON.stringify({ query_id: "q2", ranked_docids: ["d3"] }) + "\n",
    "utf8",
  );

  mergeShardOutputs(
    {
      benchmarkId: "benchmark-template",
      backendKind: "shared-bm25",
      querySetId: "dev",
      model: "openai-codex/gpt-5.4-mini",
      piSearchPromptVariant: "plain_minimal",
      outputMode: "ranked_list",
      outputModes: ["ranked_list"],
      toolInterface: "pyserini-rest-2tool",
      rankedListDepth: 1000,
      outputDir: root,
      timeoutSeconds: 300,
      thinking: "medium",
      piBin: "pi",
      extensionPath: "src/extensions/pi_search.ts",
      queryPath,
      qrelsPath,
      indexPath,
      shardCount: 2,
      host: "127.0.0.1",
      port: 12345,
      outputRoot: root,
      logDir: join(root, "logs"),
      bm25LogPath: join(root, "logs", "bm25.log"),
      shardQueryDir: join(root, "shard-queries"),
      shardOutputRoot,
      mergedOutputDir,
      controlDir: join(root, "control"),
      retryRequestPath: join(root, "control", "retry-request.json"),
      retryApprovalPath: join(root, "control", "retry-approval.json"),
      autoSummarizeOnMerge: false,
      autoEvaluateOnMerge: false,
      evaluateForce: false,
      evaluateLimit: 0,
      maxShardAttempts: 2,
      shardRetryMode: "manual",
      modelTag: "gpt54mini",
      runStamp: "20260416_000000",
      resolvedIndexPath: indexPath,
    },
    ["shard_01", "shard_02"],
    2,
  );

  assert.equal(
    readFileSync(join(mergedOutputDir, "ranked_list.trec"), "utf8"),
    "q1 Q0 d2 1 2 pi-agent\nq1 Q0 d1 2 1 pi-agent\nq2 Q0 d3 1 1 pi-agent\n",
  );
});
