import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import test from "node:test";

for (const mode of ["answer", "ranked_list", "answer+ranked_list"]) {
  void test(`timeout steer and retrieval blocks preserve ${mode} output`, () => {
    const stdout = execFileSync(
      process.execPath,
      [
        "--import",
        "tsx",
        "--input-type=module",
        "-e",
        `
      import assert from 'node:assert/strict';
      import { registerPiSearchExtension } from './src/pi-search/extension.ts';
      const handlers = new Map();
      const messages = [];
      registerPiSearchExtension({
        on: (name, handler) => handlers.set(name, handler),
        registerTool: () => {},
        sendUserMessage: (text, options) => messages.push({ text, options }),
      }, {
        resolveConfig: () => ({ backend: { kind: 'mock', documents: [] } }),
        toolInterface: 'pyserini-rest-2tool',
      });
      try {
        await handlers.get('agent_start')({}, { isIdle: () => false });
        await new Promise(resolve => setTimeout(resolve, 50));
        assert.equal(messages.length, 1);
        assert.deepEqual(messages[0].options, { deliverAs: 'steer' });
        for (const toolName of ['search', 'read_document', 'read_search_results']) {
          const result = await handlers.get('tool_call')({ toolName });
          assert.equal(result.block, true);
          assert.equal(result.reason, messages[0].text);
        }
        assert.equal(await handlers.get('tool_call')({ toolName: 'other' }), undefined);
        await handlers.get('agent_end')();
        assert.equal(await handlers.get('tool_call')({ toolName: 'search' }), undefined);
        console.log(JSON.stringify(messages[0].text));
      } finally {
        await handlers.get('session_shutdown')();
      }
    `,
      ],
      {
        cwd: process.cwd(),
        encoding: "utf8",
        timeout: 10000,
        env: {
          ...process.env,
          TIMEOUT_SECONDS: "0.01",
          OUTPUT_MODE: mode,
          RANKED_LIST_DEPTH: "9",
          RANKED_LIST_COUNT: "3",
        },
        stdio: ["ignore", "pipe", "pipe"],
      },
    );
    const message = JSON.parse(stdout) as string;
    assert.equal(message.includes("Exact Answer:"), mode.includes("answer"));
    assert.equal(message.includes("Ranked List:"), mode.includes("ranked_list"));
    if (mode.includes("ranked_list"))
      assert.match(message, /exactly 3 unique document ids \(maximum depth 9\)/);
  });
}
