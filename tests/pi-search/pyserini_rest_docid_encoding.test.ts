import assert from "node:assert/strict";
import { createServer } from "node:http";
import test from "node:test";

import { buildPyseriniRestExtensionConfig } from "../../src/pi-search/config";
import { createPiSearchBackend } from "../../src/pi-search/searcher/adapters/create";

for (const mode of ["standard", "double", "missing", "unauthorized"] as const) {
  void test(`pyserini-rest slash docid encoding: ${mode}`, async () => {
    const docid = "folder/a space%?#文.txt";
    const prefix = "/v1/bright-earth-science/doc/";
    const standard = prefix + encodeURIComponent(docid);
    const double = prefix + encodeURIComponent(encodeURIComponent(docid));
    const requests: string[] = [];
    const server = createServer((request, response) => {
      requests.push(request.url ?? "");
      response.setHeader("content-type", "application/json");
      if (mode === "unauthorized") {
        response.statusCode = 401;
        response.end(JSON.stringify({ error: "Unauthorized" }));
      } else if (
        (mode === "standard" && request.url === standard) ||
        (mode === "double" && request.url === double)
      ) {
        response.end(JSON.stringify({ docid, doc: "first line\nsecond line" }));
      } else {
        response.statusCode = 404;
        response.end(JSON.stringify({ error: "Not found" }));
      }
    });
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    try {
      const address = server.address();
      assert.ok(address && typeof address !== "string");
      const backend = createPiSearchBackend(
        process.cwd(),
        buildPyseriniRestExtensionConfig({
          baseUrl: `http://127.0.0.1:${address.port}`,
          index: "bright-earth-science",
          readMode: "paginated",
        }),
      );
      if (mode === "unauthorized") {
        await assert.rejects(() => backend.readDocument({ docid }), /401|Unauthorized/i);
      } else {
        const result = await backend.readDocument({ docid, offset: 2, limit: 1 });
        assert.equal(result.found, mode !== "missing");
        assert.equal(result.docid, docid);
        if (result.found) {
          assert.equal(result.text, "second line");
        }
      }
      assert.deepEqual(
        requests,
        mode === "standard" || mode === "unauthorized" ? [standard] : [standard, double],
      );
    } finally {
      await new Promise<void>((resolve, reject) =>
        server.close((error) => (error ? reject(error) : resolve())),
      );
    }
  });
}
