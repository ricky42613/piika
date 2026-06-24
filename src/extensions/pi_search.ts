import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { createRepoPiSearchBackend } from "../search-providers/anserini/pi_search_backend_factory";
import { registerPiSearchExtension, registerSelfBuiltSearchExtension } from "../pi-search/extension";

export default function (pi: ExtensionAPI) {
  if (process.env.PI_SEARCH_TOOL_INTERFACE === "self-built") {
    registerSelfBuiltSearchExtension(pi);
  } else {
    registerPiSearchExtension(pi, {
      createBackend: createRepoPiSearchBackend,
    });
  }
  
  pi.registerProvider("Qwen", {
    name: "Local Qwen",
    baseUrl: process.env.OPENAI_BASE_URL ?? "http://127.0.0.1:9002/v1",
    apiKey: process.env.OPENAI_API_KEY ?? "my_token",
    api: "openai-completions",
    models: [
      {
        id: "qwen36-27b",
        name: "qwen3.6-27B",
        reasoning: false,
        maxTokens: 8192,
        input: ["text"],
        cost: {
          input: 0,
          output: 0,
          cacheRead: 0,
          cacheWrite: 0,
        },
        contextWindow: 131072,
      },
    ],
  });
}
