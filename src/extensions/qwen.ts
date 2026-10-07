import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export default function (pi: ExtensionAPI) {
  pi.registerProvider("qwen-local", {
    baseUrl: "http://192.168.31.44:18012/v1",
    apiKey: "local", // Placeholder for the keyless vLLM server; Pi requires configured auth.
    api: "openai-completions",
    models: [
      {
        id: "qwen38-27b-nvfp4",
        name: "Qwen3.8 27B NVFP4 (Local)",
        reasoning: true,
        input: ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: 262144, // Reported by /v1/models.
        maxTokens: 16384, // Conservative per-response output limit.
        compat: {
          supportsStore: false,
          supportsDeveloperRole: false,
          supportsReasoningEffort: false,
          supportsStrictMode: false,
          maxTokensField: "max_tokens",
          thinkingFormat: "qwen-chat-template",
        },
      },
    ],
  });
}
