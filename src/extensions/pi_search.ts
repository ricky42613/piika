import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { createRepoPiSearchBackend } from "../search-providers/anserini/pi_search_backend_factory";
import { registerPiSearchExtension } from "../pi-search/extension";
import registerQwenLocalProvider from "./qwen";

export default function (pi: ExtensionAPI) {
  registerQwenLocalProvider(pi);
  registerPiSearchExtension(pi, {
    createBackend: createRepoPiSearchBackend,
  });
}
