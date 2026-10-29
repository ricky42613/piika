import { Type, type Static } from "typebox";
import { compileJsonValidator, type JsonValidationError } from "./protocol/validation";

const PiSearchSharedRpcBackendConfigSchema = Type.Object(
  {
    kind: Type.Literal("anserini-bm25"),
    transport: Type.Object(
      {
        kind: Type.Literal("tcp"),
        host: Type.String({ minLength: 1 }),
        port: Type.Number({ minimum: 1 }),
      },
      { additionalProperties: false },
    ),
  },
  { additionalProperties: false },
);

const PiSearchLocalStdioBackendConfigSchema = Type.Object(
  {
    kind: Type.Literal("anserini-bm25"),
    transport: Type.Object(
      {
        kind: Type.Literal("stdio"),
        indexPath: Type.String({ minLength: 1 }),
      },
      { additionalProperties: false },
    ),
  },
  { additionalProperties: false },
);

const PiSearchHttpJsonBackendConfigSchema = Type.Object(
  {
    kind: Type.Literal("http-json"),
    capabilities: Type.Object(
      {
        backendId: Type.String({ minLength: 1 }),
        supportsScore: Type.Boolean(),
        supportsSnippets: Type.Boolean(),
        supportsExactTotalHits: Type.Boolean(),
        maxPageSize: Type.Optional(Type.Number({ minimum: 1 })),
        maxReadLimit: Type.Optional(Type.Number({ minimum: 1 })),
      },
      { additionalProperties: false },
    ),
    endpoints: Type.Object(
      {
        searchUrl: Type.String({ minLength: 1 }),
        readDocumentUrl: Type.String({ minLength: 1 }),
      },
      { additionalProperties: false },
    ),
  },
  { additionalProperties: false },
);

const PiSearchPyseriniRestBackendConfigSchema = Type.Object(
  {
    kind: Type.Literal("pyserini-rest"),
    baseUrl: Type.String({ minLength: 1 }),
    index: Type.String({ minLength: 1 }),
    tokenEnv: Type.Optional(Type.String({ minLength: 1 })),
    searchMaxDocLength: Type.Optional(Type.Number({ minimum: 1 })),
    readMode: Type.Optional(Type.Union([Type.Literal("full"), Type.Literal("paginated")])),
  },
  { additionalProperties: false },
);

const PiSearchMockBackendConfigSchema = Type.Object(
  {
    kind: Type.Literal("mock"),
    documents: Type.Array(
      Type.Object(
        {
          docid: Type.String({ minLength: 1 }),
          title: Type.Optional(Type.String()),
          snippet: Type.Optional(Type.String()),
          text: Type.String(),
        },
        { additionalProperties: false },
      ),
    ),
  },
  { additionalProperties: false },
);

const RerankerEndpointFields = {
  baseUrl: Type.String({ pattern: "^https?://", minLength: 1 }),
  model: Type.String({ pattern: "\\S", minLength: 1 }),
  timeoutMs: Type.Optional(Type.Integer({ minimum: 1, maximum: 2_147_483_647 })),
};

const PplxDeciderScorerConfigSchema = Type.Object(
  {
    provider: Type.Literal("pplx-decider"),
    ...RerankerEndpointFields,
    concurrency: Type.Optional(Type.Integer({ minimum: 1, maximum: Number.MAX_SAFE_INTEGER })),
  },
  { additionalProperties: false },
);

const QwenRerankerScorerConfigSchema = Type.Object(
  { provider: Type.Literal("qwen-reranker"), ...RerankerEndpointFields },
  { additionalProperties: false },
);

const QorrectGeneratorConfigSchema = Type.Object(
  {
    provider: Type.Literal("qorrect"),
    ...RerankerEndpointFields,
    tokenEnv: Type.Optional(Type.String({ pattern: "\\S", minLength: 1 })),
    maxTokens: Type.Optional(Type.Integer({ minimum: 1, maximum: Number.MAX_SAFE_INTEGER })),
    temperature: Type.Optional(Type.Number({ minimum: 0, maximum: 2 })),
  },
  { additionalProperties: false },
);

export const PiSearchRerankingConfigSchema = Type.Union([
  Type.Object({ mode: Type.Literal("off") }, { additionalProperties: false }),
  Type.Object(
    {
      mode: Type.Literal("fixed-template"),
      candidateK: Type.Integer({ minimum: 1, maximum: Number.MAX_SAFE_INTEGER }),
      scorer: Type.Union([QwenRerankerScorerConfigSchema, PplxDeciderScorerConfigSchema]),
    },
    { additionalProperties: false },
  ),
  Type.Object(
    {
      mode: Type.Literal("customized"),
      candidateK: Type.Integer({ minimum: 1, maximum: Number.MAX_SAFE_INTEGER }),
      generator: QorrectGeneratorConfigSchema,
      scorer: PplxDeciderScorerConfigSchema,
    },
    { additionalProperties: false },
  ),
]);

export type PiSearchRerankingConfig = Static<typeof PiSearchRerankingConfigSchema>;

export const PiSearchExtensionConfigSchema = Type.Object(
  {
    backend: Type.Union([
      PiSearchSharedRpcBackendConfigSchema,
      PiSearchLocalStdioBackendConfigSchema,
      PiSearchHttpJsonBackendConfigSchema,
      PiSearchPyseriniRestBackendConfigSchema,
      PiSearchMockBackendConfigSchema,
    ]),
    reranking: Type.Optional(PiSearchRerankingConfigSchema),
  },
  { additionalProperties: false },
);

export type PiSearchExtensionConfig = Static<typeof PiSearchExtensionConfigSchema>;

const piSearchExtensionConfigValidator = compileJsonValidator(PiSearchExtensionConfigSchema);

function formatValidationErrors(errors: JsonValidationError[]): string {
  if (!errors || errors.length === 0) {
    return "schema validation failed without detailed errors.";
  }
  return errors
    .map((error) => {
      const path = error.instancePath || "/";
      return `${path} ${error.message ?? "is invalid"}`.trim();
    })
    .join("; ");
}

export function buildAnseriniBm25TcpExtensionConfig(options: {
  host: string;
  port: number;
}): PiSearchExtensionConfig {
  return {
    backend: {
      kind: "anserini-bm25",
      transport: {
        kind: "tcp",
        host: options.host,
        port: options.port,
      },
    },
  };
}

export function buildAnseriniBm25StdioExtensionConfig(options: {
  indexPath: string;
}): PiSearchExtensionConfig {
  return {
    backend: {
      kind: "anserini-bm25",
      transport: {
        kind: "stdio",
        indexPath: options.indexPath,
      },
    },
  };
}

export function buildHttpJsonExtensionConfig(options: {
  capabilities: {
    backendId: string;
    supportsScore: boolean;
    supportsSnippets: boolean;
    supportsExactTotalHits: boolean;
    maxPageSize?: number;
    maxReadLimit?: number;
  };
  searchUrl: string;
  readDocumentUrl: string;
}): PiSearchExtensionConfig {
  return {
    backend: {
      kind: "http-json",
      capabilities: options.capabilities,
      endpoints: {
        searchUrl: options.searchUrl,
        readDocumentUrl: options.readDocumentUrl,
      },
    },
  };
}

export function buildPyseriniRestExtensionConfig(options: {
  baseUrl: string;
  index: string;
  tokenEnv?: string;
  searchMaxDocLength?: number;
  readMode?: "full" | "paginated";
}): PiSearchExtensionConfig {
  return {
    backend: {
      kind: "pyserini-rest",
      baseUrl: options.baseUrl,
      index: options.index,
      tokenEnv: options.tokenEnv,
      searchMaxDocLength: options.searchMaxDocLength,
      readMode: options.readMode,
    },
  };
}

export function buildMockExtensionConfig(options: {
  documents: Array<{
    docid: string;
    title?: string;
    snippet?: string;
    text: string;
  }>;
}): PiSearchExtensionConfig {
  return {
    backend: {
      kind: "mock",
      documents: options.documents,
    },
  };
}

const rerankingConfigValidator = compileJsonValidator(PiSearchRerankingConfigSchema);

export function parsePiSearchRerankingConfig(text: string): PiSearchRerankingConfig {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    // Do not echo the input: an invalid file could contain inline credentials.
    throw new Error("Failed to parse reranking config: expected valid JSON.");
  }
  if (rerankingConfigValidator.check(value)) return value;
  throw new Error(
    `Invalid reranking config: ${formatValidationErrors(rerankingConfigValidator.errors(value))}`,
  );
}

export function parsePiSearchExtensionConfig(text: string): PiSearchExtensionConfig {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch (error) {
    throw new Error(`Failed to parse PI_SEARCH_EXTENSION_CONFIG: ${text}\n${String(error)}`);
  }
  if (piSearchExtensionConfigValidator.check(value)) {
    return value;
  }
  throw new Error(
    `Invalid PI_SEARCH_EXTENSION_CONFIG: ${formatValidationErrors(
      piSearchExtensionConfigValidator.errors(value),
    )}`,
  );
}

export function resolvePiSearchExtensionConfigFromEnv(
  env: NodeJS.ProcessEnv = process.env,
): PiSearchExtensionConfig {
  const raw = env.PI_SEARCH_EXTENSION_CONFIG?.trim();
  if (!raw) {
    throw new Error(
      "Missing PI_SEARCH_EXTENSION_CONFIG. The pi-search extension now requires explicit backend config from its caller.",
    );
  }
  return parsePiSearchExtensionConfig(raw);
}
