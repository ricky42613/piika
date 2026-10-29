import json
from pathlib import Path
import os
import subprocess
import tempfile
import unittest


class QwenRerankerLauncherTests(unittest.TestCase):
    script = Path(__file__).resolve().parents[2] / "scripts" / "rerankers" / "qwen_reranker_server.sh"

    def run_fake(self, arguments):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "fake vllm"
            executable.write_text('#!/bin/bash\nprintf "%s\\n" "$PWD" "$@"\nexit 7\n')
            executable.chmod(0o700)
            result = subprocess.run(
                ["bash", str(self.script), *arguments], cwd=directory,
                env={**os.environ, "PIIKA_QWEN_RERANKER_VLLM": str(executable)},
                capture_output=True, text=True,
            )
            lines = result.stdout.splitlines()
            if lines:
                self.assertEqual(Path(lines[0]).resolve(), Path(directory).resolve())
            return result, lines[1:]

    def test_original_checkpoint_pooling_overrides_and_template(self):
        result, arguments = self.run_fake(["models/my reranker", "--port", "9003"])
        self.assertEqual(result.returncode, 7)
        self.assertEqual(arguments[:2], ["serve", "models/my reranker"])
        self.assertEqual(arguments[arguments.index("--host") + 1], "127.0.0.1")
        self.assertEqual(arguments[arguments.index("--port") + 1], "8003")
        self.assertEqual(arguments[arguments.index("--runner") + 1], "pooling")
        self.assertEqual(arguments[arguments.index("--served-model-name") + 1], "qwen3-reranker")
        self.assertEqual(json.loads(arguments[arguments.index("--hf-overrides") + 1]), {
            "architectures": ["Qwen3ForSequenceClassification"],
            "classifier_from_token": ["no", "yes"],
            "is_original_qwen3_reranker": True,
        })
        template = Path(arguments[arguments.index("--chat-template") + 1])
        self.assertEqual(template, self.script.parent / "qwen3_reranker.jinja")
        self.assertTrue(template.is_file())
        self.assertEqual(arguments[-2:], ["--port", "9003"])

    def test_help_does_not_require_model(self):
        result, arguments = self.run_fake(["--help"])
        self.assertEqual(result.returncode, 7)
        self.assertEqual(arguments, ["serve", "--help"])

    def test_missing_model_does_not_start_vllm(self):
        for arguments in [[], ["--port", "8003"]]:
            result, forwarded = self.run_fake(arguments)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(forwarded, [])


if __name__ == "__main__":
    unittest.main()
