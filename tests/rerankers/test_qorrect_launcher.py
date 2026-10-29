from pathlib import Path
import os
import subprocess
import tempfile
import unittest


class QorrectLauncherTests(unittest.TestCase):
    script = Path(__file__).resolve().parents[2] / "scripts" / "rerankers" / "qorrect_server.sh"

    def run_fake(self, arguments):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "fake vllm"
            executable.write_text('#!/bin/bash\nprintf "%s\\n" "$PWD" "$@"\nexit 7\n')
            executable.chmod(0o700)
            result = subprocess.run(
                ["bash", str(self.script), *arguments], cwd=directory,
                env={**os.environ, "PIIKA_QORRECT_VLLM": str(executable)},
                capture_output=True, text=True,
            )
            lines = result.stdout.splitlines()
            if lines:
                self.assertEqual(Path(lines[0]).resolve(), Path(directory).resolve())
            return result, lines[1:]

    def test_defaults_argument_forwarding_and_exit_status(self):
        result, arguments = self.run_fake(["models/my checkpoint", "--port", "9001"])
        self.assertEqual(result.returncode, 7)
        self.assertEqual(arguments, [
            "serve", "models/my checkpoint",
            "--host", "127.0.0.1", "--port", "8001",
            "--served-model-name", "qorrect", "--max-model-len", "2048",
            "--generation-config", "vllm",
            "--default-chat-template-kwargs", '{"enable_thinking":false}',
            "--port", "9001",
        ])

    def test_help_forwards_without_model(self):
        result, arguments = self.run_fake(["--help"])
        self.assertEqual(result.returncode, 7)
        self.assertEqual(arguments, ["serve", "--help"])

    def test_missing_model_does_not_start_vllm(self):
        for arguments in [[], ["--port", "8001"]]:
            with self.subTest(arguments=arguments):
                result, forwarded = self.run_fake(arguments)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(forwarded, [])


if __name__ == "__main__":
    unittest.main()
