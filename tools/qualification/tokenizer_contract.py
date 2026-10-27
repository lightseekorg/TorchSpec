"""Required tokenizer gate. Any skipped/xfail/deselected case fails the run."""

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

import pytest

from tests.tokenizer_contract.support import load_manifest, resolve_checkpoint


class RequiredCases:
    def __init__(self):
        self.invalid = []
        self.passed = 0

    def pytest_deselected(self, items):
        self.invalid.extend(item.nodeid for item in items)

    def pytest_runtest_logreport(self, report):
        if report.skipped or hasattr(report, "wasxfail"):
            self.invalid.append(report.nodeid)
        if report.when == "call" and report.passed:
            self.passed += 1

    def pytest_collectreport(self, report):
        if report.skipped:
            self.invalid.append(report.nodeid)

    def pytest_sessionfinish(self, session, exitstatus):
        if self.invalid or not self.passed:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["hf", "smg"], required=True)
    parser.add_argument(
        "--encode-path",
        help="Require an exact encoder, e.g. v1; default uses the manifest's per-model paths",
    )
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    if args.backend == "smg" and not args.encode_path:
        parser.error("--backend smg requires --encode-path to detect fallback")
    manifest = load_manifest()
    versions = {name: importlib.metadata.version(name) for name in manifest["reference_versions"]}
    if versions != manifest["reference_versions"]:
        raise RuntimeError(
            f"Unqualified reference versions: {versions}; expected {manifest['reference_versions']}"
        )
    info = {
        "backend": args.backend,
        "platform": platform.platform(),
        "versions": versions,
        "profiles": manifest["profiles"],
    }
    files = ["tests/tokenizer_contract/test_reference.py"]
    if args.backend == "smg":
        from smg_tokenizer import Tokenizer, build_features

        info.update(build_features=build_features, required_encode_path=args.encode_path)
        info["versions"]["smg-tokenizer"] = importlib.metadata.version("smg-tokenizer")
        info["actual_encode_paths"] = {
            name: Tokenizer(resolve_checkpoint(name)).encode_path for name in manifest["profiles"]
        }
        expected_paths = {
            name: profile["default_encode_path"]
            if args.encode_path == "default"
            else args.encode_path
            for name, profile in manifest["profiles"].items()
        }
        if args.encode_path == "default" and build_features != "default":
            raise RuntimeError("The default encoder profile requires a default-feature binding")
        if info["actual_encode_paths"] != expected_paths:
            raise RuntimeError(f"Encoder fallback: {info['actual_encode_paths']}")
        os.environ["TORCHSPEC_REQUIRED_ENCODE_PATHS"] = json.dumps(expected_paths)
        files.append("tests/tokenizer_contract/test_smg.py")
    plugin = RequiredCases()
    # Do not inherit caller flags that could select only passing cases.
    os.environ.pop("PYTEST_ADDOPTS", None)
    code = pytest.main(
        [
            *files,
            "-q",
            "-o",
            "addopts=",
            "--strict-markers",
            "--timeout=180",
            "--timeout-method=thread",
        ],
        plugins=[plugin],
    )
    info.update(exit_code=int(code), passed=plugin.passed, skipped_or_deselected=plugin.invalid)
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(info, indent=2) + "\n")
    return int(code)


if __name__ == "__main__":
    sys.exit(main())
