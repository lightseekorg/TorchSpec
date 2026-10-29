import pytest
from transformers import AutoTokenizer

from tests.tokenizer_contract.support import load_manifest, resolve_checkpoint


@pytest.fixture(scope="session")
def checkpoints():
    return {name: resolve_checkpoint(name) for name in load_manifest()["profiles"]}


@pytest.fixture(scope="session")
def reference_tokenizers(checkpoints):
    return {
        name: AutoTokenizer.from_pretrained(path, trust_remote_code=False)
        for name, path in checkpoints.items()
    }
