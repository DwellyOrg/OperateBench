"""Expand literal campaign captures without importing production policy."""

import json
from copy import deepcopy
from pathlib import Path


def load_campaign_profile_goldens():
    data = json.loads(
        (Path(__file__).parent / "fixtures/campaign_profile_goldens.json").read_text()
    )
    if data["schema_version"] != 1:
        raise ValueError("unsupported campaign golden schema")
    http = {}
    for key, cell in data["http"].items():
        prompt = deepcopy(data["prompt"])
        prompt["observation"]["operation_instance_id"] = cell["operation_instance_id"]
        http[key] = {
            "mapping": cell["mapping"],
            "prompt": prompt,
            "settings": cell["settings"],
            "wire": cell["wire"],
        }
    return {
        "http": http,
        "registry": {
            key: [deepcopy(data["registry_rows"][ref]) for ref in refs]
            for key, refs in data["registry"].items()
        },
    }
