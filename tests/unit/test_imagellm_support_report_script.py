from __future__ import annotations

import json

from scripts.diagnostics.report_imagellm_support import main


def test_report_script_supports_human_and_machine_views(capsys) -> None:
    assert main(["--format", "markdown"]) == 0
    markdown = capsys.readouterr().out
    assert markdown.count("\n|") == 65

    assert main(["--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 64
    assert {item["status"] for item in payload} == {"active", "declared"}


def test_report_script_fails_closed_on_unmet_status(capsys) -> None:
    assert main(["--require_status", "active"]) == 2
    assert "declared" in capsys.readouterr().out

    assert main(["--phase", "forward", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 73
    assert {item["status"] for item in payload} == {"active", "declared"}
