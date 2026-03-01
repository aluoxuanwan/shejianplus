from __future__ import annotations

import json

from shejianplus.core.inference_session_exporter import export_inference_session_timeseries


def test_export_inference_session_writes_expected_files(tmp_path) -> None:
    result = export_inference_session_timeseries(
        export_root=tmp_path,
        session_meta={"video": "demo.avi"},
        raw_rows=[{"frame_idx": 0, "target": "archery"}],
        filtered_rows=[{"frame_idx": 0, "target": "archery"}],
    )

    assert result.raw_csv.exists()
    assert result.filtered_csv.exists()
    assert result.meta_json.exists()
    payload = json.loads(result.meta_json.read_text(encoding="utf-8"))
    assert payload["summary"]["raw_rows"] == 1