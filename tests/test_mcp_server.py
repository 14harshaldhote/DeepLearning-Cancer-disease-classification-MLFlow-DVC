import asyncio

import mcp_server


def test_tools_registered():
    tools = asyncio.run(mcp_server.mcp.list_tools())
    assert {t.name for t in tools} >= {"classify_ct_scan", "model_card"}


def test_classify_tool_writes_heatmap(tmp_path, samples):
    image = tmp_path / "scan.png"
    image.write_bytes(samples[0].read_bytes())
    result = mcp_server.classify_ct_scan(str(image))
    assert result["label"] in {"Normal", "Adenocarcinoma Cancer"}
    assert (tmp_path / "scan_gradcam.png").exists()


def test_classify_tool_missing_file():
    assert "error" in mcp_server.classify_ct_scan("/no/such/file.png")
