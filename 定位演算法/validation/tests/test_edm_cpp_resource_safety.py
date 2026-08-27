from pathlib import Path


CPP_ROOT = (
    Path(__file__).resolve().parents[2]
    / "deploy_code"
    / "runtime"
    / "EDM"
    / "deploy"
    / "edm_onnx_cpp"
)


def test_edm_inference_uses_raii_for_per_frame_memory_and_session() -> None:
    implementation = (CPP_ROOT / "edm" / "edm.cpp").read_text(encoding="utf-8")
    header = (CPP_ROOT / "edm" / "edm.h").read_text(encoding="utf-8")

    assert "new float" not in implementation
    assert "delete session" not in implementation
    assert "std::vector<float> oneInput" in implementation
    assert "std::unique_ptr<Ort::Session> session" in header


def test_edm_demo_stops_before_accessing_missing_arguments() -> None:
    source = (CPP_ROOT / "demo.cpp").read_text(encoding="utf-8")
    argc_guard = source.index("if (argc != 5)")
    first_argv_access = source.index("argv[1]")

    assert "return 1;" in source[argc_guard:first_argv_access]
