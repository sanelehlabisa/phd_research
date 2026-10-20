"""Offline packaging checks; no account access, publishing or GPU required."""

import ast
import functools
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType
from unittest.mock import Mock
from zipfile import ZipFile

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def exporter():
    return load_module("space_export_under_test", ROOT / "deployment/prepare_space.py")


@pytest.fixture
def source(tmp_path, exporter):
    template = tmp_path / "deployment/huggingface_space"
    template.parent.mkdir()
    shutil.copytree(ROOT / "deployment/huggingface_space", template)
    for name in ("demo.py", "requirements.txt"):
        shutil.copy2(ROOT / name, tmp_path / name)
    (tmp_path / "src").mkdir()
    for name in exporter.SOURCE_FILES:
        shutil.copy2(ROOT / "src" / name, tmp_path / "src" / name)
    assets = tmp_path / "demo_assets/winner"
    assets.mkdir(parents=True)
    weights = b"synthetic checkpoint for packaging tests only"
    (assets / "best_model.pth").write_bytes(weights)
    (assets / "config.json").write_text(
        json.dumps({"checkpoint_sha256": hashlib.sha256(weights).hexdigest()}),
        encoding="utf-8",
    )
    examples = tmp_path / "demo_assets/examples"
    examples.mkdir()
    for name in exporter.EXAMPLE_FILES:
        (examples / name).write_bytes(b"synthetic example fixture")
    # These files must not follow the app into a public deployment.
    for name in (
        "runs/private.txt",
        "datasets/video.mp4",
        ".env",
        "src/unneeded.py",
        "demo_assets/examples/unlisted.mp4",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not for upload", encoding="utf-8")
    return tmp_path


def test_export_is_allowlisted_and_never_overwrites_an_earlier_export(
    exporter, source, monkeypatch
):
    monkeypatch.delattr(hashlib, "file_digest", raising=False)
    # The deployment must not depend on a second requirements file.
    (source / "requirements.txt").unlink()
    destination = exporter.prepare_space(source)
    names = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    }
    expected = {
        "app.py",
        "README.md",
        "requirements.txt",
        ".gitattributes",
        "demo.py",
        "demo_assets/winner/best_model.pth",
        "demo_assets/winner/config.json",
        *(f"demo_assets/examples/{name}" for name in exporter.EXAMPLE_FILES),
        *(f"src/{name}" for name in exporter.SOURCE_FILES),
    }
    assert names == expected
    assert (destination / "demo_assets/winner/best_model.pth").read_bytes() == (
        source / "demo_assets/winner/best_model.pth"
    ).read_bytes()
    with ZipFile(destination.with_suffix(".zip")) as archive:
        assert {
            name for name in archive.namelist() if not name.endswith("/")
        } == expected
        assert (
            archive.read("demo_assets/winner/best_model.pth")
            == (source / "demo_assets/winner/best_model.pth").read_bytes()
        )
        assert archive.testzip() is None
        examples = [name for name in archive.namelist() if name.endswith(".mp4")]
        assert len(examples) == 3
        for name in examples:
            assert archive.read(name) == (source / name).read_bytes()
    assert exporter.prepare_space(source) != destination
    assert destination.is_dir()


def test_missing_example_stops_before_creating_upload_folder(exporter, source):
    (source / "demo_assets/examples/fight.mp4").unlink()
    with pytest.raises(FileNotFoundError, match="fight.mp4"):
        exporter.prepare_space(source)
    assert not (source / "runs/spaces_upload").exists()


@pytest.mark.parametrize("problem", ["pointer", "hash", "missing"])
def test_invalid_checkpoint_stops_before_creating_upload_folder(
    exporter, source, problem
):
    checkpoint = source / "demo_assets/winner/best_model.pth"
    if problem == "pointer":
        checkpoint.write_text(
            "version https://git-lfs.github.com/spec/v1\n", encoding="utf-8"
        )
    elif problem == "hash":
        checkpoint.write_bytes(b"corrupted")
    else:
        checkpoint.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        exporter.prepare_space(source)
    assert not (source / "runs/spaces_upload").exists()


def test_source_import_closure_and_space_pins(exporter):
    modules = {Path(name).stem for name in exporter.SOURCE_FILES}
    for name in exporter.SOURCE_FILES:
        tree = ast.parse(
            (ROOT / "src" / name).read_text(encoding="utf-8"), feature_version=(3, 10)
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                assert node.module.split(".")[0] in modules, (name, node.module)
    template = ROOT / "deployment/huggingface_space"
    card = (template / "README.md").read_text(encoding="utf-8")
    for line in (
        'python_version: "3.10"',
        "sdk: gradio",
        "sdk_version: 6.30.0",
        "app_file: app.py",
    ):
        assert line in card
    requirements = (template / "requirements.txt").read_text(encoding="utf-8")
    for line in (
        "torch==2.9.1",
        "torchvision==0.24.1",
        "av==16.1.0",
        "kagglehub==1.0.2",
        "matplotlib==3.10.8",
        "numpy==2.2.6",
        "pandas==2.3.3",
        "scikit-learn==1.7.0",
        "torchmetrics==1.8.2",
        "tqdm==4.67.3",
        "gradio==6.30.0",
        "spaces==0.51.3",
    ):
        assert line in requirements.splitlines()
    assert not any(
        line.startswith(("-r", "--requirement")) for line in requirements.splitlines()
    )
    assert "+cpu" not in requirements and "/whl/cpu" not in requirements
    assert "suggested_hardware: cpu-basic" not in card
    assert len([line for line in requirements.splitlines() if "==" in line]) == 12
    for path in (
        ROOT / "demo.py",
        ROOT / "deployment/prepare_space.py",
        template / "app.py",
    ):
        ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 10))


def test_spaces_entry_registers_real_gpu_callback_and_only_launches_in_main(
    monkeypatch,
):
    events = []
    predictor = Mock(mode="trained")
    factory = Mock(side_effect=lambda **kwargs: (events.append("load"), predictor)[1])
    spaces = ModuleType("spaces")

    def gpu(**options):
        assert options == {"duration": 30}
        events.append("decorate")

        def decorate(fn):
            @functools.wraps(fn)
            def allocated(*args, **kwargs):
                events.append("allocate_gpu")
                return fn(*args, **kwargs)

            return allocated

        return decorate

    spaces.GPU = gpu
    app = Mock()
    ui = ModuleType("demo")
    ui.build_app = Mock(return_value=app)
    examples = ["normal.mp4", "fight.mp4", "pollution.mp4"]
    ui.bundled_examples = Mock(return_value=examples)
    inference = ModuleType("src.demo_inference")
    inference.BUNDLED_CHECKPOINT = "bundle.pth"
    inference.BUNDLED_METADATA = "config.json"
    inference.DemoPredictor = factory
    monkeypatch.setitem(sys.modules, "spaces", spaces)
    monkeypatch.setitem(sys.modules, "demo", ui)
    monkeypatch.setitem(sys.modules, "src.demo_inference", inference)
    module = load_module(
        "space_entry_under_test", ROOT / "deployment/huggingface_space/app.py"
    )
    factory.assert_called_once_with(
        checkpoint="bundle.pth", metadata="config.json", device="cuda"
    )
    assert events == ["load", "decorate"]
    assert module.demo is app
    ui.bundled_examples.assert_called_once_with()
    ui.build_app.assert_called_once_with(
        predictor, examples=examples, predict_fn=module.predict
    )
    app.queue.assert_not_called()
    predictor.predict.assert_not_called()
    assert module.predict("sample.mp4") is predictor.predict.return_value
    assert events[-1] == "allocate_gpu"
    predictor.predict.assert_called_once_with("sample.mp4")
    module.main()
    assert factory.call_count == 1
    options = app.queue.return_value.launch.call_args.kwargs
    assert options["server_name"] == "0.0.0.0"
    assert options["server_port"] == 7860
    assert options["share"] is False
    assert options["max_file_size"] == "200mb"


def test_spaces_patching_precedes_model_imports():
    tree = ast.parse(
        (ROOT / "deployment/huggingface_space/app.py").read_text(encoding="utf-8")
    )
    imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module)
    assert imports.index("spaces") < imports.index("demo")
    assert imports.index("spaces") < imports.index("src.demo_inference")
