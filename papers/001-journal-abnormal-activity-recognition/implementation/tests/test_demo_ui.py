"""Fast checks of local-only UI wiring without training or serving a public app."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

import demo


def component_text(component):
    return str(component["props"].get("value", ""))


def descendant_ids(layout, component_id):
    """Read the layout tree, not just text that could be hidden anywhere."""
    if layout.get("id") == component_id:

        def collect(node):
            return {node["id"]} | set().union(
                *(collect(child) for child in node.get("children", []))
            )

        return collect(layout)
    for child in layout.get("children", []):
        found = descendant_ids(child, component_id)
        if found:
            return found
    return set()


def test_bundled_model_is_default_and_random_mode_remains_explicit():
    defaults = demo.parse_args([])
    assert defaults.checkpoint == demo.ROOT / "demo_assets/winner/best_model.pth"
    assert defaults.metadata == demo.ROOT / "demo_assets/winner/config.json"
    assert not defaults.random_weights
    with pytest.raises(SystemExit):
        demo.parse_args(["--random-weights", "--checkpoint", "model.pth"])
    with pytest.raises(SystemExit):
        demo.parse_args(["--random-weights", "--metadata", "metadata.json"])
    random = demo.parse_args(["--random-weights"])
    assert random.random_weights
    assert random.checkpoint is None and random.metadata is None
    explicit = demo.parse_args(["--checkpoint", "another/model.pth"])
    assert explicit.checkpoint == Path("another/model.pth")
    assert explicit.metadata is None


def test_default_paths_are_independent_of_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    args = demo.parse_args([])
    assert args.checkpoint.is_absolute() and args.metadata.is_absolute()
    assert args.checkpoint == demo.BUNDLED_CHECKPOINT
    assert args.metadata == demo.BUNDLED_METADATA


def test_ui_random_mode_is_visible_and_endpoint_is_connected():
    predictor = Mock(
        mode="random",
        status="UNTRAINED RANDOM WEIGHTS",
        class_names=["Normal Videos", "Fight"],
    )
    app = demo.build_app(predictor)
    config = app.get_config_file()
    details = next(
        component
        for component in config["components"]
        if component["type"] == "accordion"
        and component["props"]["label"] == "ℹ️ Model details"
    )
    nested = descendant_ids(config["layout"], details["id"])
    warning = next(
        component
        for component in config["components"]
        if "⚠️ UNTRAINED — UI test only" in component_text(component)
    )
    assert warning["id"] not in nested
    endpoint = next(
        dep for dep in config["dependencies"] if dep["api_name"] == "predict"
    )
    assert len(endpoint["inputs"]) == 1
    assert len(endpoint["outputs"]) == 3
    types = {component["id"]: component["type"] for component in config["components"]}
    assert [types[output] for output in endpoint["outputs"]] == [
        "label",
        "json",
        "textbox",
    ]
    assert app.analytics_enabled is False


def test_ui_trained_mode_is_concise_with_collapsed_model_details():
    predictor = Mock(
        mode="trained",
        status="Compatible trained checkpoint loaded; not the original paper checkpoint.",
        class_names=["Normal Videos", "Fight"],
    )
    app = demo.build_app(predictor)
    config = app.get_config_file()
    components = config["components"]
    content = json.dumps(config, ensure_ascii=False)
    assert "🎬 Video Activity Recognition" in content
    assert "By Sanele Hlabisa" in content
    assert any(
        component["type"] == "button" and component_text(component) == "🔍 Predict"
        for component in components
    )
    video = next(component for component in components if component["type"] == "video")
    assert video["props"]["sources"] == ["upload"]
    probabilities = next(
        component for component in components if component["type"] == "label"
    )
    assert probabilities["props"]["num_top_classes"] == len(predictor.class_names)
    details = next(
        component
        for component in components
        if component["type"] == "accordion"
        and component["props"]["label"] == "ℹ️ Model details"
    )
    assert details["props"]["open"] is False
    nested = descendant_ids(config["layout"], details["id"])
    assert nested
    for text in (predictor.status, demo.WINDOW_NOTE, "[64, 64, 128]"):
        matches = [
            component for component in components if text in component_text(component)
        ]
        assert matches, text
        assert all(component["id"] in nested for component in matches), text
    footer = next(
        component
        for component in components
        if "First 0.44 seconds only · Research demo · Scores are not calibrated."
        in component_text(component)
    )
    assert footer["id"] not in nested
    assert "not the original paper checkpoint" in content
    assert "UNTRAINED" not in content
    assert "UI test only" not in content
    assert "Uploads stay on this computer" not in content
    assert app.analytics_enabled is False


@pytest.mark.parametrize("injected", [False, True])
def test_predict_endpoint_uses_optional_callback_and_preserves_outputs(injected):
    result = {
        "probabilities": {"Normal Videos": 0.75, "Fight": 0.25},
        "sampling": {"target_fps": 16},
        "status": "Compatible trained checkpoint; first window only.",
    }
    predictor = Mock(
        mode="trained",
        status="Compatible trained checkpoint",
        class_names=["Normal Videos", "Fight"],
        predict=Mock(return_value=result),
    )
    callback = Mock(return_value=result)
    app = (
        demo.build_app(predictor, predict_fn=callback)
        if injected
        else demo.build_app(predictor)
    )
    handler = next(fn.fn for fn in app.fns.values() if fn.api_name == "predict")
    assert handler("clip.mp4") == (
        result["probabilities"],
        result["sampling"],
        result["status"],
    )
    if injected:
        callback.assert_called_once_with("clip.mp4")
        predictor.predict.assert_not_called()
    else:
        predictor.predict.assert_called_once_with("clip.mp4")
        callback.assert_not_called()


@pytest.mark.parametrize("injected", [False, True])
def test_predict_callback_preserves_empty_and_invalid_video_errors(injected):
    import gradio as gr

    predictor = Mock(
        mode="trained",
        status="Compatible trained checkpoint",
        class_names=["Normal Videos", "Fight"],
        predict=Mock(side_effect=ValueError("Cannot decode this video")),
    )
    callback = Mock(side_effect=ValueError("Cannot decode this video"))
    app = (
        demo.build_app(predictor, predict_fn=callback)
        if injected
        else demo.build_app(predictor)
    )
    handler = next(fn.fn for fn in app.fns.values() if fn.api_name == "predict")
    for empty in (None, ""):
        with pytest.raises(gr.Error, match="Upload a video"):
            handler(empty)
    callback.assert_not_called()
    predictor.predict.assert_not_called()
    with pytest.raises(gr.Error, match="Cannot decode this video"):
        handler({"invalid": "video input"})
    selected = callback if injected else predictor.predict
    selected.assert_called_once_with({"invalid": "video input"})


def test_only_validation_previews_are_discovered(tmp_path):
    for partition in ("test", "validation"):
        root = tmp_path / f"runs/experiments/{partition}/models/custom/predictions"
        (root / "correct").mkdir(parents=True)
        (root / "predictions.json").write_text(
            json.dumps({"partition": partition}), encoding="utf-8"
        )
        (root / "correct" / "example.mp4").touch()
    found = demo.validation_previews(tmp_path)
    assert len(found) == 1
    assert "validation" in Path(found[0]).parts


def test_bundled_examples_are_sorted_files_only_and_limited_to_three(tmp_path):
    assert demo.bundled_examples(tmp_path) == []
    folder = tmp_path / "demo_assets/examples"
    folder.mkdir(parents=True)
    for name in ("d.mp4", "a.mp4", "c.mp4", "b.mp4", "ignored.txt"):
        (folder / name).touch()
    (folder / "0-directory.mp4").mkdir()
    assert demo.bundled_examples(tmp_path) == [
        str((folder / name).resolve()) for name in ("a.mp4", "b.mp4", "c.mp4")
    ]


def test_main_binds_localhost_and_never_shares(monkeypatch, tmp_path):
    predictor = Mock(mode="random", status="UNTRAINED")
    app = Mock()
    monkeypatch.setattr(demo, "DemoPredictor", Mock(return_value=predictor))
    monkeypatch.setattr(demo, "build_app", Mock(return_value=app))
    monkeypatch.setattr(demo, "validation_previews", lambda: [])
    monkeypatch.setattr(
        demo,
        "bundled_examples",
        Mock(side_effect=AssertionError("Random mode retains validation previews")),
    )
    demo.main(["--random-weights", "--port", "7862"])
    options = app.queue.return_value.launch.call_args.kwargs
    assert options["server_name"] == "127.0.0.1"
    assert options["share"] is False
    assert options["server_port"] == 7862
    assert options["max_file_size"] == "200mb"
    assert options["allowed_paths"] == []


def test_main_uses_bundled_trained_model_by_default(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    predictor = Mock(mode="trained", status="Compatible trained checkpoint loaded.")
    factory = Mock(return_value=predictor)
    app = Mock()
    monkeypatch.setattr(demo, "DemoPredictor", factory)
    builder = Mock(return_value=app)
    monkeypatch.setattr(demo, "build_app", builder)
    examples = [str(tmp_path / f"example-{index}.mp4") for index in range(3)]
    bundled = Mock(return_value=examples)
    monkeypatch.setattr(demo, "bundled_examples", bundled)
    previews = Mock(
        side_effect=AssertionError("No historical random previews for trained mode")
    )
    monkeypatch.setattr(demo, "validation_previews", previews)
    demo.main([])
    factory.assert_called_once_with(
        checkpoint=demo.BUNDLED_CHECKPOINT,
        metadata=demo.BUNDLED_METADATA,
        random_weights=False,
        device="auto",
    )
    options = app.queue.return_value.launch.call_args.kwargs
    assert options["server_name"] == "127.0.0.1"
    assert options["server_port"] == 7860
    assert options["share"] is False
    assert options["allowed_paths"] == examples
    builder.assert_called_once_with(predictor, examples)
    bundled.assert_called_once_with()
    previews.assert_not_called()


@pytest.mark.parametrize("random", [False, True])
def test_explicit_example_overrides_all_discovery(monkeypatch, tmp_path, random):
    example = tmp_path / "chosen.mp4"
    example.touch()
    predictor = Mock(mode="random" if random else "trained", status="status")
    app = Mock()
    builder = Mock(return_value=app)
    monkeypatch.setattr(demo, "DemoPredictor", Mock(return_value=predictor))
    monkeypatch.setattr(demo, "build_app", builder)
    bundled = Mock(side_effect=AssertionError("Explicit examples override the bundle"))
    previews = Mock(side_effect=AssertionError("Explicit examples override previews"))
    monkeypatch.setattr(demo, "bundled_examples", bundled)
    monkeypatch.setattr(demo, "validation_previews", previews)
    args = ["--example", str(example)]
    if random:
        args.append("--random-weights")
    demo.main(args)
    expected = [str(example.resolve())]
    builder.assert_called_once_with(predictor, expected)
    assert app.queue.return_value.launch.call_args.kwargs["allowed_paths"] == expected
    bundled.assert_not_called()
    previews.assert_not_called()
