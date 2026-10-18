"""AeroNet AI -- automated test suite.

Covers dataset discovery/validation, the three classification agents
(ResNet-18, MobileNetV3, and EfficientNet-B0), the decision coordinator,
evaluation-metric calculation, image-upload validation, API routes and app startup.

Real model inference is exercised by the "smoke" tests, which are skipped
automatically when the required artifacts are unavailable. Everything else is
mocked or uses synthetic data so the suite runs quickly.

This project is image-only: there are no video, webcam or object-detection
tests.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_prep import (  # noqa: E402
    average_hash,
    class_for_source,
    discover_images,
    find_duplicates,
    hamming_distance,
    is_annotation_preview,
    validate_image,
)
from services.decision_coordinator import (  # noqa: E402
    AGREE,
    DISAGREE,
    ERROR,
    SINGLE_MODEL,
    combine,
)
from services.evaluation_service import (  # noqa: E402
    CLASS_NAMES,
    POSITIVE_CLASS,
    _metrics_block,
    _unavailable_metrics,
    stratified_split,
)


def is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _png_bytes(size, color) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def make_png(path: Path, size=(200, 150), color=(120, 160, 200)) -> Path:
    Image.new("RGB", size, color).save(path, format="PNG")
    return path


@pytest.fixture(scope="session")
def tmp_dataset(tmp_path_factory):
    """Build a small synthetic dataset tree with source images + previews."""
    root = tmp_path_factory.mktemp("ds") / "raw"
    (root / "bird").mkdir(parents=True)
    (root / "drone").mkdir(parents=True)

    make_png(root / "bird" / "b1_thumb.jpg", color=(30, 30, 60))
    make_png(root / "bird" / "b2_thumb.jpg", color=(35, 35, 70))
    (root / "bird" / "b1_annotation-1.png").write_bytes(_png_bytes((48, 48), (255, 0, 0)))
    make_png(root / "drone" / "d1_thumb.jpg", color=(200, 40, 40))
    make_png(root / "drone" / "d2_thumb.jpg", color=(210, 50, 50))
    (root / "drone" / "d1_annotation-1.png").write_bytes(_png_bytes((48, 48), (0, 255, 0)))
    # Corrupt / empty files must be flagged, never crash the pipeline.
    (root / "drone" / "broken_thumb.jpg").write_bytes(b"this-is-not-an-image")
    (root / "bird" / "empty_thumb.jpg").write_bytes(b"")
    # An unsupported file that must be ignored entirely.
    (root / "bird" / "notes.txt").write_text("not an image")
    return root


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    import app as app_module

    # Keep the suite from polluting the real detection history artifact:
    # back it up, and restore it (or delete it) when the session is done.
    history_backup = None
    if HISTORY_PATH.exists():
        history_backup = HISTORY_PATH.read_bytes()

    with TestClient(app_module.app) as c:
        yield c, app_module

    if history_backup is None:
        if HISTORY_PATH.exists():
            HISTORY_PATH.unlink()
    else:
        HISTORY_PATH.write_bytes(history_backup)

# ---------------------------------------------------------------------------
# Phase B -- dataset discovery & validation
# ---------------------------------------------------------------------------

class TestDatasetDiscovery:
    def test_discovers_sources_and_previews(self, tmp_dataset):
        sources, previews = discover_images(tmp_dataset)
        names = {r["path"].name for r in sources}
        assert "notes.txt" not in names, "unsupported extensions must be skipped"
        assert any("_thumb.jpg" in n for n in names), "source thumbs must be discovered"
        assert len(previews) == 2, f"expected 2 annotation previews, got {len(previews)}"
        assert all(r["kind"] == "source_image" for r in sources)
        assert all(r["kind"] == "annotation_preview" for r in previews)

    def test_invalid_files_are_flagged(self, tmp_dataset):
        sources, _ = discover_images(tmp_dataset)
        records = {r["path"].name: validate_image(dict(r)) for r in sources}
        assert records["broken_thumb.jpg"]["valid"] is False
        assert records["empty_thumb.jpg"]["valid"] is False
        assert records["empty_thumb.jpg"]["empty"] is True
        assert records["b1_thumb.jpg"]["valid"] is True

    def test_class_from_source_directory(self, tmp_dataset):
        sources, _ = discover_images(tmp_dataset)
        by_name = {r["path"].name: r for r in sources}
        assert class_for_source(by_name["b1_thumb.jpg"], tmp_dataset) == "bird"
        assert class_for_source(by_name["d1_thumb.jpg"], tmp_dataset) == "drone"

    def test_average_hash_and_hamming(self):
        a = np.zeros((64, 64, 3), dtype=np.uint8)
        b = np.full((64, 64, 3), 255, dtype=np.uint8)
        ha = average_hash(a)
        hb = average_hash(b)
        assert hamming_distance(ha, ha) == 0
        assert hamming_distance(ha, hb) > 0

    def test_find_duplicates_groups_identical_images(self, tmp_dataset):
        sources, _ = discover_images(tmp_dataset)
        # Two identical-colored images are near-duplicates under a low threshold.
        make_png(tmp_dataset / "bird" / "b3_thumb.jpg", color=(30, 30, 60))
        # Re-discover so the newly created file is included in the comparison.
        sources, _ = discover_images(tmp_dataset)
        groups = find_duplicates([validate_image(dict(r)) for r in sources], threshold=7)
        assert any(len(g) >= 2 for g in groups), "expected at least one duplicate group"


# ---------------------------------------------------------------------------
# Model agents
# ---------------------------------------------------------------------------

RESNET18_ARTIFACT = ROOT / "artifacts" / "resnet18_bird_drone.pt"
MOBILENET_ARTIFACT = ROOT / "artifacts" / "mobilenet_v3_bird_drone.pt"
EFFICIENTNET_ARTIFACT = ROOT / "artifacts" / "efficientnet_b0_bird_drone.pt"


def test_training_can_select_efficientnet_only(monkeypatch):
    from scripts.train_models import parse_args

    monkeypatch.setattr(sys, "argv", ["train_models.py", "--efficientnet-only"])
    assert parse_args().efficientnet_only is True


def test_training_split_keeps_near_duplicate_images_together(tmp_path):
    paths = []
    labels = []
    rng = np.random.default_rng(7)
    duplicate_path = None
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for index in range(5):
            pixels = rng.integers(0, 256, size=(40, 40, 3), dtype=np.uint8)
            path = class_dir / f"{index}.png"
            Image.fromarray(pixels).save(path)
            paths.append(path)
            labels.append(label)
            if index == 0:
                duplicate_path = path
        copy_path = class_dir / "duplicate.png"
        copy_path.write_bytes(duplicate_path.read_bytes())
        paths.append(copy_path)
        labels.append(label)

    train, val, test, *_ = stratified_split(paths, labels, 0.7, 0.15, 0.15, 42)
    split_for_path = {
        path: split
        for split, split_paths in (("train", train), ("val", val), ("test", test))
        for path in split_paths
    }
    for label, class_name in enumerate(CLASS_NAMES):
        original = tmp_path / class_name / "0.png"
        duplicate = tmp_path / class_name / "duplicate.png"
        assert split_for_path[original] == split_for_path[duplicate]
    assert set(split_for_path) == set(paths)


def test_training_uses_balanced_class_weights():
    from scripts.train_models import class_weights_for_labels

    weights = class_weights_for_labels(np.array([0, 0, 0, 1, 1, 1, 1, 1]), 2)
    assert weights[0] / weights[1] == pytest.approx(5 / 3)


def test_head_training_does_not_update_frozen_batch_norm():
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from scripts.train_models import _train_epoch

    model = nn.Sequential(
        nn.BatchNorm2d(3),
        nn.Flatten(),
        nn.Linear(3 * 8 * 8, 2),
    )
    for parameter in model[0].parameters():
        parameter.requires_grad = False
    batch_norm_mean = model[0].running_mean.clone()
    batch_norm_var = model[0].running_var.clone()
    optimizer = torch.optim.SGD(model[2].parameters(), lr=1e-3)
    dataset = TensorDataset(torch.rand(4, 3, 8, 8), torch.tensor([0, 1, 1, 1]))

    _train_epoch(model, DataLoader(dataset, batch_size=4), nn.CrossEntropyLoss(), optimizer, torch.device("cpu"))

    torch.testing.assert_close(model[0].running_mean, batch_norm_mean)
    torch.testing.assert_close(model[0].running_var, batch_norm_var)


@pytest.mark.parametrize(
    ("agent_path", "agent_class", "arch"),
    [
        ("models.resnet18_agent", "ResNet18Agent", "resnet18"),
        ("models.mobilenet_agent", "MobileNetV3Agent", "mobilenet_v3_small"),
        ("models.efficientnet_agent", "EfficientNetB0Agent", "efficientnet_b0"),
    ],
)
def test_agents_reject_reversed_checkpoint_class_mapping(
    tmp_path, agent_path, agent_class, arch
):
    import importlib

    module = importlib.import_module(agent_path)
    artifact = tmp_path / f"{arch}.pt"
    torch_payload = {
        "arch": arch,
        "class_names": ["drone", "bird"],
        "image_size": 224,
    }
    import torch

    torch.save(torch_payload, artifact)
    agent = getattr(module, agent_class)(artifact)
    agent.warm_up()
    assert agent.available is False
    assert "classes" in agent.load_error.lower()


def _artifact_missing(*paths: Path) -> bool:
    return any(not p.exists() for p in paths)


class TestModelAgents:
    def test_resnet18_agent_loads_artifact(self):
        if _artifact_missing(RESNET18_ARTIFACT):
            pytest.skip("resnet18 artifact not trained yet")
        from models.resnet18_agent import ResNet18Agent  # noqa: E402

        agent = ResNet18Agent()
        agent.warm_up()
        assert agent.available is True
        assert agent.model_name == "ResNet-18"
        assert agent.class_names == CLASS_NAMES

    def test_mobilenet_agent_loads_artifact(self):
        if _artifact_missing(MOBILENET_ARTIFACT):
            pytest.skip("mobilenet artifact not trained yet")
        from models.mobilenet_agent import MobileNetV3Agent  # noqa: E402

        agent = MobileNetV3Agent()
        agent.warm_up()
        assert agent.available is True
        assert agent.model_name == "MobileNetV3"
        assert agent.class_names == CLASS_NAMES

    def test_efficientnet_missing_artifact_is_unavailable(self):
        from models.efficientnet_agent import EfficientNetB0Agent

        agent = EfficientNetB0Agent(artifact_path=ROOT / "artifacts" / "missing_efficientnet.pt")
        agent.warm_up()
        result = agent.predict(Image.new("RGB", (64, 64)))
        assert agent.available is False
        assert result["status"] == "unavailable"
        assert "artifact not found" in result["error"].lower()

    def test_efficientnet_incompatible_artifact_is_unavailable(self, tmp_path):
        import torch
        from models.efficientnet_agent import EfficientNetB0Agent

        artifact = tmp_path / "wrong-architecture.pt"
        torch.save({"arch": "other", "class_names": ["bird", "drone"]}, artifact)
        agent = EfficientNetB0Agent(artifact)
        agent.warm_up()
        assert agent.available is False
        assert "architecture" in agent.load_error.lower()

    def test_missing_artifact_reports_unavailable(self):
        from models.resnet18_agent import ResNet18Agent  # noqa: E402

        agent = ResNet18Agent(artifact_path=ROOT / "artifacts" / "does_not_exist.pt")
        agent.warm_up()
        assert agent.available is False
        assert agent.model is None


@pytest.mark.smoke
class TestAgentInference:
    @pytest.fixture
    def rgb_image(self):
        return Image.new("RGB", (320, 240), (90, 140, 200))

    def test_resnet18_inference_shape(self, rgb_image):
        if _artifact_missing(RESNET18_ARTIFACT):
            pytest.skip("resnet18 artifact not trained yet")
        from models.resnet18_agent import ResNet18Agent  # noqa: E402

        agent = ResNet18Agent()
        agent.warm_up()
        assert agent.available, "resnet18 artifact must load"
        out = agent.predict(rgb_image)
        assert out["status"] == "ok", out.get("error")
        assert out["predicted_class"] in CLASS_NAMES
        assert set((out.get("probabilities") or {}).keys()) == set(CLASS_NAMES)
        probs = out["probabilities"]
        assert abs(sum(probs.values()) - 1.0) < 1e-4
        assert is_num(out["inference_time_ms"]) and out["inference_time_ms"] >= 0
        assert out["error"] is None

    def test_mobilenet_inference_shape(self, rgb_image):
        if _artifact_missing(MOBILENET_ARTIFACT):
            pytest.skip("mobilenet artifact not trained yet")
        from models.mobilenet_agent import MobileNetV3Agent  # noqa: E402

        agent = MobileNetV3Agent()
        agent.warm_up()
        assert agent.available, "mobilenet artifact must load"
        out = agent.predict(rgb_image)
        assert out["status"] == "ok", out.get("error")
        assert out["predicted_class"] in CLASS_NAMES
        assert set((out.get("probabilities") or {}).keys()) == set(CLASS_NAMES)
        probs = out["probabilities"]
        assert abs(sum(probs.values()) - 1.0) < 1e-4
        assert is_num(out["inference_time_ms"]) and out["inference_time_ms"] >= 0
        assert out["error"] is None

    def test_efficientnet_preprocessing_matches_imagenet_input_size(self):
        import torch
        from models.efficientnet_agent import default_transform
        from torchvision.models import EfficientNet_B0_Weights

        image = Image.new("RGB", (320, 180), (90, 140, 200))
        tensor = default_transform()(image)
        expected = EfficientNet_B0_Weights.DEFAULT.transforms()(image)
        assert tuple(tensor.shape) == (3, 224, 224)
        assert tensor.dtype == torch.float32
        torch.testing.assert_close(tensor, expected)

    def test_training_and_inference_preprocessing_match(self):
        import torch
        from scripts.train_models import TrainDataset
        from models.resnet18_agent import IMAGE_SIZE as RESNET_SIZE, default_transform as resnet_transform
        from models.mobilenet_agent import IMAGE_SIZE as MOBILENET_SIZE, default_transform as mobilenet_transform

        image = Image.new("RGB", (320, 180), (90, 140, 200))
        for image_size, inference_transform in (
            (RESNET_SIZE, resnet_transform),
            (MOBILENET_SIZE, mobilenet_transform),
        ):
            train_dataset = TrainDataset([], [], image_size)
            torch.testing.assert_close(
                train_dataset.transform(image),
                inference_transform()(image),
            )

    def test_efficientnet_builder_uses_two_class_head_without_pretrained_weights(self):
        from models.efficientnet_agent import build_efficientnet_b0

        model = build_efficientnet_b0(pretrained=False)
        assert model.classifier[-1].out_features == 2

    def test_efficientnet_controlled_model_prediction(self, rgb_image, tmp_path, monkeypatch):
        import torch
        from torch import nn
        import models.efficientnet_agent as module

        class FixedLogits(nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = nn.Parameter(torch.tensor([0.0, 2.0]))

            def forward(self, inputs):
                return self.logits.unsqueeze(0).expand(inputs.shape[0], -1)

        monkeypatch.setattr(module, "build_efficientnet_b0", lambda **kwargs: FixedLogits())
        artifact = tmp_path / "efficientnet.pt"
        torch.save({
            "arch": "efficientnet_b0",
            "class_names": ["bird", "drone"],
            "image_size": 224,
            "state_dict": FixedLogits().state_dict(),
        }, artifact)
        agent = module.EfficientNetB0Agent(artifact)
        agent.warm_up()
        out = agent.predict(rgb_image)
        assert out["status"] == "ok", out["error"]
        assert out["predicted_class"] == "drone"
        assert set(out["class_scores"]) == {"bird", "drone"}
        assert sum(out["class_scores"].values()) == pytest.approx(1.0)

    def test_efficientnet_inference_error_is_reported(self, rgb_image, tmp_path, monkeypatch):
        import torch
        from torch import nn
        import models.efficientnet_agent as module

        class BrokenModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(1))

            def forward(self, inputs):
                raise RuntimeError("inference test failure")

        monkeypatch.setattr(module, "build_efficientnet_b0", lambda **kwargs: BrokenModel())
        artifact = tmp_path / "efficientnet.pt"
        torch.save({
            "arch": "efficientnet_b0",
            "class_names": ["bird", "drone"],
            "state_dict": BrokenModel().state_dict(),
        }, artifact)
        agent = module.EfficientNetB0Agent(artifact)
        agent.warm_up()
        out = agent.predict(rgb_image)
        assert out["status"] == "error"
        assert "inference test failure" in out["error"]


# ---------------------------------------------------------------------------
# Decision coordinator
# ---------------------------------------------------------------------------

def _agent_result(status, predicted_class, scores=None, error=None, t_ms=10.0):
    return {
        "model_name": "test-agent",
        "status": status,
        "predicted_class": predicted_class,
        "class_scores": scores,
        "probabilities": scores,
        "inference_time_ms": t_ms,
        "error": error,
    }


class TestDecisionCoordinator:
    def test_both_agree_bird(self):
        out = combine(
            _agent_result("ok", "bird", {"bird": 0.9, "drone": 0.1}),
            _agent_result("ok", "bird", {"bird": 0.8, "drone": 0.2}),
        )
        assert out["final_prediction"] == "bird"
        assert out["agreement_status"] == AGREE
        assert out["error"] is None
        assert out["confidence"] is not None
        assert 0.0 <= out["confidence"] <= 1.0

    def test_all_three_agree(self):
        result = _agent_result("ok", "bird", {"bird": 0.8, "drone": 0.2})
        out = combine(result, result, result)
        assert out["final_prediction"] == "bird"
        assert out["agreement_status"] == AGREE
        assert set(out["agent_results"]) == {"resnet18", "mobilenetv3", "efficientnetb0"}

    def test_two_agree_one_disagrees_returns_uncertain(self):
        bird = _agent_result("ok", "bird", {"bird": 0.99, "drone": 0.01})
        drone = _agent_result("ok", "drone", {"bird": 0.01, "drone": 0.99})
        out = combine(bird, bird, drone)
        assert out["final_prediction"] == "uncertain"
        assert out["agreement_status"] == DISAGREE
        assert out["confidence"] is None
        assert out["agent_results"]["efficientnetb0"]["predicted_class"] == "drone"
        assert out["agent_results"]["efficientnetb0"]["class_scores"] == drone["class_scores"]

    def test_one_usable_agent_with_two_unavailable(self):
        bird = _agent_result("ok", "bird", {"bird": 0.8, "drone": 0.2})
        unavailable = _agent_result("unavailable", None, error="not trained")
        out = combine(bird, unavailable, unavailable)
        assert out["final_prediction"] == "bird"
        assert out["agreement_status"] == SINGLE_MODEL
        assert out["agent_results"]["mobilenetv3"]["status"] == "unavailable"
        assert out["agent_results"]["efficientnetb0"]["status"] == "unavailable"

    def test_two_available_agree_when_third_agent_errors(self):
        bird = _agent_result("ok", "bird", {"bird": 0.8, "drone": 0.2})
        error = _agent_result("error", None, error="inference failed")
        out = combine(bird, bird, error)
        assert out["final_prediction"] == "bird"
        assert out["agreement_status"] == AGREE
        assert out["agent_results"]["efficientnetb0"]["error"] == "inference failed"

    def test_both_agree_drone(self):
        out = combine(
            _agent_result("ok", "drone", {"bird": 0.2, "drone": 0.8}),
            _agent_result("ok", "drone", {"bird": 0.3, "drone": 0.7}),
        )
        assert out["final_prediction"] == "drone"
        assert out["agreement_status"] == AGREE

    def test_disagreement_returns_uncertain_and_preserves_agent_outputs(self):
        resnet = _agent_result("ok", "drone", {"bird": 0.2, "drone": 0.8})
        mobilenet = _agent_result("ok", "bird", {"bird": 0.65, "drone": 0.35})
        out = combine(
            resnet,
            mobilenet,
        )
        assert out["agreement_status"] == DISAGREE
        assert out["final_prediction"] == "uncertain"
        assert out["decision_reason"] == "models_disagree"
        assert out["confidence"] is None
        assert out["agent_results"]["resnet18"]["predicted_class"] == "drone"
        assert out["agent_results"]["resnet18"]["class_scores"] == resnet["class_scores"]
        assert out["agent_results"]["mobilenetv3"]["predicted_class"] == "bird"
        assert out["agent_results"]["mobilenetv3"]["class_scores"] == mobilenet["class_scores"]
        assert out["agent_results"]["efficientnetb0"]["status"] == "unavailable"

    def test_disagreement_low_confidence_becomes_uncertain(self):
        out = combine(
            _agent_result("ok", "drone", {"bird": 0.51, "drone": 0.49}),
            _agent_result("ok", "bird", {"bird": 0.49, "drone": 0.51}),
        )
        assert out["agreement_status"] == DISAGREE
        assert out["final_prediction"] == "uncertain"
        assert out["decision_reason"] == "models_disagree"
        assert out["confidence"] is None

    def test_single_model_available(self):
        out = combine(
            _agent_result("ok", "bird", {"bird": 0.9, "drone": 0.1}),
            _agent_result("error", None, None, "boom"),
        )
        assert out["final_prediction"] == "bird"
        assert out["agreement_status"] == SINGLE_MODEL
        assert "single_model_available" in out["decision_reason"]
        assert out["agent_results"]["mobilenetv3"]["status"] == "error"

    def test_single_model_other_side(self):
        out = combine(
            _agent_result("error", None, None, "boom"),
            _agent_result("ok", "drone", {"bird": 0.1, "drone": 0.9}),
        )
        assert out["final_prediction"] == "drone"
        assert out["agreement_status"] == SINGLE_MODEL
        assert out["agent_results"]["resnet18"]["status"] == "error"

    def test_both_models_fail(self):
        out = combine(
            _agent_result("error", None, None, "boom one"),
            _agent_result("error", None, None, "boom two"),
        )
        assert out["final_prediction"] is None
        assert out["agreement_status"] == ERROR
        assert out["error"] is not None
        assert out["confidence"] is None

    def test_missing_predicted_class_treated_as_failure(self):
        out = combine(
            _agent_result("ok", None, {"bird": 0.5, "drone": 0.5}),
            _agent_result("ok", "drone", {"bird": 0.1, "drone": 0.9}),
        )
        assert out["agreement_status"] == SINGLE_MODEL
        assert out["final_prediction"] == "drone"

    def test_processing_time_recorded(self):
        out = combine(
            _agent_result("ok", "bird", {"bird": 0.9, "drone": 0.1}),
            _agent_result("ok", "bird", {"bird": 0.8, "drone": 0.2}),
            processing_time_ms=42.5,
        )
        assert out["processing_time_ms"] == 42.5

    def test_structure_contains_required_keys(self):
        out = combine(
            _agent_result("ok", "bird", {"bird": 0.9, "drone": 0.1}),
            _agent_result("ok", "drone", {"bird": 0.4, "drone": 0.6}),
        )
        for key in ("final_prediction", "agreement_status", "decision_reason",
                    "agent_results", "confidence", "processing_time_ms"):
            assert key in out, f"missing key {key}"
        assert set(out["agent_results"]) == {"resnet18", "mobilenetv3", "efficientnetb0"}


# ---------------------------------------------------------------------------
# Evaluation service unit tests
# ---------------------------------------------------------------------------

BINARY_ACCURACY = lambda y, p: (np.array(y) == np.array(p)).mean()


class TestEvaluationMetrics:
    def test_metrics_block_handles_perfect_and_wrong_predictions(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 1]
        block = _metrics_block(y_true, y_pred, CLASS_NAMES, [1.0, 2.0, 3.0, 4.0])
        assert block["samples"] == 4
        assert block["evaluated"] == 4
        assert block["accuracy"] == 0.75
        # binary "drone"=1: FP=1 (idx1 predicted drone but true bird), TP=2
        assert block["precision"] == pytest.approx(2 / 3)
        assert block["recall"] == 1.0
        assert block["f1"] == pytest.approx(0.8)
        assert block["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
        assert block["confusion_matrix"] == [[1, 1], [0, 2]]
        assert block["per_class"]["bird"]["precision"] == 1.0
        assert block["per_class"]["bird"]["recall"] == 0.5
        assert block["per_class"]["bird"]["f1"] == pytest.approx(2 / 3)
        assert block["per_class"]["bird"]["support"] == 2
        assert block["avg_inference_time_ms"] == 2.5

    def test_metrics_block_all_none_is_graceful(self):
        block = _metrics_block([0, 1, 1], [None, None, None], CLASS_NAMES, [1.0, 2.0, 3.0])
        assert block["samples"] == 3
        assert block["evaluated"] == 0
        assert block["accuracy"] is None
        assert block["confusion_matrix"] is None
        assert block["per_class"]["bird"]["support"] == 1
        assert block["per_class"]["drone"]["support"] == 2

    def test_positive_class_binary_f1(self):
        y_true = [1, 1, 0, 1]
        y_pred = [1, 1, 0, 0]
        block = _metrics_block(y_true, y_pred, CLASS_NAMES, [1.0, 1.0, 1.0, 1.0])
        # TP=2, FP=0, FN=1 -> precision 1.0, recall 2/3, f1 = 0.8
        assert block["precision"] == 1.0
        assert block["recall"] == pytest.approx(2 / 3)
        assert block["f1"] == pytest.approx(0.8)

    def test_macro_is_mean_over_classes(self):
        y_true = [0, 0, 1, 1]
        y_pred = [1, 1, 0, 0]
        block = _metrics_block(y_true, y_pred, CLASS_NAMES, [1.0] * 4)
        # Every prediction is wrong: bird tp=0/fp=2/fn=2 and drone tp=0/fp=2/fn=2.
        assert block["macro_precision"] == 0.0
        assert block["macro_recall"] == 0.0
        assert block["macro_f1"] == 0.0

    def test_zero_division_returns_zero(self):
        # A model that predicts nothing for the positive class.
        y_true = [1, 1, 1, 0]
        y_pred = [0, 0, 0, 0]
        block = _metrics_block(y_true, y_pred, CLASS_NAMES, [])
        assert block["recall"] == 0.0
        assert block["f1"] == 0.0

    def test_uncertain_predictions_are_not_counted_as_class_predictions(self):
        block = _metrics_block(
            ["bird", "drone", "drone"],
            ["uncertain", "drone", None],
            CLASS_NAMES,
            [2.0, 1.0, 0.0],
        )
        assert block["samples"] == 3
        assert block["evaluated"] == 1
        assert block["accuracy"] == 1.0

    def test_unavailable_model_block(self):
        block = _unavailable_metrics(6)
        assert block["evaluated"] == 0
        assert block["samples"] == 6
        assert block["accuracy"] is None
        assert block["confusion_matrix"] is None
        assert block["note"] == "model unavailable: metrics were not measured"


# ---------------------------------------------------------------------------
# API surface tests (FastAPI TestClient, real startup + real inference)
# ---------------------------------------------------------------------------

ALLOWED_DECISIONS = {"bird", "drone", "uncertain", "error"}


class TestFastApi:
    def test_models_endpoint(self, client):
        c, _ = client
        resp = c.get("/api/models")
        assert resp.status_code == 200
        state = resp.json()
        assert set(state.keys()) == {"resnet18", "mobilenetv3",
                                     "efficientnetb0", "logistic_regression_baseline"}
        for agent in state.values():
            assert set(agent.keys()) >= {"name", "available", "status"}
            if agent.get("available"):
                assert agent["status"] == "ok"

    def test_manifest_endpoint(self, client):
        c, _ = client
        resp = c.get("/api/manifest")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)
        assert "classes" in data and "counts" in data

    def test_sample_image_static_paths(self, client):
        c, _ = client
        for name in ("bird.png", "drone.png", "building.png"):
            response = c.get(f"/samples/{name}")
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("image/")

    def test_app_startup_warms_models(self, client):
        c, app_module = client
        assert {"resnet18", "mobilenetv3"} <= set(app_module.MODEL_STATE.keys())
        assert "efficientnetb0" in app_module.MODEL_STATE
        efficientnet_available = EFFICIENTNET_ARTIFACT.exists()
        assert app_module.MODEL_STATE["efficientnetb0"]["available"] is efficientnet_available
        if not efficientnet_available:
            assert "artifact" in app_module.MODEL_STATE["efficientnetb0"]["note"].lower()
        assert all(app_module.MODEL_STATE[a]["available"] for a in
                   ("resnet18", "mobilenetv3"))

    def test_detect_requires_valid_image(self, client):
        c, _ = client
        for fname, content in (("file.txt", b"not an image"),
                               ("empty.jpg", b""),
                               ("broken.jpg", b"\x89notanimage")):
            resp = c.post("/api/detect", files={"image": (fname, content)})
            assert resp.status_code == 400, (fname, resp.status_code, resp.text)
            assert "error" in resp.json()

    def test_detect_returns_structured_decision(self, client, tmp_dataset):
        # Reuse a clean valid source image as the probe payload.
        probe = tmp_dataset / "drone" / "d1_thumb.jpg"
        c, _ = client
        resp = c.post("/api/detect", files={"image": (probe.name, probe.read_bytes())})
        assert resp.status_code == 200
        body = resp.json()
        assert body["decision"] in ALLOWED_DECISIONS
        assert set(body.keys()) >= {"decision", "confidence", "reason",
                                    "processing_time_ms", "models", "result",
                                    "uploaded_image"}
        result = body["result"]
        assert set(result.keys()) >= {"final_prediction", "agreement_status",
                                      "decision_reason", "agent_results",
                                      "confidence", "processing_time_ms"}
        assert set(result["agent_results"].keys()) == {
            "resnet18", "mobilenetv3", "efficientnetb0"
        }
        assert set(body["models"].keys()) == {
            "resnet18", "mobilenetv3", "efficientnetb0"
        }
        assert body["models"]["efficientnetb0"]["status"] == (
            "ok" if EFFICIENTNET_ARTIFACT.exists() else "unavailable"
        )
        for agent in result["agent_results"].values():
            # Agent summary uses "model" (not the legacy "model_name") and a
            # "status"; "error" is only present and populated when the agent
            # failed (otherwise it is None and not included in the front-end
            # error handling paths).
            assert set(agent.keys()) >= {"model", "status"}
            if agent.get("error"):
                assert isinstance(agent["error"], str) and agent["error"]
        for model_name, agent in body["models"].items():
            assert set(agent.keys()) >= {"status", "predicted_class",
                                         "inference_time_ms", "error"}

    def test_detect_sample_images(self, client):
        c, _ = client
        for sample in ("samples/drone.png", "samples/bird.png"):
            resp = c.post("/api/detect",
                          files={"image": (Path(sample).name, Path(sample).read_bytes())})
            assert resp.status_code == 200, (sample, resp.text)
            assert resp.json()["decision"] in ALLOWED_DECISIONS

    def test_predict_history_records_requests(self, client, tmp_dataset):
        c, _ = client
        probe = tmp_dataset / "bird" / "b1_thumb.jpg"
        c.post("/api/detect", files={"image": (probe.name, probe.read_bytes())})
        resp = c.get("/api/history.json")
        assert resp.status_code == 200
        history = resp.json()
        assert isinstance(history, list) and len(history) >= 1
        # File names are normalized with secure_filename(), so the stored name
        # is the hash-prefixed original name, not the raw basename.
        assert history[0]["file_name"].endswith(probe.name)
        assert history[0]["decision"] in ALLOWED_DECISIONS
        stats = c.get("/api/stats").json()
        assert stats["total"] == len(history)
        assert stats["recent_detections"][0]["file_name"].endswith(probe.name)

    def test_evaluation_endpoint(self, client):
        c, _ = client
        resp = c.get("/api/evaluation")
        # Artifacts exist in the real repo, but the suite must not depend on
        # that: report a controlled 404 before any training run.
        assert resp.status_code in (200, 404)
        if resp.status_code == 200:
            report = resp.json()
            assert "models" in report and "dataset" in report
            assert "split" in report
            assert "efficientnetb0" in report["models"]
            unavailable = report["models"]["efficientnetb0"]
            if unavailable["available"] is False:
                assert unavailable["accuracy"] is None
                assert unavailable["confusion_matrix"] is None

    def test_evaluate_endpoint_response_without_training_side_effect(self, client, monkeypatch):
        c, app_module = client
        monkeypatch.setattr(
            app_module.subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(stdout="mock training complete", stderr=""),
        )
        response = c.post("/api/evaluate")
        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is True
        assert "efficientnetb0" in body["report"]["models"]

    def test_no_video_webcam_or_download_endpoints(self, client):
        c, app_module = client
        defined = {r.path for r in app_module.app.routes}
        for forbidden in ("/video", "/webcam", "/download", "/stream", "/capture"):
            assert forbidden not in defined, f"legacy endpoint still registered: {forbidden}"
        expected = {"/", "/api/detect", "/api/models", "/api/evaluation",
                    "/api/evaluate", "/api/manifest", "/api/stats",
                    "/api/history.json"}
        assert expected <= defined, f"missing routes: {expected - defined}"


# ---------------------------------------------------------------------------
# Model artifacts loading tests (pytest markers: slow)
# ---------------------------------------------------------------------------


class TestModelArtifactLoading:
    def test_resnet18_artifact_exists(self):
        assert RESNET18_ARTIFACT.exists(), "run scripts/train_models.py first"
        assert RESNET18_ARTIFACT.stat().st_size > 0

    def test_mobilenet_artifact_exists(self):
        assert MOBILENET_ARTIFACT.exists(), "run scripts/train_models.py first"
        assert MOBILENET_ARTIFACT.stat().st_size > 0

    def test_resnet18_artifact_memmap_safe(self):
        # Artifacts must be loadable without holding the file open forever.
        import torch

        ckpt = torch.load(RESNET18_ARTIFACT, map_location="cpu", weights_only=True)
        assert "class_names" in ckpt and "state_dict" in ckpt
        assert ckpt["class_names"] == ["bird", "drone"]


# ---------------------------------------------------------------------------
# Image-only application firewall tests
# ---------------------------------------------------------------------------


ALLOWED_ROUTES = {"/", "/api/detect", "/api/models", "/api/evaluation",
                  "/api/evaluate", "/api/manifest", "/api/stats",
                  "/api/history.json"}


class TestImageOnlyFirewall:
    def test_no_streaming_video_or_webcam_endpoints(self):
        import app as app_module

        defined = {r.path for r in app_module.app.routes}
        legacy_patterns = ("/video", "/cam", "/webcam", "/stream", "/upload", "/download")
        for pattern in legacy_patterns:
            assert not any(p.startswith(pattern) for p in defined), \
                f"legacy endpoint still registered: {pattern}"

    def test_only_allowed_routes_registered(self):
        import app as app_module

        defined = {r.path for r in app_module.app.routes}
        # FastAPI/Starlette auto-register these docs routes; they are not part
        # of the image-only API surface.
        framework_routes = {"/openapi.json", "/docs", "/redoc", "/docs/oauth2-redirect",
                            "/samples"}
        assert ALLOWED_ROUTES <= defined, f"missing routes: {ALLOWED_ROUTES - defined}"
        for path in defined - ALLOWED_ROUTES - framework_routes:
            pytest.fail(f"unexpected route registered: {path}")

    def test_index_is_html_page(self, client):
        c, _ = client
        resp = c.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "AeroNet" in resp.text


# ---------------------------------------------------------------------------
# Entry point + environment
# ---------------------------------------------------------------------------


def test_requirements_omit_video_nonsmoke():
    req_path = ROOT / "requirements.txt"
    text = req_path.read_text(encoding="utf-8")
    assert "ultralytics" not in text
    assert "opencv-python" not in text
    assert "fastapi" in text
    assert "uvicorn" in text


def test_gitignore_does_not_ignore_repo_src():
    gitignore = ROOT / ".gitignore"
    text = gitignore.read_text(encoding="utf-8")
    for secret in ("dataset/", "dataset/raw", "artifacts/*.pt", "*.pyc"):
        assert secret not in text, f"\"{secret}\" must not be gated by .gitignore"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
