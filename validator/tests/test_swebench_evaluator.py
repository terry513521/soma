import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import docker
import pytest


TESTS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(TESTS_DIR, "../.."))
MCP_PLATFORM_DIR = os.path.abspath(os.path.join(TESTS_DIR, "../../mcp_platform"))
FIXTURES_DIR = Path(TESTS_DIR) / "fixtures"
os.environ.setdefault("VALIDATOR_DISABLE_APP_INIT", "1")

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if MCP_PLATFORM_DIR not in sys.path:
    sys.path.insert(0, MCP_PLATFORM_DIR)

from validator.evaluation.evaluator import Evaluator
from validator.evaluation.swebench_evaluator import SWEBenchContainerEvaluator


def test_empty_diff_returns_zero_without_loading_harness(monkeypatch):
    evaluator = SWEBenchContainerEvaluator(settings=SimpleNamespace())

    def _unexpected_load():
        raise AssertionError("Harness should not be loaded for empty diffs")

    monkeypatch.setattr(evaluator, "_load_harness_api", _unexpected_load)

    result = evaluator._evaluate_instance_diff_sync(
        instance_id="django__django-11119",
        diff="   ",
    )

    assert result.score == 0
    assert result.resolved is False
    assert result.image_name == "ghcr.io/epoch-research/swe-bench.eval.x86_64.django__django-11119"
    assert result.report["django__django-11119"]["error"] == "empty_diff"


def test_invalid_diff_format_returns_zero_without_loading_harness(monkeypatch):
    evaluator = SWEBenchContainerEvaluator(settings=SimpleNamespace())

    def _unexpected_load():
        raise AssertionError("Harness should not be loaded for invalid diffs")

    monkeypatch.setattr(evaluator, "_load_harness_api", _unexpected_load)

    result = evaluator._evaluate_instance_diff_sync(
        instance_id="django__django-11119",
        diff="this is not a patch",
    )

    assert result.score == 0
    assert result.resolved is False
    assert result.image_name == "ghcr.io/epoch-research/swe-bench.eval.x86_64.django__django-11119"
    assert result.report["django__django-11119"]["error"] == "invalid_diff_format"


def test_sync_runner_uses_explicit_image_and_returns_binary_success(monkeypatch):
    evaluator = SWEBenchContainerEvaluator(
        settings=SimpleNamespace(
            swebench_dataset_name="SWE-bench/SWE-bench_Verified",
            swebench_dataset_split="test",
            swebench_eval_timeout_seconds=123,
            swebench_eval_model_name="validator-test",
        )
    )

    fake_client = SimpleNamespace(close=lambda: None)
    fake_test_spec = SimpleNamespace(arch="x86_64")
    observed = {}

    def fake_make_test_spec(instance, namespace=None, instance_image_tag="latest"):
        observed["instance"] = dict(instance)
        observed["namespace"] = namespace
        observed["instance_image_tag"] = instance_image_tag
        return fake_test_spec

    def fake_run_instance(test_spec, prediction, rm_image, force_rebuild, client, run_id, timeout):
        observed["prediction"] = dict(prediction)
        observed["rm_image"] = rm_image
        observed["force_rebuild"] = force_rebuild
        observed["client"] = client
        observed["run_id"] = run_id
        observed["timeout"] = timeout
        assert test_spec is fake_test_spec
        return prediction["instance_id"], {
            prediction["instance_id"]: {"resolved": True}
        }

    fake_harness = SimpleNamespace(
        docker=SimpleNamespace(from_env=lambda timeout=600: fake_client),
        KEY_INSTANCE_ID="instance_id",
        KEY_MODEL="model_name_or_path",
        KEY_PREDICTION="model_patch",
        run_instance=fake_run_instance,
        make_test_spec=fake_make_test_spec,
        load_swebench_dataset=lambda name, split, instance_ids: [
            {"instance_id": instance_ids[0], "repo": "django/django"}
        ],
    )
    monkeypatch.setattr(evaluator, "_load_harness_api", lambda: fake_harness)

    result = evaluator._evaluate_instance_diff_sync(
        instance_id="django__django-11119",
        diff="diff --git a/foo.py b/foo.py\n",
        arch="arm64",
        image_name="ghcr.io/epoch-research/custom-image:latest",
    )

    assert result.score == 1
    assert result.resolved is True
    assert result.image_name == "ghcr.io/epoch-research/custom-image:latest"
    assert fake_test_spec.arch == "arm64"
    assert observed["instance"]["image_name"] == "ghcr.io/epoch-research/custom-image:latest"
    assert observed["namespace"] is None
    assert observed["instance_image_tag"] == "latest"
    assert observed["prediction"] == {
        "instance_id": "django__django-11119",
        "model_name_or_path": "validator-test",
        "model_patch": "diff --git a/foo.py b/foo.py\n",
    }
    assert observed["rm_image"] is True
    assert observed["force_rebuild"] is False
    assert observed["client"] is fake_client
    assert observed["timeout"] == 123
    assert observed["run_id"].startswith("validator-django--django-11119-")


def test_sync_runner_can_disable_image_cleanup(monkeypatch):
    evaluator = SWEBenchContainerEvaluator(
        settings=SimpleNamespace(
            swebench_eval_remove_image_after_run=False,
        )
    )

    fake_client = SimpleNamespace(close=lambda: None)
    fake_test_spec = SimpleNamespace(arch="x86_64")
    observed = {}

    def fake_run_instance(test_spec, prediction, rm_image, force_rebuild, client, run_id, timeout):
        observed["rm_image"] = rm_image
        observed["force_rebuild"] = force_rebuild
        return prediction["instance_id"], {
            prediction["instance_id"]: {"resolved": True}
        }

    fake_harness = SimpleNamespace(
        docker=SimpleNamespace(from_env=lambda timeout=600: fake_client),
        KEY_INSTANCE_ID="instance_id",
        KEY_MODEL="model_name_or_path",
        KEY_PREDICTION="model_patch",
        run_instance=fake_run_instance,
        make_test_spec=lambda instance, namespace=None, instance_image_tag="latest": fake_test_spec,
        load_swebench_dataset=lambda name, split, instance_ids: [
            {"instance_id": instance_ids[0], "repo": "django/django"}
        ],
    )
    monkeypatch.setattr(evaluator, "_load_harness_api", lambda: fake_harness)

    result = evaluator._evaluate_instance_diff_sync(
        instance_id="django__django-11119",
        diff="diff --git a/foo.py b/foo.py\n",
    )

    assert result.score == 1
    assert observed["rm_image"] is False
    assert observed["force_rebuild"] is False


def test_load_harness_api_requires_installed_package(monkeypatch):
    evaluator = SWEBenchContainerEvaluator(settings=SimpleNamespace())

    def _missing_swebench():
        raise ModuleNotFoundError("No module named 'swebench'", name="swebench")

    monkeypatch.setattr(evaluator, "_import_harness_modules", _missing_swebench)

    with pytest.raises(
        RuntimeError,
        match="SWE-Bench harness is not installed in the validator environment",
    ):
        evaluator._load_harness_api()


def test_extract_test_logs_prefers_pytest_section_inside_markers():
    raw_output = "\n".join(
        [
            "setup output",
            ">>>>> Start Test Output",
            "environment bootstrap",
            "============================= test session starts ==============================",
            "platform linux -- Python 3.12.3, pytest-8.3.4",
            "collected 1 item",
            "",
            "tests/test_example.py .                                                [100%]",
            "",
            "============================== 1 passed in 0.10s ==============================",
            ">>>>> End Test Output",
            "cleanup output",
        ]
    )

    extracted = SWEBenchContainerEvaluator._extract_test_logs(raw_output)

    assert extracted == "\n".join(
        [
            "============================= test session starts ==============================",
            "platform linux -- Python 3.12.3, pytest-8.3.4",
            "collected 1 item",
            "",
            "tests/test_example.py .                                                [100%]",
            "",
            "============================== 1 passed in 0.10s ==============================",
        ]
    )


def test_extract_test_logs_from_realistic_astropy_sample_fixture():
    raw_output = (FIXTURES_DIR / "swebench_astropy_test_output.txt").read_text(
        encoding="utf-8"
    )

    extracted = SWEBenchContainerEvaluator._extract_test_logs(raw_output)

    assert extracted.startswith(
        "============================= test session starts =============================="
    )
    assert "collected 15 items" in extracted
    assert "astropy/modeling/tests/test_separable.py ...............                 [100%]" in extracted
    assert "============================== 15 passed in 0.32s ==============================" in extracted
    assert ">>>>> Start Test Output" not in extracted
    assert ">>>>> End Test Output" not in extracted
    assert "git checkout d16bfe05a744909de4b27f5875fe0d4ed41ce607" not in extracted


def test_sync_runner_reads_extracted_test_logs_from_harness_output(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    evaluator = SWEBenchContainerEvaluator(
        settings=SimpleNamespace(
            swebench_eval_model_name="validator/test",
        )
    )

    fake_client = SimpleNamespace(close=lambda: None)
    fake_test_spec = SimpleNamespace(arch="x86_64")

    def fake_run_instance(test_spec, prediction, rm_image, force_rebuild, client, run_id, timeout):
        log_dir = (
            tmp_path
            / "logs"
            / "run_evaluation"
            / run_id
            / "validator__test"
            / prediction["instance_id"]
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "test_output.txt").write_text(
            "\n".join(
                [
                    "setup output",
                    ">>>>> Start Test Output",
                    "environment bootstrap",
                    "============================= test session starts ==============================",
                    "platform linux -- Python 3.12.3, pytest-8.3.4",
                    "collected 1 item",
                    "",
                    "tests/test_example.py .                                                [100%]",
                    "",
                    "============================== 1 passed in 0.10s ==============================",
                    ">>>>> End Test Output",
                    "cleanup output",
                ]
            ),
            encoding="utf-8",
        )
        return prediction["instance_id"], {
            prediction["instance_id"]: {"resolved": True}
        }

    fake_harness = SimpleNamespace(
        docker=SimpleNamespace(from_env=lambda timeout=600: fake_client),
        KEY_INSTANCE_ID="instance_id",
        KEY_MODEL="model_name_or_path",
        KEY_PREDICTION="model_patch",
        run_instance=fake_run_instance,
        make_test_spec=lambda instance, namespace=None, instance_image_tag="latest": fake_test_spec,
        load_swebench_dataset=lambda name, split, instance_ids: [
            {"instance_id": instance_ids[0], "repo": "django/django"}
        ],
    )
    monkeypatch.setattr(evaluator, "_load_harness_api", lambda: fake_harness)

    result = evaluator._evaluate_instance_diff_sync(
        instance_id="django__django-11119",
        diff="diff --git a/foo.py b/foo.py\n",
    )

    assert result.score == 1
    assert result.logs == "\n".join(
        [
            "============================= test session starts ==============================",
            "platform linux -- Python 3.12.3, pytest-8.3.4",
            "collected 1 item",
            "",
            "tests/test_example.py .                                                [100%]",
            "",
            "============================== 1 passed in 0.10s ==============================",
        ]
    )


def test_invalid_arch_is_rejected():
    evaluator = SWEBenchContainerEvaluator(settings=SimpleNamespace())

    with pytest.raises(ValueError, match="Unsupported SWE-Bench architecture"):
        evaluator._evaluate_instance_diff_sync(
            instance_id="django__django-11119",
            diff="diff --git a/foo.py b/foo.py\n",
            arch="sparc",
        )


@pytest.mark.asyncio
async def test_evaluator_delegates_swebench_patch_to_runner():
    evaluator = Evaluator.__new__(Evaluator)
    evaluator._swebench_evaluator = SimpleNamespace(
        evaluate_instance_diff=AsyncMock(return_value=SimpleNamespace(score=1, resolved=True))
    )

    result = await evaluator.evaluate_swebench_patch(
        instance_id="django__django-11119",
        diff="diff --git a/foo.py b/foo.py\n",
        arch="x86_64",
        image_name="ghcr.io/epoch-research/custom-image:latest",
    )

    assert result.score == 1


@pytest.mark.asyncio
async def test_evaluator_builds_question_scores_from_swebench_result():
    evaluator = Evaluator(settings=SimpleNamespace(max_concurrent_evaluations=2))

    async def fake_eval(**kwargs):
        assert kwargs == {
            "instance_id": "django__django-11119",
            "diff": "diff --git a/foo.py b/foo.py\n",
            "arch": "arm64",
        }
        return SimpleNamespace(
            resolved=True,
            score=1,
            image_name="ghcr.io/epoch-research/custom-image:latest",
        )

    evaluator.evaluate_swebench_patch = fake_eval

    task = SimpleNamespace(
        validation_id=123,
        instance_id="django__django-11119",
        diff="diff --git a/foo.py b/foo.py\n",
        arch="arm64",
        image_name="ghcr.io/epoch-research/custom-image:latest",
    )

    result = await evaluator.evaluate(task)

    assert result["batch_id"] == "123"
    assert result["resolved"] is True
    assert "custom-image" in result["logs"]
    assert len(result["question_scores"]) == 1
    question_score = result["question_scores"][0]
    assert question_score.batch_challenge_id == "123"
    assert question_score.question_id == "123"
    assert question_score.produced_answer == "1"
    assert question_score.score == 1.0
    assert question_score.details == {
        "image_name": "ghcr.io/epoch-research/custom-image:latest",
        "binary_resolved": 1,
    }


@pytest.mark.network
def test_real_swebench_container_evaluation_for_known_verified_instance():
    client = None
    try:
        client = docker.from_env(timeout=600)
        client.ping()
    except Exception as exc:
        if client is not None:
            client.close()
        pytest.skip(f"Docker is unavailable for SWE-Bench integration test: {exc}")
    finally:
        if client is not None:
            client.close()

    evaluator = SWEBenchContainerEvaluator(
        settings=SimpleNamespace(
            swebench_dataset_name="SWE-bench/SWE-bench_Verified",
            swebench_dataset_split="test",
            swebench_eval_arch="x86_64",
            swebench_eval_timeout_seconds=1800,
            swebench_eval_model_name="validator-network-test",
        )
    )

    harness = evaluator._load_harness_api()
    dataset = harness.load_swebench_dataset(
        name="SWE-bench/SWE-bench_Verified",
        split="test",
        instance_ids=["django__django-11119"],
    )
    row = dict(dataset[0])

    result = evaluator._evaluate_instance_diff_sync(
        instance_id="django__django-11119",
        diff=row["patch"],
        arch="x86_64",
    )

    assert result.resolved is True
    assert result.score == 1
    assert result.image_name == (
        "ghcr.io/epoch-research/swe-bench.eval.x86_64.django__django-11119"
    )