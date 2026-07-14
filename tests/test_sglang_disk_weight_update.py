import pytest

from slime.backends.sglang_utils.sglang_engine import SGLangEngine


def _recording_engine():
    engine = object.__new__(SGLangEngine)
    calls = []
    engine._make_request = lambda endpoint, payload: calls.append((endpoint, payload))
    return engine, calls


@pytest.mark.unit
def test_disk_weight_update_preserves_backward_compatible_payload_by_default():
    engine, calls = _recording_engine()

    engine.update_weights_from_disk("/checkpoints/weight_v000003", weight_version="3")

    assert calls == [
        (
            "update_weights_from_disk",
            {
                "model_path": "/checkpoints/weight_v000003",
                "weight_version": "3",
            },
        )
    ]


@pytest.mark.unit
def test_disk_weight_update_can_preserve_speculative_draft():
    engine, calls = _recording_engine()

    engine.update_weights_from_disk(
        "/checkpoints/weight_v000003",
        weight_version="3",
        disable_draft_model=True,
    )

    assert calls[0][1]["disable_draft_model"] is True
