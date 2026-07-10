from __future__ import annotations

import ast
from pathlib import Path

ACTOR_PATH = (
    Path(__file__).resolve().parents[1]
    / "slime"
    / "backends"
    / "megatron_utils"
    / "actor.py"
)


def _actor_method(name: str) -> ast.FunctionDef:
    tree = ast.parse(ACTOR_PATH.read_text(encoding="utf-8"))
    actor_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MegatronTrainRayActor"
    )
    return next(
        node
        for node in actor_class.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_megatron_global_trainable_token_check_uses_data_parallel_sum() -> None:
    method = _actor_method("_has_global_trainable_tokens")
    source = ast.unparse(method)

    assert "loss_mask.sum(dtype=torch.float32)" in source
    assert "dist.all_reduce" in source
    assert "mpu.get_data_parallel_group(with_context_parallel=False)" in source
    assert "return bool(count_tensor.item() > 0)" in source


def test_megatron_train_skips_optimizer_path_for_all_padding() -> None:
    method = _actor_method("train")
    source = ast.unparse(method)

    check_position = source.index(
        "has_trainable_tokens = self._has_global_trainable_tokens(rollout_data)"
    )
    skip_position = source.index("if not has_trainable_tokens:")
    critic_position = source.index("elif self.role == 'critic':")
    actor_position = source.index(
        "self.train_actor(rollout_id, rollout_data, external_data=external_data)"
    )

    assert check_position < skip_position < critic_position < actor_position
    skip_branch = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.op, ast.Not)
        and isinstance(node.test.operand, ast.Name)
        and node.test.operand.id == "has_trainable_tokens"
    )
    assert all(
        not (
            isinstance(call.func, ast.Attribute)
            and call.func.attr in {"train_actor", "train_critic"}
        )
        for statement in skip_branch.body
        for call in ast.walk(statement)
        if isinstance(call, ast.Call)
    )
