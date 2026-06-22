from __future__ import annotations

import ast
from pathlib import Path


def test_glm5_mla_passes_packed_seq_params_to_rope() -> None:
    source = Path("slime/slime_plugins/models/glm5/glm5.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.in_target = False
            self.rope_calls: list[ast.Call] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            old = self.in_target
            self.in_target = node.name == "get_absorb_query_key_value_tensors"
            self.generic_visit(node)
            self.in_target = old

        def visit_Call(self, node: ast.Call) -> None:
            if self.in_target and isinstance(node.func, ast.Attribute):
                value = node.func.value
                if (
                    node.func.attr == "rotary_pos_emb"
                    and isinstance(value, ast.Name)
                    and value.id == "self"
                ):
                    self.rope_calls.append(node)
            self.generic_visit(node)

    visitor = Visitor()
    visitor.visit(tree)

    assert visitor.rope_calls, "GLM5 MLA should call self.rotary_pos_emb in QKV preparation"
    rope_call = visitor.rope_calls[0]
    keyword_names = {keyword.arg for keyword in rope_call.keywords}
    assert "packed_seq_params" in keyword_names
    assert "packed_seq" not in keyword_names
