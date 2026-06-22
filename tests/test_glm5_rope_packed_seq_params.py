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


def test_glm5_mla_rope_has_unfused_thd_fallback() -> None:
    source = Path("slime/slime_plugins/models/glm5/glm5.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    apex_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("apex")
    ]
    assert not apex_imports

    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "_apply_rotary_pos_emb_thd" in imported_names
    assert "_apply_rotary_pos_emb_bshd" in imported_names
    assert "fused_apply_rotary_pos_emb_thd" in imported_names

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.in_fuse_rope = False
            self.references_apply_rope_fusion = False
            self.calls_unfused = False

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            old = self.in_fuse_rope
            self.in_fuse_rope = node.name == "fuse_rope"
            self.generic_visit(node)
            self.in_fuse_rope = old

        def visit_Attribute(self, node: ast.Attribute) -> None:
            if self.in_fuse_rope and node.attr == "apply_rope_fusion":
                self.references_apply_rope_fusion = True
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            if self.in_fuse_rope and isinstance(node.func, ast.Name):
                if node.func.id == "apply_unfused_rope":
                    self.calls_unfused = True
            self.generic_visit(node)

    visitor = Visitor()
    visitor.visit(tree)

    assert visitor.references_apply_rope_fusion
    assert visitor.calls_unfused
