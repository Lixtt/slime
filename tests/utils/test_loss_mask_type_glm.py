from slime.utils.mask_utils import MultiTurnLossMaskGenerator


class FakeGLMTokenizer:
    """A tiny char-level tokenizer that models the GLM5.2 chat markers."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        encoded = {"input_ids": [ord(ch) for ch in text]}
        if return_offsets_mapping:
            encoded["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return encoded

    def decode(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids)

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        tools=None,
        add_generation_prompt=False,
        return_dict=False,
        add_special_tokens=False,
        **kwargs,
    ):
        rendered = self.render(messages, add_generation_prompt=add_generation_prompt)
        if tokenize:
            return [ord(ch) for ch in rendered]
        return rendered

    def render(self, messages, add_generation_prompt=False):
        rendered, _ = self.render_with_expected_mask(messages, add_generation_prompt=add_generation_prompt)
        return rendered

    def render_with_expected_mask(self, messages, add_generation_prompt=False):
        pieces = ["[gMASK]<sop>"]
        mask = [0] * len(pieces[0])
        last_user_index = self._find_last_user_index(messages)

        for index, message in enumerate(messages):
            role = message["role"]
            if role == "system":
                piece = f"<|system|>{message['content']}"
                pieces.append(piece)
                mask.extend([0] * len(piece))
                continue

            if role == "user":
                piece = f"<|user|>{message['content']}"
                pieces.append(piece)
                mask.extend([0] * len(piece))
                continue

            if role == "tool":
                piece = ""
                if index == 0 or messages[index - 1]["role"] != "tool":
                    piece += "<|observation|>"
                piece += f"<tool_response>{message['content']}</tool_response>"
                pieces.append(piece)
                mask.extend([0] * len(piece))
                continue

            if role != "assistant":
                raise NotImplementedError(f"Unsupported role in test tokenizer: {role}")

            content = message["content"]
            reasoning_content = message.get("reasoning_content")
            if reasoning_content is None and "</think>" in content:
                reasoning_content = content.split("</think>")[0].split("<think>")[-1]
                content = content.split("</think>")[-1]

            if reasoning_content is not None and index > last_user_index:
                target = f"<think>{reasoning_content}</think>"
            else:
                target = "<think></think>"

            if content.strip():
                target += content.strip()
            target += self._render_tool_calls(message.get("tool_calls"))

            header = "<|assistant|>"
            pieces.append(header + target)
            mask.extend([0] * len(header))
            mask.extend([1] * len(target) if message.get("step_loss_mask", 1) == 1 else [0] * len(target))

        if add_generation_prompt:
            piece = "<|assistant|><think>"
            pieces.append(piece)
            mask.extend([0] * len(piece))

        return "".join(pieces), mask

    @staticmethod
    def _render_tool_calls(tool_calls):
        if not tool_calls:
            return ""

        pieces = []
        for tool_call in tool_calls:
            function_call = tool_call.get("function", tool_call)
            pieces.append(f"<tool_call>{function_call['name']}")
            for key, value in function_call.get("arguments", {}).items():
                pieces.append(f"<arg_key>{key}</arg_key><arg_value>{value}</arg_value>")
            pieces.append("</tool_call>")
        return "".join(pieces)

    @staticmethod
    def _find_last_user_index(messages):
        last_user_index = -1
        for index, message in enumerate(messages):
            if message["role"] == "user":
                last_user_index = index
        return last_user_index


def test_glm_loss_mask_matches_multiturn_tool_flow():
    tokenizer = FakeGLMTokenizer()
    messages = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "USER"},
        {
            "role": "assistant",
            "content": "TOOL_CALL",
            "tool_calls": [{"function": {"name": "terminal", "arguments": {"command": "ls"}}}],
        },
        {"role": "tool", "content": "README.md"},
        {"role": "assistant", "content": "<think>REASONING</think>\nFINAL"},
    ]

    expected_text, expected_mask = tokenizer.render_with_expected_mask(messages)
    expected_token_ids = tokenizer(expected_text, add_special_tokens=False)["input_ids"]

    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="glm")
    token_ids, loss_mask = generator.get_loss_mask(messages)

    assert token_ids == expected_token_ids
    assert loss_mask == expected_mask
    assert generator.get_text_from_loss_mask(token_ids, loss_mask) == [
        "<think></think>TOOL_CALL<tool_call>terminal<arg_key>command</arg_key><arg_value>ls</arg_value></tool_call>",
        "<think>REASONING</think>FINAL",
    ]


def test_glm_loss_mask_honors_step_loss_mask():
    tokenizer = FakeGLMTokenizer()
    messages = [
        {"role": "user", "content": "USER_1"},
        {"role": "assistant", "content": "ANSWER_1", "step_loss_mask": 0},
        {"role": "user", "content": "USER_2"},
        {"role": "assistant", "content": "ANSWER_2"},
    ]

    expected_text, expected_mask = tokenizer.render_with_expected_mask(messages)
    expected_token_ids = tokenizer(expected_text, add_special_tokens=False)["input_ids"]

    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="glm")
    token_ids, loss_mask = generator.get_loss_mask(messages)

    assert token_ids == expected_token_ids
    assert loss_mask == expected_mask
    assert generator.get_text_from_loss_mask(token_ids, loss_mask) == ["<think></think>ANSWER_2"]
