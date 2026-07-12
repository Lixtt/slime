import sys
import types

import torch

from slime.utils.ppo_utils import (
    get_advantages_and_returns_batch,
    get_trajectory_advantages_and_returns,
    vanilla_gae,
)


def test_trajectory_gae_chains_segments_over_masked_context():
    advantages, returns = get_trajectory_advantages_and_returns(
        response_lengths=[3, 2],
        values_list=[torch.zeros(3), torch.zeros(2)],
        token_rewards_list=[torch.zeros(3), torch.zeros(2)],
        loss_masks=[torch.tensor([1, 0, 1]), torch.tensor([0, 1])],
        trajectory_ids=[7, 7],
        segment_indices=[0, 1],
        segment_counts=[2, 2],
        terminal_rewards=[1.0, 1.0],
        gamma=0.5,
        lambd=1.0,
        chunked=False,
    )

    torch.testing.assert_close(advantages[0], torch.tensor([0.25, 0.0, 0.5]))
    torch.testing.assert_close(advantages[1], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(returns[0], advantages[0])
    torch.testing.assert_close(returns[1], advantages[1])


def test_trajectory_gae_sorts_segments_and_matches_whole_episode():
    values = [torch.tensor([0.3, 0.4]), torch.tensor([0.1])]
    token_rewards = [torch.tensor([-0.03, -0.04]), torch.tensor([-0.01])]

    advantages, returns = get_trajectory_advantages_and_returns(
        response_lengths=[2, 1],
        values_list=values,
        token_rewards_list=token_rewards,
        loss_masks=[torch.ones(2), torch.ones(1)],
        trajectory_ids=[11, 11],
        segment_indices=[1, 0],
        segment_counts=[2, 2],
        terminal_rewards=[2.0, 2.0],
        gamma=0.9,
        lambd=0.8,
        chunked=False,
    )

    whole_values = torch.tensor([[0.1, 0.3, 0.4]])
    whole_rewards = torch.tensor([[-0.01, -0.03, 1.96]])
    expected_advantages, expected_returns = vanilla_gae(
        whole_rewards,
        whole_values,
        gamma=0.9,
        lambd=0.8,
    )

    torch.testing.assert_close(advantages[1], expected_advantages[0, :1])
    torch.testing.assert_close(advantages[0], expected_advantages[0, 1:])
    torch.testing.assert_close(returns[1], expected_returns[0, :1])
    torch.testing.assert_close(returns[0], expected_returns[0, 1:])


def test_zero_mask_padding_trajectory_stays_zero():
    advantages, returns = get_trajectory_advantages_and_returns(
        response_lengths=[2],
        values_list=[torch.tensor([3.0, 4.0])],
        token_rewards_list=[torch.tensor([5.0, 6.0])],
        loss_masks=[torch.zeros(2)],
        trajectory_ids=[-1],
        segment_indices=[0],
        segment_counts=[1],
        terminal_rewards=[-1.0],
        gamma=0.99,
        lambd=0.95,
    )

    torch.testing.assert_close(advantages[0], torch.zeros(2))
    torch.testing.assert_close(returns[0], torch.zeros(2))


def test_batched_gae_uses_trajectory_layout(monkeypatch):
    mpu = types.SimpleNamespace(get_context_parallel_world_size=lambda: 1)
    megatron_core = types.ModuleType("megatron.core")
    megatron_core.mpu = mpu
    monkeypatch.setitem(sys.modules, "megatron", types.ModuleType("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.core", megatron_core)

    advantages, returns = get_advantages_and_returns_batch(
        total_lengths=[3, 2],
        response_lengths=[2, 1],
        values_list=[torch.zeros(2), torch.zeros(1)],
        rewards_list=[torch.zeros(2), torch.zeros(1)],
        gamma=0.5,
        lambd=1.0,
        chunked=False,
        loss_masks=[torch.ones(2), torch.ones(1)],
        trajectory_ids=[3, 3],
        segment_indices=[0, 1],
        segment_counts=[2, 2],
        terminal_rewards=[1.0, 1.0],
    )

    torch.testing.assert_close(advantages[0], torch.tensor([0.25, 0.5]))
    torch.testing.assert_close(advantages[1], torch.tensor([1.0]))
    torch.testing.assert_close(returns[0], advantages[0])
    torch.testing.assert_close(returns[1], advantages[1])

