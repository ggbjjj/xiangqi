import numpy as np
import pytest

torch = pytest.importorskip('torch')
torch.set_num_threads(min(2, torch.get_num_threads()))

from ai import NeuralAI, load_network
from board import Board
from encoder import NUM_ACTIONS, move_to_action
from mcts import MCTS, MCTSNode
from net import PolicyValueNet
from piece import Piece
from train import _play_one_self_game, build_samples


def test_selection_uses_parent_perspective():
    root = MCTSNode()
    root.N = 20
    winning, losing = MCTSNode(0.5), MCTSNode(0.5)
    winning.N = losing.N = 10
    winning.W, losing.W = -10, 10
    root.children = {1: winning, 2: losing}
    assert MCTS(None)._select_child(root) == 1


def test_mcts_selects_forced_mate():
    board = Board([[None] * 9 for _ in range(10)])
    for r, c, color, kind in [(9, 5, 'r', 'K'), (0, 4, 'b', 'K'),
                               (2, 3, 'r', 'R'), (1, 8, 'r', 'R'), (2, 4, 'r', 'P')]:
        board.grid[r][c] = Piece(color, kind)
    mating_move = (1, 8, 1, 4)
    assert board.apply_move(*mating_move).is_checkmate('b')

    class FixedNet:
        def __call__(self, state):
            logits = torch.zeros((1, NUM_ACTIONS))
            logits[0, move_to_action(mating_move)] = 2
            return logits, torch.zeros((1, 1))

    ai = NeuralAI(FixedNet(), num_simulations=30)
    selected = ai.choose_move(board, 'r')
    assert board.apply_move(*selected).is_checkmate('b')
    assert np.isclose(ai.last_policy.sum(), 1)


def test_terminal_root_does_not_call_network():
    board = Board([[None] * 9 for _ in range(10)])
    root, value = MCTS(None, num_simulations=1).run(board, 'r')
    assert value == -1 and not root.children


def test_selfplay_samples_share_the_selected_policy():
    net = PolicyValueNet(channels=8, num_blocks=0)
    result = _play_one_self_game(net.state_dict(), 2, 2, 2, 42, False, 8, 0)
    states, policies, winner, count = result
    assert count == 2 and len(states) == len(policies) == 2
    assert np.allclose(policies.sum(axis=1), 1)
    assert np.isfinite(policies).all()
    assert winner is None


def test_value_labels_follow_encoded_turn():
    states = np.zeros((2, 15, 10, 9), dtype=np.float32)
    states[1, 14] = 1
    _, _, values = build_samples(states, np.zeros((2, NUM_ACTIONS)), 'r')
    assert values.tolist() == [-1, 1]


def test_checkpoint_metadata_loads_custom_network(tmp_path):
    net = PolicyValueNet(channels=8, num_blocks=0)
    path = tmp_path / 'model.pt'
    torch.save({'model': net.state_dict(), 'config': {'channels': 8, 'num_blocks': 0}}, path)
    loaded = load_network(path)
    assert loaded.stem[0].out_channels == 8
    assert len(loaded.backbone) == 0
    assert not loaded.training
