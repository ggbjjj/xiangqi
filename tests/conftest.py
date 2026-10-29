import os

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'


import pytest


@pytest.fixture
def assert_rollout_replay():
    """Audit original sampling with full neural history, outside cropped training."""
    def replay(model,path,sample):
        import json
        import torch
        from data_pipeline.notation import from_fen
        from encoder import encode_state,action_to_move
        from selfplay.store import load_game
        from training.exploration import behavior_log_probs
        from training.loss import split_output
        game,_,_=load_game(path,sample.source_game_id)
        prefix=json.loads(game['prefix_json'])
        actions=prefix+json.loads(game['moves_json'])[:sample.start_ply+sample.length]
        board,turn=from_fen(game['start_fen']);cache=None
        with torch.inference_mode():
            for ply,action in enumerate(actions):
                x=torch.from_numpy(encode_state(board,turn))[None]
                output,cache=model.forward_step(x,cache)
                t=ply-len(prefix)-sample.start_ply
                if t>=0:
                    policy,_=split_output(output)
                    log_mu=behavior_log_probs(policy[0],sample.legal[t],sample.temperature[t],sample.epsilon[t])
                    assert float(log_mu[action])==pytest.approx(float(sample.old_log_mu[t]),abs=2e-5)
                board=board.apply_move(*action_to_move(action))
                turn='b' if turn=='r' else 'r'
    return replay
