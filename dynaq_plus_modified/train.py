import os
from typing import List

import numpy as np

import events as e

from . import config as C
from .features import ACTIONS, TERMINAL, canonical, state_to_features, transform_action
from .rewards import custom_events, reward_from_events

STATS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stats')
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), C.MODEL_FILE)

STATS_HEADER = ("round,steps,score,coins,crates,kills,invalid,suicide,died,"
                "epsilon,mean_delta,coverage,visited_states,mean_bonus,t\n")


def setup_training(self):
    self.epsilon = getattr(self.agent, 'epsilon', C.EPS_START)
    self.eps_decay = (C.EPS_END / C.EPS_START) ** (1.0 / max(1, C.EPS_DECAY_ROUNDS))
    self.processed_step = -1
    _reset_round_stats(self)

    os.makedirs(STATS_DIR, exist_ok=True)
    self.stats_path = os.path.join(STATS_DIR, 'train.csv')
    if not os.path.isfile(self.stats_path):
        with open(self.stats_path, 'w') as fh:
            fh.write(STATS_HEADER)
    self.logger.info(f"Dyna-Q+ training: n_planning={C.N_PLANNING}, kappa={C.KAPPA}, "
                     f"gamma={C.GAMMA}, alpha={C.ALPHA}, symmetry={C.USE_SYMMETRY}, "
                     f"stochastic_model={C.STOCHASTIC_MODEL}")


def _reset_round_stats(self):
    self.round_stats = dict(coins=0, crates=0, kills=0, invalid=0,
                            suicide=0, died=0, delta_sum=0.0, delta_n=0)


def _tally(self, events):
    st = self.round_stats
    st['coins'] += events.count(e.COIN_COLLECTED)
    st['crates'] += events.count(e.CRATE_DESTROYED)
    st['kills'] += events.count(e.KILLED_OPPONENT)
    st['invalid'] += events.count(e.INVALID_ACTION)
    if e.KILLED_SELF in events:
        st['suicide'] = 1
    if e.GOT_KILLED in events:
        st['died'] = 1


def _learn(self, old_state, action_str, new_state, events, terminal):
    if action_str is None:
        return

    if old_state is not None and old_state['step'] == self.last_step and self.last_c is not None:
        c, g, old_feat = self.last_c, self.last_g, self.last_feat
    else:
        old_feat = state_to_features(old_state)
        if old_feat is None:
            return
        c, g = canonical(old_feat, C.USE_SYMMETRY)

    a_real = ACTIONS.index(action_str)
    a_canon = transform_action(a_real, g)

    if terminal:
        new_feat, c_next = None, TERMINAL
    else:
        new_feat = state_to_features(new_state)
        c_next, _ = canonical(new_feat, C.USE_SYMMETRY)

    events = list(events) + custom_events(old_feat, a_real, new_feat, events)
    reward = reward_from_events(events)

    delta = self.agent.observe(c, a_canon, reward, c_next)
    self.round_stats['delta_sum'] += delta
    self.round_stats['delta_n'] += 1

    self.agent.plan(C.N_PLANNING_END if terminal else C.N_PLANNING)


def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: List[str]):
    _tally(self, events)
    _learn(self, old_game_state, self_action, new_game_state, events, terminal=False)
    self.processed_step = new_game_state['step']


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    already_seen = (last_game_state is not None
                    and last_game_state['step'] == self.processed_step)

    if not already_seen:
        _tally(self, events)
        _learn(self, last_game_state, last_action, None, events, terminal=True)
    else:
        self.agent.plan(C.N_PLANNING_END)

    self.agent.rounds += 1
    self.epsilon = max(C.EPS_END, self.epsilon * self.eps_decay)
    self.agent.epsilon = self.epsilon
    _write_stats(self, last_game_state)
    _reset_round_stats(self)

    if self.agent.rounds % C.SAVE_EVERY == 0:
        self.agent.save(MODEL_PATH)
        self.logger.info(f"Checkpoint at round {self.agent.rounds}, t={self.agent.t}, "
                         f"coverage={self.agent.coverage()}")


def _write_stats(self, last_game_state):
    st = self.round_stats
    steps = last_game_state['step'] if last_game_state else 0
    score = last_game_state['self'][1] if last_game_state else 0
    mean_delta = st['delta_sum'] / st['delta_n'] if st['delta_n'] else 0.0
    with open(self.stats_path, 'a') as fh:
        fh.write(f"{self.agent.rounds},{steps},{score},{st['coins']},{st['crates']},"
                 f"{st['kills']},{st['invalid']},{st['suicide']},{st['died']},"
                 f"{self.epsilon:.4f},{mean_delta:.5f},{self.agent.coverage()},"
                 f"{len(self.agent.visited_list)},{self.agent.mean_bonus():.5f},"
                 f"{self.agent.t}\n")
