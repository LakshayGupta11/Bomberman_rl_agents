import os

import numpy as np

from . import config as C
from .features import (ACTIONS, BOMB, N_ACTIONS, canonical, inverse_action, safe_actions,
                       state_to_features, transform_action)
from .dyna import DynaQPlus

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), C.MODEL_FILE)

TRACE_PATH = os.environ.get('DYNAQ_TRACE')


def setup(self):
    """Called once per match, not once per round."""
    self.rng = np.random.default_rng()
    self.agent = DynaQPlus(rng=self.rng)

    if os.path.isfile(MODEL_PATH):
        try:
            status = self.agent.load(MODEL_PATH)
            self.logger.info(f"Loaded model from {C.MODEL_FILE} ({status}), "
                             f"t={self.agent.t}, rounds={self.agent.rounds}")
        except BaseException as exc:
            self.logger.error(f"Could not load {C.MODEL_FILE} ({exc!r}); starting fresh.")
            self.agent = DynaQPlus(rng=self.rng)
    else:
        self.logger.info("No checkpoint found, starting from a zeroed table.")

    self.epsilon = C.EPS_START if self.train else C.EPS_EVAL
    self.current_round = 0
    self.last_c = None
    self.last_feat = None
    self.last_g = 0
    self.last_action_idx = None
    self.last_step = -1


def _apply_safety_filter(self, game_state, feat, c, g, a_real):
    ok, robust = safe_actions(game_state)
    allowed = list(ok)
    if C.ROBUST_BOMB_DROP:
        allowed[BOMB] = robust[BOMB]       
    if C.ROBUST_ESCAPE and feat[5] > 0 and any(a and r for a, r in zip(allowed, robust)):
        allowed = [a and r for a, r in zip(allowed, robust)]
    if allowed[a_real] or not any(allowed):
        return a_real
    q = self.agent.Q[c]
    cand = [a for a in range(N_ACTIONS) if allowed[a]]
    vals = np.array([q[transform_action(a, g)] for a in cand])
    best = [cand[i] for i in np.flatnonzero(vals == vals.max())]
    return int(self.rng.choice(best))


def act(self, game_state: dict) -> str:
    if game_state is None:
        return 'WAIT'

    if game_state['round'] != self.current_round:
        self.current_round = game_state['round']
        self.last_c = None
        self.last_step = -1

    feat = state_to_features(game_state)
    c, g = canonical(feat, C.USE_SYMMETRY)

    epsilon = self.epsilon if self.train else C.EPS_EVAL
    a_canon = self.agent.epsilon_greedy(c, epsilon)
    a_real = inverse_action(a_canon, g)

    if C.SAFETY_FILTER_PLAY:
        a_real = _apply_safety_filter(self, game_state, feat, c, g, a_real)

    self.last_feat = feat
    self.last_c = c
    self.last_g = g
    self.last_action_idx = a_real
    self.last_step = game_state['step']

    if TRACE_PATH:
        with open(TRACE_PATH, 'a') as fh:
            fh.write(','.join(str(v) for v in feat) + f',{a_real},{c}\n')

    return ACTIONS[a_real]
