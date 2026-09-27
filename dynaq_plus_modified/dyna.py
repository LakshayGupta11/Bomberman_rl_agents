import os
import pickle

import numpy as np

from . import config as C
from .features import N_ROWS, N_ACTIONS, TERMINAL


class DynaQPlus:
    def __init__(self, rng=None, n_rows=N_ROWS, n_actions=N_ACTIONS, terminal=TERMINAL):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.n_rows, self.n_actions, self.terminal = n_rows, n_actions, terminal
        self.Q = np.zeros((n_rows, n_actions), dtype=np.float32)
        self.N = np.zeros((n_rows, n_actions), dtype=np.int32)
        self.last_tried = np.zeros((n_rows, n_actions), dtype=np.int32)
        self.visited = np.zeros(n_rows, dtype=bool)
        self.visited_list = []         
        self.t = 0                     
        self.rounds = 0
        self.epsilon = C.EPS_START

       
        self.m_next = np.full((n_rows, n_actions), terminal, dtype=np.int32)
        self.m_rew = np.zeros((n_rows, n_actions), dtype=np.float32)
        self.m_seen = np.zeros((n_rows, n_actions), dtype=bool)
        
        self.m_counts = {}

    def greedy(self, c):
        q = self.Q[c]
        best = np.flatnonzero(q == q.max())
        return int(best[0]) if best.size == 1 else int(self.rng.choice(best))

    def epsilon_greedy(self, c, epsilon):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.greedy(c)

    # learning
    def q_update(self, c, a, r, c_next, alpha=None, gamma=None):
        alpha = C.ALPHA if alpha is None else alpha
        gamma = C.GAMMA if gamma is None else gamma
        target = r + gamma * self.Q[c_next].max()
        delta = target - self.Q[c, a]
        self.Q[c, a] += alpha * delta
        return abs(float(delta))

    def observe(self, c, a, r, c_next):
        self.t += 1
        delta = self.q_update(c, a, r, c_next)

        if not self.visited[c]:
            self.visited[c] = True
            self.visited_list.append(c)

        self.m_next[c, a] = c_next
        self.m_rew[c, a] = r
        self.m_seen[c, a] = True
        if C.STOCHASTIC_MODEL:
            outcomes = self.m_counts.setdefault((c, a), {})
            entry = outcomes.get(c_next)
            if entry is None:
                if len(outcomes) >= C.MAX_OUTCOMES:
                    worst = min(outcomes, key=lambda k: outcomes[k][0])
                    del outcomes[worst]
                outcomes[c_next] = [1, float(r)]
            else:
                entry[0] += 1
                entry[1] += float(r)

        self.N[c, a] += 1
        self.last_tried[c, a] = self.t
        return delta

    #  planning
    def _sample_outcome(self, c, a):
        if C.STOCHASTIC_MODEL:
            outcomes = self.m_counts.get((c, a))
            if not outcomes:
                return 0.0, c, False
            keys = list(outcomes.keys())
            counts = np.fromiter((outcomes[k][0] for k in keys), dtype=np.float64,
                                 count=len(keys))
            pick = keys[int(self.rng.choice(len(keys), p=counts / counts.sum()))]
            count, rsum = outcomes[pick]
            return rsum / count, pick, True
        if not self.m_seen[c, a]:
            return 0.0, c, False
        return float(self.m_rew[c, a]), int(self.m_next[c, a]), True

    def plan(self, n_steps):
        if n_steps <= 0 or not self.visited_list:
            return 0.0
        total = 0.0
        states = self.visited_list
        n_states = len(states)
        for _ in range(n_steps):
            c = states[int(self.rng.integers(n_states))]
            a = int(self.rng.integers(self.n_actions))
            r, c_next, known = self._sample_outcome(c, a)

            tau = self.t - self.last_tried[c, a]
            bonus = C.KAPPA * np.sqrt(min(float(tau), C.TAU_CAP))

            if (not C.BONUS_ON_DEATH and known and c_next == self.terminal and r < 0) or r <= -0.5:
                bonus = 0.0

            total += self.q_update(c, a, r + bonus, c_next)
        return total / n_steps

    def mean_bonus(self):
        if not self.visited_list:
            return 0.0
        rows = np.array(self.visited_list)
        tau = np.minimum(self.t - self.last_tried[rows], C.TAU_CAP)
        return float(C.KAPPA * np.sqrt(tau).mean())

    def coverage(self, min_visits=1):
        return int((self.N >= min_visits).sum())

    def state_dict(self, slim=False):
        d = {
            'feature_version': C.FEATURE_VERSION,
            'reward_version': C.REWARD_VERSION,
            'Q': self.Q,
            'rounds': self.rounds,
            't': self.t,
            'epsilon': self.epsilon,
        }
        if not slim:
            d.update({
                'N': self.N,
                'last_tried': self.last_tried,
                'visited': self.visited,
                'm_next': self.m_next,
                'm_rew': self.m_rew,
                'm_seen': self.m_seen,
                'm_counts': self.m_counts,
            })
        return d

    def load_state_dict(self, d):
        if d.get('feature_version') != C.FEATURE_VERSION:
            return 'rejected: feature version mismatch'
        self.Q = np.asarray(d['Q'], dtype=np.float32)
        self.rounds = int(d.get('rounds', 0))
        self.t = int(d.get('t', 0))
        self.epsilon = float(d.get('epsilon', C.EPS_START))
        if d.get('reward_version') != C.REWARD_VERSION:
            return 'partial: reward version changed, model discarded, Q kept'
        if 'm_next' in d:
            self.N = np.asarray(d['N'], dtype=np.int32)
            self.last_tried = np.asarray(d['last_tried'], dtype=np.int32)
            self.visited = np.asarray(d['visited'], dtype=bool)
            self.visited_list = [int(c) for c in np.flatnonzero(self.visited)]
            self.m_next = np.asarray(d['m_next'], dtype=np.int32)
            self.m_rew = np.asarray(d['m_rew'], dtype=np.float32)
            self.m_seen = np.asarray(d['m_seen'], dtype=bool)
            self.m_counts = d.get('m_counts', {})
            return 'full'
        return 'partial: Q only'

    def save(self, path, slim=False):
        tmp = path + '.tmp'
        with open(tmp, 'wb') as fh:
            pickle.dump(self.state_dict(slim=slim), fh, protocol=4)
        os.replace(tmp, path)

    def load(self, path):
        with open(path, 'rb') as fh:
            return self.load_state_dict(pickle.load(fh))
