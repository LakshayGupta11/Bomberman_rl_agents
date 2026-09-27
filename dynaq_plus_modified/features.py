from collections import deque

import numpy as np
import settings as s

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
N_ACTIONS = len(ACTIONS)
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]   # UP, RIGHT, DOWN, LEFT  
WAIT, BOMB = 4, 5

HORIZON = s.BOMB_TIMER + s.EXPLOSION_TIMER  

# Mixed-radix layout of the encoded state.
RADICES = (3, 3, 3, 3,   
           6,           
           4,           
           5,            
           2,           
           5)            
N_FIELDS = len(RADICES)
N_STATES = int(np.prod(RADICES))           
TERMINAL = N_STATES                        
N_ROWS = N_STATES + 1

# Tile status values
T_BLOCKED, T_FREE, T_DEADLY = 0, 1, 2
# target_dir values beyond the four directions
TGT_HERE, TGT_NONE = 4, 5
# bomb_effect values
B_UNSAFE, B_USELESS, B_CRATE, B_OPPONENT, B_KILL = 0, 1, 2, 3, 4


def _build_perms():
    perms = []
    for m in (0, 1):
        for r in range(4):
            perms.append(tuple(((4 - d) % 4 if m else d) + r for d in range(4)))
    return tuple(tuple(v % 4 for v in p) for p in perms)


PERM = _build_perms()                      
N_GROUP = len(PERM)                         
def _inv_perm(g):
    inv = [0] * 4
    for d in range(4):
        inv[PERM[g][d]] = d
    return tuple(inv)


INV_PERM = tuple(_inv_perm(g) for g in range(N_GROUP))


def transform_dir(v, g):
    return PERM[g][v] if v < 4 else v


def transform_action(a, g):
    return PERM[g][a] if a < 4 else a


def inverse_action(a, g):
    return INV_PERM[g][a] if a < 4 else a


def transform_features(feat, g):
    tiles = [0] * 4
    for d in range(4):
        tiles[PERM[g][d]] = feat[d]
    return (tiles[0], tiles[1], tiles[2], tiles[3],
            transform_dir(feat[4], g),
            feat[5],
            transform_dir(feat[6], g),
            feat[7], feat[8])


def encode(feat):
    idx = 0
    for value, radix in zip(feat, RADICES):
        idx = idx * radix + value
    return idx


def canonical(feat, use_symmetry=True):
    if not use_symmetry:
        return encode(feat), 0
    best, best_g = None, 0
    for g in range(N_GROUP):
        cand = transform_features(feat, g)
        if best is None or cand < best:
            best, best_g = cand, g
    return encode(best), best_g


def blast_coords(field, x, y, power=s.BOMB_POWER):
    coords = [(x, y)]
    for dx, dy in DIRS:
        for i in range(1, power + 1):
            nx, ny = x + dx * i, y + dy * i
            if field[nx, ny] == -1:
                break
            coords.append((nx, ny))
    return coords


def danger_maps(field, bombs, explosion_map, extra_bomb=None):
    w, h = field.shape
    lethal = np.zeros((HORIZON, w, h), dtype=bool)

    for step in range(min(HORIZON, s.EXPLOSION_TIMER)):
        lethal[step] |= (explosion_map > step)

    all_bombs = list(bombs)
    if extra_bomb is not None:
        all_bombs.append(extra_bomb)
        
    for (bx, by), t in all_bombs:
        coords = blast_coords(field, bx, by)
        for step in range(t, min(HORIZON, t + s.EXPLOSION_TIMER)):
            if 0 <= step < HORIZON:
                for (cx, cy) in coords:
                    lethal[step, cx, cy] = True
    return lethal


def passable(field, bombs, others):
    free = (field == 0)
    for (bx, by), _ in bombs:
        free[bx, by] = False
    for other in others:
        ox, oy = other[3]
        free[ox, oy] = False
    return free


def bfs_first_step(free, start, target_mask):
    if target_mask[start]:
        return 4, 0
    w, h = free.shape
    first = -np.ones((w, h), dtype=np.int8)
    seen = np.zeros((w, h), dtype=bool)
    seen[start] = True
    queue = deque()
    for d, (dx, dy) in enumerate(DIRS):
        nx, ny = start[0] + dx, start[1] + dy
        if free[nx, ny] and not seen[nx, ny]:
            seen[nx, ny] = True
            first[nx, ny] = d
            queue.append((nx, ny, 1))
    while queue:
        x, y, dist = queue.popleft()
        if target_mask[x, y]:
            return int(first[x, y]), dist
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if free[nx, ny] and not seen[nx, ny]:
                seen[nx, ny] = True
                first[nx, ny] = first[x, y]
                queue.append((nx, ny, dist + 1))
    return 4, -1


def find_escape(blocked, lethal, start, h_start=0):
    horizon = lethal.shape[0]
    frontier = {start: None}
    for h in range(h_start, horizon):
        nxt = {}
        for (x, y), first in frontier.items():
            for d, (dx, dy) in enumerate(DIRS):
                nx, ny = x + dx, y + dy
                if blocked[nx, ny] or lethal[h, nx, ny]:
                    continue
                if (nx, ny) not in nxt:
                    nxt[(nx, ny)] = d if first is None else first
            if not lethal[h, x, y]:
                if (x, y) not in nxt:
                    nxt[(x, y)] = WAIT if first is None else first
        if not nxt:
            return None
        frontier = nxt
    return next(iter(frontier.values()))



def kill_position_mask(field, bombs, explosion_map, others, free, blocked, me,
                       max_dist=4):
    mask = np.zeros_like(free)
    if not others:
        return mask
    opp = {o[3] for o in others}

    reach = [me]
    seen = {me}
    frontier = [me]
    for _ in range(max_dist):
        nxt = []
        for (x, y) in frontier:
            for dx, dy in DIRS:
                n = (x + dx, y + dy)
                if n not in seen and free[n]:
                    seen.add(n)
                    nxt.append(n)
                    reach.append(n)
        frontier = nxt
        if not frontier:
            break

    for (x, y) in reach:
        coords = blast_coords(field, x, y)
        hit = [c for c in coords if c in opp]
        if not hit:
            continue
        hyp = danger_maps(field, bombs, explosion_map, extra_bomb=((x, y), s.BOMB_TIMER))
        if hyp[0, x, y]:
            continue
        ours = blocked.copy()
        ours[x, y] = True
        if find_escape(ours, hyp, (x, y), h_start=1) is None:
            continue                       
        for o in hit:
            theirs = ours.copy()
            theirs[o] = False
            if find_escape(theirs, hyp, o, h_start=0) is None:
                mask[x, y] = True
                break
    return mask


def state_to_features(game_state):
    if game_state is None:
        return None

    field = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    coins = game_state['coins']
    others = game_state['others']
    _, _, bombs_left, (ax, ay) = game_state['self']

    lethal = danger_maps(field, bombs, explosion_map)
    free = passable(field, bombs, others)
    blocked = ~free

    
    tiles = []
    for dx, dy in DIRS:
        nx, ny = ax + dx, ay + dy
        if not free[nx, ny]:
            tiles.append(T_BLOCKED)
        elif lethal[0, nx, ny]:
            tiles.append(T_DEADLY)
        else:
            tiles.append(T_FREE)

   
    coin_mask = np.zeros_like(free)
    for (cx, cy) in coins:
        coin_mask[cx, cy] = True
    opp = np.zeros_like(free)
    for other in others:
        ox, oy = other[3]
        opp[ox, oy] = True


    target_dir = TGT_NONE
    step, dist = (4, -1)
    kill_mask = kill_position_mask(field, bombs, explosion_map, others,
                                   free, blocked, (ax, ay))
    if kill_mask.any():
        step, dist = bfs_first_step(free, (ax, ay), kill_mask)
    if dist < 0 and coin_mask.any():
        step, dist = bfs_first_step(free, (ax, ay), coin_mask)
    if dist < 0:
        secondary = _adjacent_mask(field, free, field == 1)
        if others:
            secondary = secondary | _adjacent_mask(field, free, opp)
        if secondary.any():
            step, dist = bfs_first_step(free, (ax, ay), secondary)
    if dist == 0:
        target_dir = TGT_HERE
    elif dist > 0:
        target_dir = step

   
    here = np.flatnonzero(lethal[:, ax, ay])
    in_danger = 0 if here.size == 0 else min(int(here[0]) + 1, 3)


    if in_danger:
        escape = find_escape(blocked, lethal, (ax, ay), h_start=0)
        escape_dir = 4 if escape is None else escape
    else:
        escape_dir = 4

   
    bomb_available = int(bool(bombs_left))
    bomb_effect = _bomb_effect(field, bombs, others, explosion_map,
                              blocked, (ax, ay), bomb_available)

    return (tiles[0], tiles[1], tiles[2], tiles[3],
            target_dir, in_danger, escape_dir, bomb_available, bomb_effect)


def _adjacent_mask(field, free, target):
    mask = np.zeros_like(free)
    xs, ys = np.nonzero(target)
    for x, y in zip(xs, ys):
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1] and free[nx, ny]:
                mask[nx, ny] = True
    return mask


def _bomb_effect(field, bombs, others, explosion_map, blocked, pos, bomb_available):
    if not bomb_available:
        return B_UNSAFE
    ax, ay = pos
    coords = blast_coords(field, ax, ay)

    hypothetical = danger_maps(field, bombs, explosion_map,
                               extra_bomb=((ax, ay), s.BOMB_TIMER))
    if hypothetical[0, ax, ay]:
        return B_UNSAFE
    own_blocked = blocked.copy()
    own_blocked[ax, ay] = True            
    if find_escape(own_blocked, hypothetical, (ax, ay), h_start=1) is None:
        return B_UNSAFE

    opponents = {other[3] for other in others}
    hit = [c for c in coords if c in opponents]
    if hit:
        for opos in hit:
            opp_blocked = own_blocked.copy()
            opp_blocked[opos] = False          
            if find_escape(opp_blocked, hypothetical, opos, h_start=0) is None:
                return B_KILL
        return B_OPPONENT
    if any(field[c] == 1 for c in coords):
        return B_CRATE
    return B_USELESS


def opponent_reach(field, bombs, others):
    w, h = field.shape
    dist = np.full((w, h), np.inf)
    walkable = (field == 0)
    for (bx, by), _ in bombs:
        walkable[bx, by] = False
    queue = deque()
    for other in others:
        ox, oy = other[3]
        dist[ox, oy] = 0
        queue.append((ox, oy))
    while queue:
        x, y = queue.popleft()
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if walkable[nx, ny] and dist[nx, ny] == np.inf:
                dist[nx, ny] = dist[x, y] + 1
                queue.append((nx, ny))
    return dist


def find_escape_robust(blocked, lethal, start, opp_dist, h_start=0):
    horizon = lethal.shape[0]
    frontier = {start: None}
    for h in range(h_start, horizon):
        nxt = {}
        for (x, y), first in frontier.items():
            for d, (dx, dy) in enumerate(DIRS):
                nx, ny = x + dx, y + dy
                if blocked[nx, ny] or lethal[h, nx, ny] or opp_dist[nx, ny] <= h + 1:
                    continue
                if (nx, ny) not in nxt:
                    nxt[(nx, ny)] = d if first is None else first
            if not lethal[h, x, y]:
                if (x, y) not in nxt:
                    nxt[(x, y)] = WAIT if first is None else first
        if not nxt:
            return None
        frontier = nxt
    return next(iter(frontier.values()))


def safe_actions(game_state):
    field = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    others = game_state['others']
    _, _, bombs_left, (ax, ay) = game_state['self']

    lethal = danger_maps(field, bombs, explosion_map)
    free = passable(field, bombs, others)
    blocked = ~free
    opp_dist = opponent_reach(field, bombs, others) if others else np.full(field.shape, np.inf)

    def survives(tile, robust):
        if lethal[0, tile[0], tile[1]]:
            return False
        if robust:
            return find_escape_robust(blocked, lethal, tile, opp_dist, h_start=1) is not None
        return find_escape(blocked, lethal, tile, h_start=1) is not None

    ok = [False] * N_ACTIONS
    robust = [False] * N_ACTIONS
    for flag, out in ((False, ok), (True, robust)):
        wait_ok = survives((ax, ay), flag)
        for d, (dx, dy) in enumerate(DIRS):
            nx, ny = ax + dx, ay + dy
            if 0 <= nx < free.shape[0] and 0 <= ny < free.shape[1] and free[nx, ny]:
                out[d] = survives((nx, ny), flag)
            else:
                out[d] = False
        out[WAIT] = wait_ok

    if bombs_left:
        hyp = danger_maps(field, bombs, explosion_map,
                          extra_bomb=((ax, ay), s.BOMB_TIMER))
        own_blocked = blocked.copy()
        own_blocked[ax, ay] = True
        if not hyp[0, ax, ay]:
            ok[BOMB] = find_escape(own_blocked, hyp, (ax, ay), h_start=1) is not None
            robust[BOMB] = find_escape_robust(own_blocked, hyp, (ax, ay), opp_dist,
                                              h_start=1) is not None
    return ok, robust


def decode(idx):
    values = []
    for radix in reversed(RADICES):
        values.append(idx % radix)
        idx //= radix
    return tuple(reversed(values))
