import events as e

from .features import B_UNSAFE, B_CRATE, B_OPPONENT, B_KILL

MOVED_TOWARD_TARGET = 'MOVED_TOWARD_TARGET'
MOVED_AWAY_FROM_TARGET = 'MOVED_AWAY_FROM_TARGET'
ESCAPED_DANGER = 'ESCAPED_DANGER'
MOVED_INTO_DANGER = 'MOVED_INTO_DANGER'
WAITED_IN_DANGER = 'WAITED_IN_DANGER'
USEFUL_BOMB = 'USEFUL_BOMB'
SUICIDAL_BOMB = 'SUICIDAL_BOMB'
REVERSED_DIRECTION = 'REVERSED_DIRECTION' 

REWARDS = {
    e.COIN_COLLECTED: 2.0,
    e.KILLED_OPPONENT: 5.0,
    e.CRATE_DESTROYED: 0.3,     
    e.COIN_FOUND: 0.2,
    e.OPPONENT_ELIMINATED: 0.1,
    e.SURVIVED_ROUND: 0.0,      
    e.BOMB_DROPPED: 0.0,        
    e.BOMB_EXPLODED: 0.0,
    e.INVALID_ACTION: -0.5,
    e.WAITED: -0.05,             
    e.MOVED_UP: -0.02,
    e.MOVED_RIGHT: -0.02,
    e.MOVED_DOWN: -0.02,
    e.MOVED_LEFT: -0.02,
    e.KILLED_SELF: -12.0,  
    e.GOT_KILLED: -4.0,

    MOVED_TOWARD_TARGET: 0.15,
    MOVED_AWAY_FROM_TARGET: -0.18,   
    ESCAPED_DANGER: 0.5,
    MOVED_INTO_DANGER: -0.5,
    WAITED_IN_DANGER: -0.3,
    USEFUL_BOMB: 0.2,
    SUICIDAL_BOMB: -3.0,
    REVERSED_DIRECTION: -0.15,  
}


def custom_events(old_feat, action_idx, new_feat, events, last_action_idx=None):
    extra = []
    if old_feat is None:
        return extra

    old_target, old_danger, old_bomb = old_feat[4], old_feat[5], old_feat[8]
    dropped = e.BOMB_DROPPED in events

    if dropped:
        if old_bomb == B_UNSAFE:
            extra.append(SUICIDAL_BOMB)
        elif old_bomb in (B_CRATE, B_OPPONENT, B_KILL):
            extra.append(USEFUL_BOMB)

    if (action_idx is not None and action_idx < 4
            and old_target < 4 and old_danger == 0):
        extra.append(MOVED_TOWARD_TARGET if action_idx == old_target
                     else MOVED_AWAY_FROM_TARGET)

    
    if (action_idx is not None and action_idx < 4 
            and last_action_idx is not None and last_action_idx < 4
            and old_danger == 0):
        if action_idx == (last_action_idx + 2) % 4:
            extra.append(REVERSED_DIRECTION)

    if e.WAITED in events and old_danger > 0:
        extra.append(WAITED_IN_DANGER)

    if new_feat is not None:
        new_danger = new_feat[5]
        if old_danger > 0 and new_danger == 0:
            extra.append(ESCAPED_DANGER)
        elif old_danger == 0 and new_danger > 0 and not dropped:
            extra.append(MOVED_INTO_DANGER)

    return extra


def reward_from_events(events):
    return sum(REWARDS.get(ev, 0.0) for ev in events)