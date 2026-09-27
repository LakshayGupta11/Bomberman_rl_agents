FEATURE_VERSION = 4
REWARD_VERSION = 3

ALPHA = 0.10         
GAMMA = 0.95          
KAPPA = 0.001         
N_PLANNING = 20       
N_PLANNING_END = 200  

# exploration
EPS_START = 0.30
EPS_END = 0.05
EPS_DECAY_ROUNDS = 4000   
EPS_EVAL = 0.01          

# model
STOCHASTIC_MODEL = True   
MAX_OUTCOMES = 8         
TAU_CAP = 100_000.0       
BONUS_ON_DEATH = False    


SAFETY_FILTER_PLAY = True
ROBUST_BOMB_DROP = True  
ROBUST_ESCAPE = True      

USE_SYMMETRY = True      

#checkpoint
SAVE_EVERY = 100         
MODEL_FILE = "dynaq_plus_model.pt"
