import numpy as np

from gridworld import *
from mdp import MDP
import m0n0_staticobstacle
import copy
import itertools

nrows = 9
ncols = 7
targets = [[50]]
obstacles = [31,24,38,45,10,17]
initial = [54]
nagents = 1
moveobstacles = []
regionkeys = {'pavement','gravel','grass','sand','deterministic'}
regions = dict.fromkeys(regionkeys,{-1})
# regions['deterministic']= {42,43,44,64,65,66,85,86,87,88,107,108,109,110,111,112,113,130,131,132,133,152,153,172,173,192,193,194,195,213,214,215}
# regions['pavement'] = set(range(nrows*ncols)) - regions['deterministic']
regions['sand'] = set(range(nrows*ncols))
gwg = Gridworld(initial, nrows, ncols,nagents, targets, obstacles, moveobstacles,regions)
gwg.render()

gwg.draw_state_labels()

gridstates = range(gwg.nstates)
obsstates = np.array([0]*gwg.nstates)
possobs = [52,59]
trueobsstates = np.array([0]*gwg.nstates)
trueobsstates[gwg.obstacles] = 1
trueobsstates = tuple(trueobsstates)
allobsstates = list(itertools.product([0,0.5,1],repeat=len(possobs)))
alphabet = range(gwg.nactions)
accepting_states = set()
transitions = []
states = []
dist = 2
initialobs = (0.5,)*len(possobs)
for s in gridstates:
    for o in allobsstates:
        combinedstate = tuple([s])+o
        states.append(combinedstate)
        for a in alphabet:
            if s not in targets[0] and not (s in possobs and o[possobs.index(s)]==1):
                for t in np.nonzero(gwg.prob[gwg.actlist[a]][s])[0]:
                    p = gwg.prob[gwg.actlist[a]][s][t]
                    cs = set(gwg.close_states(s,dist))
                    cs = list(cs.intersection(possobs))
                    # cs = [x for x in cs if o[x] > 0]
                    if len(cs) == 0:
                        transitions.append((tuple([s])+o, a, tuple([t])+o, p))
                    else:
                        obscombos = list(itertools.product(range(2),repeat=len(cs)))
                        for i in obscombos:
                            new_o = list(copy.deepcopy(o))
                            p2 = copy.deepcopy(p)
                            for j in cs:
                                new_o[possobs.index(j)] = i[cs.index(j)]
                                if i[cs.index(j)] == 0:
                                    p2 = p2*(1-(o[cs.index(j)]))
                                else:
                                    p2 = p2*((o[cs.index(j)]))
                            if p2 > 0:
                                transitions.append((tuple([s])+o, a, tuple([t])+tuple(new_o), p2))
            elif s in possobs and o[possobs.index(s)]==1:
                transitions.append((tuple([s])+o,a,tuple([s])+o,1))
            else:
                accepting_states.add(tuple([s])+o)
                transitions.append((tuple([s])+o,a,tuple([s])+o,1))
mdp = MDP(states, accepting_states, alphabet, transitions)
T = 20
beta = 0.5
cost = []
plt = True
pltiter = 1
iter = 30
m0n0_staticobstacle.alg_m0n0_staticobstacle(gwg,mdp,possobs,iter,T,beta,cost, plt,pltiter)

# U, policy = mdp.value_iteration(10)
# print U[tuple(initial) + initialobs]
# truepolicy1 = dict()
# truepolicy2 = copy.deepcopy(truepolicy1)
# for s in gridstates:
#     truepolicy1[s] = policy[tuple([s]) + initialobs]
#     truepolicy2[s] = set()
#     for a in truepolicy1[s]:
#         act = gwg.actlist[a]
#     truepolicy2[s].add(act)
# # print truepolicy2
# asdf = 1