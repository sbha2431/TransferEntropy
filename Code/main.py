import numpy as np

from gridworld import *
from mdp import MDP
import m0n0_moveobstacle



#Define gridworld parameters
nrows = 15
ncols = 20
nagents = 1
initial = [237]
targets = [[261]]
moveobstacles = [267]
# obstacles = [153,154,155,173,174,175,193,194,195,213,214,215,233,234,235,68,69,88,89,108,109,128,129,183,184,185,186,187,203,204,205,206,207,223,224,225,226,227]
obstacles = [62,63,83,84,104,105,106,126,127,128,129,149,150,151,171,191,191,211,45,46,47,48,49,50,70,67,68,69,89,90,91,92,93,94,114,134,154,174,175,176,196,216]
# obstacles = [15,16,19]


# nrows = 9
# ncols = 7
# targets = [[50]]
# obstacles = [31,24]
# initial = [54]
# nagents = 1
# moveobstacles = [52]
regionkeys = {'pavement','gravel','grass','sand','deterministic'}
regions = dict.fromkeys(regionkeys,{-1})
# regions['deterministic']= {42,43,44,64,65,66,85,86,87,88,107,108,109,110,111,112,113,130,131,132,133,152,153,172,173,192,193,194,195,213,214,215}
# regions['pavement'] = set(range(nrows*ncols)) - regions['deterministic']
regions['deterministic'] = set(range(nrows*ncols))
gwg = Gridworld(initial, nrows, ncols,nagents, targets, obstacles, moveobstacles,regions)
gwg.render()

gwg.draw_state_labels()

states = range(gwg.nstates)
alphabet = range(gwg.nactions)
accepting_states = set()
transitions = []
for s in states:
    for a in alphabet:
        if s not in targets:
            for t in np.nonzero(gwg.prob[gwg.actlist[a]][s])[0]:
                p = gwg.prob[gwg.actlist[a]][s][t]
                transitions.append((s, a, t, p))
        else:
            accepting_states.add(s)
            transitions.append((s,a,s,1))
mdp = MDP(states, accepting_states, alphabet, transitions)

# obstaclemovestates = [38,45,52]
obstaclemovestates = [267,247,227,207,187,167,147]

accepting_states = set()
transitions = []
alphabet = range(1)
for s in states:
    for a in alphabet:
        if s in obstaclemovestates:
            neigh_states = set()
            for act in gwg.actlist:
                neigh_states = neigh_states.union(set(np.nonzero(gwg.prob[act][s])[0]))
            neigh_states = neigh_states.intersection(set(obstaclemovestates))
            for t in neigh_states:
                transitions.append((s, a, t, 1.0/len(neigh_states)))
        else:
            transitions.append((s, a, s, 0))
# for s in obstaclemovestates:
#     for a in alphabet:
#         for t in obstaclemovestates:
#             transitions.append((s, a, t, 1.0/len(obstaclemovestates)))
#
#
obstacle_mdp = MDP(states, accepting_states, alphabet, transitions)


T = 60
beta = 10/2.0
# cost = np.full((gwg.nactions),1.0)
cost = np.full((gwg.ncols,gwg.nrows,gwg.ncols*len(moveobstacles),gwg.nrows*len(moveobstacles),gwg.nactions),0)
# for x in range(gwg.ncols):
#     for y in range(gwg.nrows):
#         for x2 in range(gwg.ncols):
#             for y2 in range(gwg.nrows):
#                 if x == x2 and y == y2:
#                     cost[x,y,x2,y2,:] = np.full(gwg.nactions,10)
for s in obstaclemovestates:
    for s2 in obstaclemovestates:
        if s == s2:
            cost[gwg.coords(s)[1],gwg.coords(s)[0],gwg.coords(s2)[1],gwg.coords(s2)[0],:] = np.full(gwg.nactions,10)

pltiter = 1
plt = True
iter = 30
expensive_state = 'y' #either 'y' or 'x'
# Forw_Back_m0n1.alg_m0n1(gwg,mdp,iter,T,beta,cost,plt,pltiter)
# m0n1_moveobstacle.alg_m0n1_moveobstacle(gwg,mdp,obstacle_mdp,obstaclemovestates, iter,T,beta,cost,moveobstacles,plt,pltiter)
m0n0_moveobstacle.alg_m0n0_moveobstacle(gwg,mdp,obstacle_mdp,obstaclemovestates, iter,T,beta,cost,moveobstacles,plt,pltiter)