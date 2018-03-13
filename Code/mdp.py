"""
Author information:
Guillermo A. Perez
Universite Libre de Bruxelles
gperezme@ulb.ac.be
"""

from nfa import NFA
import numpy as np


class MDP(NFA):
    def __init__(self, states, accepting_states, alphabet, transitions=[]):
        # we call the underlying NFA constructor but drop the probabilities
        trans = [(s, a, t) for s, a, t, p in transitions]
        super(MDP, self).__init__(states, accepting_states, alphabet, trans)
        # in addition to the NFA we need a probabilistic transition
        # function
        self._prob_cache = dict()
        for s, a, t, p in transitions:
            self._prob_cache[(s, a, t)] = p
        self._prepare_post_cache()

    def prob_delta(self, s, a, t):
        return self._prob_cache[(s, a, t)]

    # Not working as well as it should. don't know what's wrong.
    def sample(self, state, action):
        """Sample the next state according to the current state, the action, and
        the transition probability. """
        if action not in self.available(state):
            return None
        # N = len(self.post(state, action))
        prob = []
        for t in self.post(state, action):
            prob.append(self.prob_delta(state, action, t))

        next_state = np.random.choice(list(self.post(state, action)),
                                      1, p=prob)[0]
        # Note that only one element is chosen from the array, which is the
        # output by random.choice
        return next_state

    def set_prob_delta(self, s, a, t, p):
        self._prob_cache[(s, a, t)] = p

    def value_iteration(self, T, unknown_trans=set(),
                        epsilon=0.001, gamma=1):
        """Solving an MDP by value iteration"""
        U1 = dict([(s, 0) for s in self.states])
        U1.update([(s, 1) for s in self.accepting_states])
        R = dict([((s, a, t), 0) for s in self.states
                  for a in self.available(s)
                  for t in self.post(s, a)])
        # R.update({(s,a,t):1 for s in self.states for a in self.available(s) for t in self.post(s,a) if t in self.accepting_states})
        P = dict([((s, a, t), set()) for s in self.states
                  for a in self.available(s)
                  for t in self.post(s, a)])
        policy = dict([(s, set()) for s in self.states])
        for s in self.states:
            for a in self.available(s):
                for t in self.post(s, a):
                    P[(s, a, t)].add(self.prob_delta(s, a, t))
                    if (s, a, t) in unknown_trans:
                        R[s, a, t] = 1
                    # elif s not in self.accepting_states and \
                    #         t in self.accepting_states:
                    #     R[s, a, t] = 1
        t = T
        while t > 0:
            U = U1.copy()
            delta = 0
            for s in self.states:
                U1[s] = max([sum([p * (gamma * U[s1] + R[s, a, s1])
                                  for s1 in self.post(s, a)
                                  for p in P[(s, a, s1)]])]
                            for a in self.available(s))[0]
                delta = max(delta, abs(U1[s] - U[s]))
            # if delta < epsilon * (1 - gamma) / gamma:
            #     break
            t = t - 1
        for s in self.states:
            Vmax = dict()
            for a in self.available(s):
                Vmax[a] = [sum([p * (gamma*U[s1] + R[s, a, s1])
                                for s1 in self.post(s, a)
                                for p in P[(s, a, s1)]])][0]
            maxV = max(Vmax.values())
            for a in Vmax.keys():
                if Vmax[a] == maxV:
                    policy[s].add(a)
        return U, policy

    def E_step_value_iter(self, epsilon=0.00001):
        """
        infinite time horizon
        Value iteration: Vstate[s] the maximal probability of hitting the
        target AEC within T steps.
        """
        policyT = dict([])
        Vstate1 = dict([])
        Win = self.accepting_states
        NAEC = set(self.states) - Win
        Vstate1.update({s: 1 for s in list(Win)})
        Vstate1.update({s: 0 for s in list(NAEC)})
        policyT.update({s: self.available(s) for s in list(Win)})
        e = 1
        Q = dict([])
        while e > epsilon:
            Vstate = Vstate1.copy()
            for s in set(self.states) - Win:
                acts = self.available(s)
                optimal = 0
                act = None
                for a in self.available(s):
                    Q[(s, a)] = sum([self.prob_delta(s, a, next_s) *
                                     Vstate[next_s]
                                     for next_s in self.post(s, a)])
                    if Q[(s, a)] >= optimal:
                        optimal = Q[(s, a)]
                        act = a
                    else:
                        pass
                acts = set([])
                for act in self.available(s):
                    if Q[(s, act)] == optimal:
                        acts.add(act)
                Vstate1[s] = optimal
                policyT[s] = acts
                e = abs(max([Vstate1[s] -
                             Vstate[s] for s in self.states]))  # the abs error
                # print "iteration: {} and the state
                # value is {}".format(t, Vstate1)
        return Vstate1, policyT
