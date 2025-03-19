import numpy as np

from collections import namedtuple
from .algos import TransitionsAdapter


# types utilitaires pour construire graphe de transistions
State = namedtuple("State", ["name", "terminal"], defaults=(None, False))
Action = namedtuple("Action", ["name", "transitions"])
Transition = namedtuple("Transition", ["name", "s_prime", "r", "p"])

class GraphAdapter(TransitionsAdapter):
    def __init__(self, transitions_graph):
        self._transitions = transitions_graph

        # conversion key en index
        self._states = [None] * len(transitions_graph)
        self._stateToIndex = {}
        for i, (s, a) in enumerate(transitions_graph.items()):
            self._states[i] = (s, a)
            self._stateToIndex[s] = i

    @property
    def numStates(self):
        return len(self._states)

    @property
    def numActions(self):
        return np.array([len(a) for _, a in self._states])

    def states(self):
        for i, (s, _) in enumerate(self._states):
            yield i, s.terminal

    def actions(self, s):
        numActions = len(self._states[s][1])
        for a in range(numActions):
            yield a

    def transitions(self, s, a):
        for tr in self._states[s][1][a].transitions:
            yield self._stateToIndex[tr.s_prime], tr.r, tr.p

    def stateName(self, s):
        return self._states[s][0].name

    def actionName(self, s, a):
        return self._states[s][1][a].name

    def print_policy(self, P):
        numActions = self.numActions
        for s in range(self.numStates):
            a = np.argmax(P[s, :numActions[s]])
            print(self.stateName(s), ":", self.actionName(s, a))

    def print_policy_probabilities(self, P):
        numActions = self.numActions
        for s in range(self.numStates):
            print(self.stateName(s), ":", P[s, :numActions[s]])
