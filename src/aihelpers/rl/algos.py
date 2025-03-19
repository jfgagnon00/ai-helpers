import numpy as np

from abc import ABC, abstractmethod, abstractproperty
from copy import deepcopy
from sympy import solve, Symbol, Add, Eq


class TransitionsAdapter(ABC):
    """
    Permet de decoupler le code de transisions des algos.
    Les algos ci-bas s'attendend a avoir un adapter pour fonctionner.
    """
    def equiprobable_policy(self):
        "Retourne une politique equiprobable (meme probalitites pour toutes les actions)"
        return np.ones((self.numStates, self.numActions.max())) / \
            self.numActions[..., np.newaxis]

    def random_choice_policy(self):
        """
        Retourne une politique random (chaque action est aleatoire mais
        1 seule action possible par etat)
        """
        s = self.numStates
        a = self.numActions.max()
        return np.eye(a)[ np.random.choice(a, size=(s)) ]

    @abstractproperty
    def numStates(self):
        "A implementer. Retourne nombre total d'etats"
        pass

    @abstractproperty
    def numActions(self):
        """
        A implementer. Retourne tableau numpy ou chaque element (etat)
        est le nombre d'actions.
        """
        pass

    @abstractmethod
    def states(self):
        "Retourne iterateur sur les etats. Doit retourner un index et non l'etat lui-meme."
        pass

    @abstractmethod
    def actions(self, s):
        """
        Retourne iterateur sur les actions pour l'etat s.
        Doit retourner un index et non l'action elle-meme.
        """
        pass

    @abstractmethod
    def transitions(self, s, a):
        """
        Retourne iterateur sur les tramsotopms pour le tuple (etat, action).
        Doit retourner un tupe (prochain etat, reward, probabilite)
        """
        pass

    def stateName(self, s):
        """
        Utilitaire pour convertir etat en string. A overider au besoin.
        Parametres:
            s: index de l'etat

        Retour:
            string representant l'etat s
        """
        return str(s)

    def actionName(self, s, a):
        """
        Utilitaire pour convertir action en string. A overider au besoin.
        Parametres:
            s: index de l'etat
            a: index de l'action

        Retour:
            string representant l'action s
        """
        return str(a)


def q_from_v(V, gamma, adapter):
    """
    Utilitaire pour obtenir Q a partir de V.
    Parametres:
        V      : etats valeur
        gamma  : facteur de discount
        adapter: instance de TransitionsAdapter

    Retour:
        Q
    """
    Q = np.zeros((adapter.numStates, adapter.numActions.max()))

    for s, terminal in adapter.states():
        if terminal:
            continue

        for a in adapter.actions(s):
            for s_prime, r, p_tr in adapter.transitions(s, a):
                Q[s, a] += (r + gamma * V[s_prime]) * p_tr

    return Q

def update_state_value(P, V, gamma, adapter):
    """
    Utilitaire pour obtenir V a partir de P.
    Parametres:
        P      : policy (probablites, shape => (num etats, num actions))
        V      : etats valeur
        gamma  : facteur de discount
        adapter: instance de TransitionsAdapter

    Retour:
        V et delta maximal
    """
    V_updated = deepcopy(V)
    delta = 0

    for s, terminal in adapter.states():
        if terminal:
            continue

        v = 0
        for a in adapter.actions(s):

            v_tr = 0
            for s_prime, r, p_tr in adapter.transitions(s, a):
                v_tr += (r + gamma * V[s_prime]) * p_tr

            v += v_tr * P[s, a]

        V_updated[s] = v
        delta = max(delta, abs(v - V[s]))

    return V_updated, delta

def policy_evaluate(P,
                    theta,
                    gamma,
                    adapter,
                    max_iteration=-1,
                    V_init=None):
    """
    Premiere etate de l'algo iterations strategie
    Parametres:
        P            : policy (probablites, shape => (num etats, num actions))
        theta        : threshold d'arret
        gamma        : facteur de discount
        adapter      : instance de TransitionsAdapter
        max_iteration: nombre maximal iterations (utile pour examiner valeurs intermediaires)
        V_init       : None => algo initialize a 0 sytematiquement, not None => valeurs a utiliser pour initializer

    Retour:
        V
    """
    V = np.zeros(adapter.numStates) if V_init is None else deepcopy(V_init)
    k = 0
    while True:
        V, delta = update_state_value(P, V, gamma, adapter)

        k += 1
        if delta < theta or (k >= max_iteration and max_iteration > 0):
            break

    return V

def policy_improve(P, V, gamma, adapter, output_q=False):
    """
    Deuxieme etate de l'algo iterations strategie
    Parametres:
        P       : policy (probablites, shape => (num etats, num actions))
        V       : etats valeur
        gamma   : facteur de discount
        adapter : instance de TransitionsAdapter
        output_q: True => ajouter Q au retour

    Retour:
        Tuple (P ameliore, stable) ou stable est un boolean
    """
    numActions = adapter.numActions
    numActionsMax = numActions.max()

    P_prime = np.empty((adapter.numStates, numActionsMax))
    Q = np.zeros_like(P_prime)
    stable = True

    for s, terminal in adapter.states():
        if terminal:
            P_prime[s] = np.zeros(numActionsMax)
            continue

        old_action = np.argmax(P[s, :numActions[s]])

        for a in adapter.actions(s):
            for s_prime, r, p_tr in adapter.transitions(s, a):
                Q[s, a] += (r + gamma * V[s_prime]) * p_tr

        new_action = np.argmax(Q[s, :numActions[s]])

        if np.argmin(Q[s, :numActions[s]]) == new_action:
            # quelques fois, les valeurs de Q sont egales
            # donc si min == max alors pas de changment
            new_action = old_action

        P_prime[s] = np.eye(numActionsMax)[new_action]

        stable = stable and (old_action == new_action)

    if output_q:
        return P_prime, stable, Q
    else:
        return P_prime, stable

def policy_iterate(P, theta, gamma, adapter):
    """
    Algo iterations strategie complet
    Parametres:
        P      : policy (probablites, shape => (num etats, num actions))
        theta  : threshold d'arret pour evaluation
        gamma  : facteur de discount
        adapter: instance de TransitionsAdapter

    Retour:
        Tuple (P optimal, V optimal)
    """
    while True:
        V = policy_evaluate(P, theta, gamma, adapter)
        P, stable = policy_improve(P, V, gamma, adapter)

        if stable:
            break

    return P, V

def solve_v_from_policy(P, gamma, adapter):
    V = [Symbol( f"v_{adapter.stateName(s)}") for s, _ in adapter.states()]

    equations = []
    for s, _ in adapter.states():
        a = np.argmax(P[s])

        expr = []
        for s_prime, r, p  in adapter.transitions(s, a):
            expr.append( p * (r + gamma * V[s_prime]) )

        eq = Eq(V[s], Add(*expr))
        equations.append(eq)

    solution = solve(equations, V)
    V = [float(solution[v]) for v in V]

    return np.array(V)

def value_iterate(theta, gamma, adapter):
    """
    Algo iteration valeurs
    Parametres:
        theta  : threshold d'arret pour evaluation
        gamma  : facteur de discount
        adapter: instance de TransitionsAdapter

    Retour:
        Tuple (V optimal, Q optimal)
    """
    numActions = adapter.numActions
    numActionsMax = numActions.max()

    # initialize la fonction valeurs etats
    V = np.zeros(adapter.numStates)

    while True:
        # fonction valeurs etats temporaire
        # pour faire les calculs intermediaires de mise a jour
        V_prime = np.empty_like(V)

        # fonction valeurs actions calculee a partir de V courant,
        # de la table de transition et de gamma
        Q = np.zeros((adapter.numStates, numActionsMax))

        # parcourir tous les etats
        for s, terminal in adapter.states():
            if terminal:
                # etat cible DOIT etre == 0
                Q[s, :] = 0
            else:
                # parcourir toutes les actions de l'etat courant
                for a in adapter.actions(s):
                    # parcourir toutes les transitions de l'action courante
                    for s_prime, r, p_tr in adapter.transitions(s, a):
                        # mettre a jour la fonction valeurs actions
                        Q[s, a] += p_tr * (r + gamma * V[s_prime])

            # mettre a jour la fonction valeurs etats temporaire
            # avec la valeur maximale de Q
            V_prime[s] = Q[s].max()

        # critere de convergence
        delta = np.abs(V_prime - V).max()
        if delta < theta:
            # l'algorithme a converge, terminer
            break

        # mettre a jour la fonction valeur finale
        # pas besoin de copie explicite, V_prime sera recreee a la prochaine iteration
        V = V_prime

    return V, Q
