import numpy as np

class UniformModel:
    def get_probabilities(self, _):
        return np.ones(4) / 4

class Model:
    def __init__(self):
        self._table = {}
        self._learning_rate = 0.2

    def _ensure_context(self, context):
        if context not in self._table:
            self._table[context] = np.zeros(4, dtype=float)
        return self._table[context]

    def _softmax(self, logits):
        shifted = logits - np.max(logits)
        exp_logits = np.exp(shifted)
        return exp_logits / np.sum(exp_logits)

    def get_probabilities(self, context):
        weights = self._ensure_context(context)
        return self._softmax(weights)
    
    def update(self, path):
        states = path.get_states()
        actions = path.get_actions()

        for state, action in zip(states, actions):
            regular, reversed_context = state.get_reversed_context()

            # Regular context
            weights = self._ensure_context(regular)
            probs = self._softmax(weights)
            gradient = probs.copy()
            gradient[action] -= 1
            self._table[regular] -= self._learning_rate * gradient

            # Reversed context
            weights_rev = self._ensure_context(reversed_context)
            probs_rev = self._softmax(weights_rev)
            gradient_rev = probs_rev.copy()
            gradient_rev[action] -= 1
            self._table[reversed_context] -= self._learning_rate * gradient_rev