import mlrose_hiive as mlrose
import numpy as np

class CustomKnapsack(mlrose.Knapsack):
    def __init__(self, weights, values, max_weight_pct=0.35, target_weight_pct=0.3):
        super().__init__(weights, values, max_weight_pct)
        self.target_weight = np.ceil(np.sum(self.weights) * target_weight_pct)

    def evaluate(self, state):
        """Evaluate the fitness of a state vector.

        Parameters
        ----------
        state: array
            State array for evaluation. Must be the same length as the weights
            and values arrays.

        Returns
        -------
        fitness: float
            Value of fitness function.
        """

        if len(state) != len(self.weights):
            raise Exception(
                """The state array must be the same size as the"""
                + """ weight and values arrays."""
            )

        # Calculate total weight and value of knapsack
        total_weight = np.sum(state * self.weights)
        total_value = np.sum(state * self.values)

        # Allow for weight constraint
        if total_weight <= self._w:
            fitness = abs(total_value - self.target_weight)
        else:
            fitness = self._w
        return fitness
