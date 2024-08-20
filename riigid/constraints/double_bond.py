import numpy as np

from riigid.constraints.constraint import Constraint


class Double_Bond(Constraint):
    """Two-atomic bond."""

    def __init__(self):
        pass

    def init_internally(self, start_structure):
        pass

    def check(self, structure, cutoff=1e-6):
        pass

    def apply(self, structure):
        pass
