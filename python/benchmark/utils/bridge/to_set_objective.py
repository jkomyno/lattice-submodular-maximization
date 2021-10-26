from typing import AbstractSet
from ...objective import Objective
from ...set_objective import SetObjective
from .to_integer_lattice import to_integer_lattice


def to_set_objective(f: Objective) -> SetObjective:
    """
    Convert an integer lattice submodular function to a set submodular function
    via ground set expansion.
    """
    class SetObjectiveImpl(SetObjective):
        def __init__(self):
            super().__init__(ground_set=f.V, B=f.B)

        def value(self, S: AbstractSet[int]) -> int:
            """
            Value oracle for the submodular problem.
            :param S: subset of the ground set
            :return: value oracle for S in the submodular problem
            """
            x = to_integer_lattice(self, S)
            return f.value(x)

    f_prime = SetObjectiveImpl()

    return f_prime
