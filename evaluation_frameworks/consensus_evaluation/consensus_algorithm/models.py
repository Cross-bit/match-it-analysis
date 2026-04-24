
from dataclasses import dataclass


# The Vote class represents a single vote with a unique id and a value.
@dataclass
class Vote:
    id: int
    value: int

    def __eq__(self, other):
        # Two Vote objects are considered equal if they have the same id
        return isinstance(other, Vote) and self.id == other.id

    def __hash__(self):
        # The hash is computed from the id to ensure that Vote objects can be used in sets and dicts
        return hash(self.id)