from dataclasses import dataclass, field


@dataclass(order=True) # Allows the pq sort properly
class Node:
    F: float # g + h
    g: float = field(compare=False)
    h: float = field(compare=False) # compare=False enforces that the variable should not be considered when sorting Nodes in pq
    state: dict = field(compare=False)
    parent: list = field(compare=False)

    def __post_init__(self):
        self.F = self.g + self.h