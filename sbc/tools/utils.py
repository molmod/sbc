###########################################################################################
# Training script
# Authors: Sander Vandenhaute
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################


from __future__ import annotations  # necessary for type-guarding class methods
from typing import Sequence, Iterable, Optional

import torch


class PhaseTable:

    def __init__(self, phases: Sequence[str]):
        self.phases = phases

    def __len__(self) -> int:
        return len(self.phases)

    def __str__(self):
        return f"PhaseTable: {tuple(s for s in self.phases)}"

    def index_to_phase(self, index: int) -> int:
        return self.phases[index]

    def phase_to_index(self, phase: str) -> int:
        return self.phases.index(phase)

    def to_one_hot(self, phase: str) -> torch.Tensor:
        """
        Generates one-hot encoding with <num_classes> classes from <indices>
        :param indices: (N x 1) tensor
        :param num_classes: number of classes
        :param device: torch device
        :return: (N x num_classes) tensor
        """
        index = self.phase_to_index(phase)
        return torch.nn.functional.one_hot(torch.tensor(index), num_classes=len(self))


    @classmethod
    def from_phases(cls, phases: Iterable[Optional[str]]) -> PhaseTable:
        phases = [p for p in phases if p is not None]
        return cls(sorted(list(set(phases))))
