from dataclasses import dataclass


@dataclass(frozen=True)
class LearningRate:
    """
        Wrapper around the learning rate hyper-parameter. It has to
        be in the interval ]0,1[ to avoid the gradient explosion problem.
    """
    value: float

    def __post_init__(self):
        if not 0 < self.value < 1:
            raise ValueError("Learning rate must be on the interval ]0,1[.")
