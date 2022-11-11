from typing import Hashable, Protocol


class Data(Protocol, Hashable):
    """
        Base class to represent Data in the Any2Vec model. The
        only requirement for is to be Hashable (needed to create
        OneHotEncodings
    """
    pass
