from typing import Any

from torch import Tensor


class HelperClass:
    """<description>.

    Args:
        param2 (Any): <description>.
    """

    def __init__(self, param2: Any):
        self.param2 = param2

    def method(self, param1: Tensor) -> Tensor:
        """
        <description>
        Args:
            param1 (Tensor): <description>.

        Returns:
            Tensor: <description>.
        """
        raise NotImplementedError()


def helper_function(
    param1: Tensor,
    param2: Any,
) -> Tensor:
    """
    <description>

    Args:
        param1 (Tensor): <description>.
        param2 (Any): <description>.

    Returns:
        Tensor: <description>.
    """
    raise NotImplementedError()
