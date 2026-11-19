# SPDX-License-Identifier: Apache-2.0
"""Errors a request client raises when a request does not succeed."""

# Standard
from typing import Any


class MPServerUnavailableError(TimeoutError, ConnectionError):
    """The server went away before it answered a request.

    It is also a ``TimeoutError`` because over ZMQ the same failure shows up
    as a timeout, and callers catch it as one.

    Args:
        code: The gRPC status code name, such as ``"UNAVAILABLE"``.
        details: The status message.
    """

    def __init__(self, code: str, details: str) -> None:
        super().__init__(f"LMCache server unavailable ({code}): {details}")
        self.code = code
        self.details = details

    def __reduce__(self) -> tuple[type, tuple[str, str]]:
        return type(self), (self.code, self.details)


class MPRequestError(RuntimeError):
    """The server failed a request.

    Args:
        code: The failure's code, such as ``"INTERNAL"`` on gRPC.
        details: The server's message.
    """

    def __init__(self, code: str, details: str) -> None:
        super().__init__(f"LMCache request failed ({code}): {details}")
        self.code = code
        self.details = details

    def __reduce__(self) -> tuple[type, tuple[str, str]]:
        return type(self), (self.code, self.details)


class MPInstanceNotRegisteredError(MPRequestError, ValueError):
    """The server has no registration for the calling worker instance.

    Raised when the instance never registered or was reaped after going
    silent. Retrying the request alone cannot succeed: the worker must
    register again. The server raises it from its transfer handlers (it is
    a ``ValueError`` there, as before); the ZMQ client raises it when the
    server answers with its ``code``.

    Args:
        details: The server's message.
    """

    code = "instance_not_registered"
    """The error code the server sends for this failure."""

    def __init__(self, details: str) -> None:
        super().__init__(type(self).code, details)

    def __reduce__(self) -> tuple[Any, ...]:
        return type(self), (self.details,)


# What waiting on a request raises when it did not succeed: MPRequestError if the
# server answered with a failure, a timeout if it never answered.
REQUEST_FAILURES = (TimeoutError, MPRequestError)
