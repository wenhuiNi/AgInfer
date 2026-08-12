class AgInferError(Exception):
    """Base class for deterministic, user-facing compiler failures."""


class ValidationError(AgInferError):
    """The input is unsafe, malformed, or unsupported."""


class FormatError(AgInferError):
    """An AIM container is malformed or has failed integrity checks."""


class CompatibilityError(AgInferError):
    """The requested target is not compatible with the package."""
