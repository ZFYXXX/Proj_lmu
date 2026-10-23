"""Project-specific exceptions."""


class MLLMCompError(Exception):
    """Base error for platform failures."""


class ConfigError(MLLMCompError):
    """Raised when a config violates the platform contract."""


class RegistryError(MLLMCompError):
    """Raised when a requested capability is missing or inactive."""


class InvariantError(MLLMCompError):
    """Raised when state or operation invariants fail."""


class ManifestValidationError(MLLMCompError):
    """Raised when a manifest fails partial or final validation."""


class BackendIntegrationError(MLLMCompError):
    """Raised when an external framework bridge fails."""


class NotActivatedError(MLLMCompError, NotImplementedError):
    """Raised for skeleton modules that are not active capabilities yet."""
