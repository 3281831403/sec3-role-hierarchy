"""Role hierarchy expansion and permission checks."""

from .expand import (
    CycleError,
    DepthExceeded,
    RoleHierarchyError,
    check_declared,
    closure,
    detect_cycle,
    normalize_role,
    parse_roles,
    permissions_of,
    role_batch,
    validate_roles,
)

__all__ = [
    "CycleError",
    "DepthExceeded",
    "RoleHierarchyError",
    "check_declared",
    "closure",
    "detect_cycle",
    "normalize_role",
    "parse_roles",
    "permissions_of",
    "role_batch",
    "validate_roles",
]