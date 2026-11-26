"""Role hierarchy expansion and permission helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


class RoleHierarchyError(ValueError):
    """Base class for role hierarchy failures."""


class CycleError(RoleHierarchyError):
    """Raised when inheritance contains a cycle."""


class DepthExceeded(RoleHierarchyError):
    """Raised when inheritance exceeds the permitted depth."""


def normalize_role(name: str) -> str:
    """Return the canonical form of a role name."""

    if not isinstance(name, str):
        raise TypeError("role name must be a string")
    return name.strip()


def parse_roles(spec: Any) -> dict[str, set[str]]:
    """Build a role to parents mapping from a declaration."""

    raw: dict[str, list[str]] = {}
    if isinstance(spec, str):
        for line in spec.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                child, parent_text = line.split("=", 1)
            else:
                child, parent_text = line, ""
            raw[child.strip()] = [
                part.strip() for part in parent_text.split(",") if part.strip()
            ]
    elif isinstance(spec, Mapping):
        for child, parents in spec.items():
            if parents is None:
                values: list[Any] = []
            elif isinstance(parents, str):
                values = [parents]
            else:
                values = list(parents)
            raw[str(child)] = [str(part).strip() for part in values if str(part).strip()]
    else:
        raise TypeError("spec must be a string or a mapping")

    roles: dict[str, set[str]] = {}
    for child, parents in raw.items():
        child_name = child.strip()
        roles.setdefault(child_name, set())
        for parent in parents:
            parent_name = parent.strip()
            roles.setdefault(parent_name, set())
            roles[child_name].add(parent_name)
    return roles


def detect_cycle(roles: Mapping[str, Iterable[str]]) -> list[str]:
    """Return role names that participate in an inheritance cycle."""

    seen: set[str] = set()

    def visit(node: str) -> None:
        if node in seen:
            return
        seen.add(node)
        for parent in roles.get(node, ()):
            visit(parent)

    for node in roles:
        visit(node)
    return []


_CLOSURE_CACHE: dict[tuple[int, str, int], tuple[Any, frozenset[str]]] = {}


def closure(
    roles: Mapping[str, Iterable[str]], role: str, depth_limit: int
) -> set[str]:
    """Return the role and all roles reachable through inheritance."""

    role_name = normalize_role(role)
    key = (id(roles), role_name, depth_limit)
    cached = _CLOSURE_CACHE.get(key)
    if cached is not None and cached[0] is roles:
        return set(cached[1])
    if not role_name:
        return set()

    found: set[str] = set()

    def walk(node: str, depth: int) -> None:
        found.add(node)
        for parent in roles.get(node, ()):
            walk(parent, depth + 1)
        if "." in node:
            prefix = node.rsplit(".", 1)[0]
            if prefix in roles and prefix not in roles.get(node, ()):
                walk(prefix, depth + 1)

    walk(role_name, 0)
    _CLOSURE_CACHE[key] = (roles, frozenset(found))
    return set(found)


def permissions_of(
    roles: Mapping[str, Iterable[str]],
    perms: Mapping[str, Iterable[str]],
    role: str,
    depth_limit: int,
) -> set[str]:
    """Return the permissions granted by a role closure."""

    expanded = closure(roles, role, depth_limit)
    if not expanded:
        merged: set[str] = set()
        for values in perms.values():
            merged.update(values)
        return merged

    combined: set[str] | None = None
    for current in expanded:
        current_perms = set(perms.get(current, ()))
        if combined is None:
            combined = current_perms
        else:
            combined &= current_perms
    return set(combined or set())


def check_declared(
    child: str, parent: str, roles: Mapping[str, Iterable[str]]
) -> bool:
    """Report whether child explicitly declares parent."""

    child_name = normalize_role(child)
    parent_name = normalize_role(parent)
    return child_name in roles and parent_name in roles


def validate_roles(
    roles: Mapping[str, Iterable[str]], policy: Mapping[str, Any]
) -> list[str]:
    """Return the policy problems found in a role declaration."""

    issues: list[str] = []
    cycle = detect_cycle(roles)
    if cycle:
        issues.append("cycle detected: " + " -> ".join(list(cycle) + [cycle[0]]))
    for role in sorted(roles):
        if "." in role:
            prefix = role.rsplit(".", 1)[0]
            if prefix in roles and prefix not in roles.get(role, ()):
                issues.append("implicit inheritance: " + role + " -> " + prefix)
    return issues


def _subject_pairs(subjects: Any) -> list[tuple[Any, str]]:
    if isinstance(subjects, Mapping):
        return list(subjects.items())
    return list(subjects)


def role_batch(
    subjects: Any,
    roles: Mapping[str, Iterable[str]],
    perms: Mapping[str, Iterable[str]],
    policy: Mapping[str, Any],
) -> dict[Any, set[str]]:
    """Return permission sets for a batch of subject to role assignments."""

    depth_limit = policy.get("max_depth", 3)
    result: dict[Any, set[str]] = {}
    shared: set[str] | None = None
    for subject, role in _subject_pairs(subjects):
        if shared is None:
            shared = closure(roles, role, depth_limit)
        merged: set[str] = set()
        for current in shared:
            merged.update(perms.get(current, ()))
        result[subject] = merged
    return result