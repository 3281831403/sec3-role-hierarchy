"""Role hierarchy expansion and permission helpers.

Security contract enforced by this module:

* Inheritance edges exist only when explicitly declared in the role
  specification; a dotted role name such as ``admin.audit`` never implies
  an edge to ``admin``.
* Cycles are detected across the whole declaration and rejected before any
  expansion happens, no matter where the traversal starts.
* Expansion is bounded by a depth limit. If *any* declared chain from the
  queried role is longer than the limit, :class:`DepthExceeded` is raised;
  a role that also has a shorter route is not allowed to hide the
  over-deep chain. The decision is independent of parent-set iteration
  order.
* Effective permissions are the union of the permissions of exactly the
  roles in the subject's closure; nothing outside the closure contributes.
* Role names are normalized (surrounding whitespace removed and names
  lowercased) before any comparison, so ``" Admin "`` and ``"admin"`` are
  one role. Permission-table keys are normalized the same way.
* Batch subjects are expanded independently with freshly allocated sets,
  so one subject's closure can never leak into another subject's result.
* Expansion is a pure function of its inputs: there is no process-global
  cache, so mutating a declaration and expanding again is always observed
  and repeated expansion of one role yields the same result.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from typing import Any


class RoleHierarchyError(ValueError):
    """Base class for role hierarchy failures."""


class CycleError(RoleHierarchyError):
    """Raised when inheritance contains a cycle."""


class DepthExceeded(RoleHierarchyError):
    """Raised when inheritance exceeds the permitted depth."""


def normalize_role(name: str) -> str:
    """Return the canonical form of a role name.

    Surrounding whitespace is removed and the remaining name is lowercased
    so that case or padding differences can never split one role into two.
    Only leading and trailing whitespace is removed; whitespace inside a
    name is significant.
    """

    if not isinstance(name, str):
        raise TypeError("role name must be a string")
    return name.strip().lower()


def _parent_values(parents: Any) -> list[Any]:
    """Return declared parents from one declaration value.

    ``None`` means "no parents" and a bare string is a single role name
    rather than an iterable of characters.
    """

    if parents is None:
        return []
    if isinstance(parents, str):
        return [parents]
    return list(parents)


def _normalized_graph(roles: Mapping[str, Iterable[str]]) -> dict[str, set[str]]:
    """Rebuild the child to parents mapping with every name normalized.

    The input mapping is never mutated and only edges that were actually
    declared are copied; normalization never invents an edge. Parent roles
    are registered as known roles (with no parents of their own) so graph
    walks and cycle scans can resolve them.
    """

    graph: dict[str, set[str]] = {}
    for child, parents in roles.items():
        child_name = normalize_role(child)
        declared = graph.setdefault(child_name, set())
        for parent in _parent_values(parents):
            parent_name = normalize_role(parent)
            if not parent_name:
                # A blank name cannot identify a role; ignore the phantom
                # edge instead of walking an empty node.
                continue
            graph.setdefault(parent_name, set())
            declared.add(parent_name)
    return graph


def parse_roles(spec: Any) -> dict[str, set[str]]:
    """Build a normalized role to parents mapping from a declaration.

    ``spec`` is either a mapping ``{child: parent-or-parents}`` or a text
    declaration with one ``child = p1, p2`` line per role (``#`` comments
    and blank lines are ignored). Names differing only in case or
    surrounding whitespace merge into a single role. A blank child name is
    rejected with :class:`RoleHierarchyError`.
    """

    pairs: list[tuple[Any, list[Any]]] = []
    if isinstance(spec, str):
        for line in spec.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                child, parent_text = line.split("=", 1)
            else:
                child, parent_text = line, ""
            parents = [
                part.strip() for part in parent_text.split(",") if part.strip()
            ]
            pairs.append((child, parents))
    elif isinstance(spec, Mapping):
        for child, parents in spec.items():
            values = _parent_values(parents)
            parent_names = [
                str(value) for value in values if str(value).strip()
            ]
            pairs.append((str(child), parent_names))
    else:
        raise TypeError("spec must be a string or a mapping")

    roles: dict[str, set[str]] = {}
    for child, parents in pairs:
        child_name = normalize_role(child)
        if not child_name:
            raise RoleHierarchyError("role name must not be empty")
        declared = roles.setdefault(child_name, set())
        for parent in parents:
            parent_name = normalize_role(parent)
            if not parent_name:
                continue
            roles.setdefault(parent_name, set())
            declared.add(parent_name)
    return roles


# DFS color markers for cycle detection.
_WHITE, _GRAY, _BLACK = 0, 1, 2


def detect_cycle(roles: Mapping[str, Iterable[str]]) -> list[str]:
    """Return the normalized names of roles participating in a cycle.

    An empty list means the declaration is acyclic. The whole graph is
    inspected, including components unrelated to any particular starting
    role, so a cycle cannot hide behind an innocent root. When a cycle is
    present its members are returned as a deterministically ordered list.
    """

    graph = _normalized_graph(roles)
    color: dict[str, int] = {node: _WHITE for node in graph}
    cycle_members: set[str] = set()

    for start in sorted(graph):
        if color[start] != _WHITE:
            continue
        color[start] = _GRAY
        path: list[str] = [start]
        # Each frame pairs a node with an iterator over its parents, which
        # keeps the traversal iterative (no recursion-depth blow-up).
        stack: list[tuple[str, Any]] = [
            (start, iter(sorted(graph.get(start, ()))))
        ]
        while stack:
            node, parents = stack[-1]
            try:
                nxt = next(parents)
            except StopIteration:
                color[node] = _BLACK
                stack.pop()
                path.pop()
                continue

            if nxt not in color:
                color[nxt] = _WHITE
            if color[nxt] == _GRAY:
                # Back edge: the current path segment from nxt to the top
                # of the stack is one inheritance cycle.
                start_index = path.index(nxt)
                cycle_members.update(path[start_index:])
            elif color[nxt] == _WHITE:
                color[nxt] = _GRAY
                path.append(nxt)
                stack.append((nxt, iter(sorted(graph.get(nxt, ())))))
            # A BLACK neighbor is fully explored; nothing to do.

    return sorted(cycle_members)


def _validate_depth_limit(depth_limit: int) -> None:
    """Reject depth limits that cannot bound an expansion."""

    if not isinstance(depth_limit, int) or isinstance(depth_limit, bool):
        raise TypeError("depth_limit must be an integer")
    if depth_limit < 0:
        raise DepthExceeded("depth_limit must be non-negative")


def _ensure_acyclic(graph: Mapping[str, Iterable[str]]) -> None:
    """Validate the whole graph and raise before any expansion."""

    cycle = detect_cycle(graph)
    if cycle:
        raise CycleError(
            "cycle detected: " + " -> ".join(cycle + [cycle[0]])
        )


def closure(
    roles: Mapping[str, Iterable[str]], role: str, depth_limit: int
) -> set[str]:
    """Return the role and every role reachable through declared inheritance.

    The declaration is normalized without mutating the caller's mapping and
    checked for cycles across the whole graph before traversal starts.

    Depth is counted in crossed inheritance edges from ``role`` (the root
    sits at depth 0). The bound is enforced against the *longest* declared
    route to each role: if any chain would cross more than ``depth_limit``
    edges, :class:`DepthExceeded` is raised. This fail-closed rule is
    independent of parent-set iteration order and prevents a shared role
    reached first along a short branch from hiding an over-deep branch.

    A fresh set is returned on every call; no result is cached globally.
    """

    _validate_depth_limit(depth_limit)
    role_name = normalize_role(role)
    if not role_name:
        raise RoleHierarchyError("role name must not be empty")

    graph = _normalized_graph(roles)
    # Reject cycles before expanding anything, including cycles outside the
    # component reachable from role_name.
    _ensure_acyclic(graph)

    # best[node] is the deepest known distance from the root. A deeper
    # arrival re-enqueues the node so its outgoing edges are re-checked at
    # the greater depth; this is what makes over-deep routes through shared
    # nodes observable regardless of traversal order.
    best: dict[str, int] = {role_name: 0}
    queue: deque[str] = deque([role_name])

    while queue:
        node = queue.popleft()
        node_depth = best[node]
        for parent in sorted(graph.get(node, ())):
            next_depth = node_depth + 1
            # Refuse before crossing an edge past the limit, even when the
            # parent was already reached along a shorter route.
            if next_depth > depth_limit:
                raise DepthExceeded(
                    "inheritance depth %d exceeds limit %d at %s -> %s"
                    % (next_depth, depth_limit, node, parent)
                )
            if next_depth > best.get(parent, -1):
                best[parent] = next_depth
                queue.append(parent)

    return set(best)


def _normalized_permissions(
    perms: Mapping[str, Iterable[str]],
) -> dict[str, set[Any]]:
    """Normalize permission-table role keys, merging name variants.

    Permission values are copied verbatim: permission strings are not role
    names and their casing may be significant.
    """

    table: dict[str, set[Any]] = {}
    for role, values in perms.items():
        role_name = normalize_role(role)
        if not role_name:
            continue
        bucket = table.setdefault(role_name, set())
        bucket.update(_parent_values(values))
    return table


def permissions_of(
    roles: Mapping[str, Iterable[str]],
    perms: Mapping[str, Iterable[str]],
    role: str,
    depth_limit: int,
) -> set[str]:
    """Return the union of permissions granted by a role's closure.

    Only roles inside the closure contribute permissions, so an unrelated
    role's permissions can never leak in. An unknown or blank role closes
    with an empty permission set (fail closed) rather than granting
    anything.
    """

    _validate_depth_limit(depth_limit)
    role_name = normalize_role(role)
    if not role_name:
        return set()

    expanded = closure(roles, role_name, depth_limit)
    table = _normalized_permissions(perms)

    merged: set[str] = set()
    for current in expanded:
        merged.update(table.get(current, ()))
    return merged


def check_declared(
    child: str, parent: str, roles: Mapping[str, Iterable[str]]
) -> bool:
    """Report whether ``child`` explicitly declares ``parent``.

    Both names and the declaration are normalized before comparison. Mere
    coexistence of the two role names, or a dotted-name prefix
    relationship, is not a declaration: the edge must be present.
    """

    child_name = normalize_role(child)
    parent_name = normalize_role(parent)
    if not child_name or not parent_name:
        return False
    graph = _normalized_graph(roles)
    return parent_name in graph.get(child_name, set())


def validate_roles(
    roles: Mapping[str, Iterable[str]], policy: Mapping[str, Any]
) -> list[str]:
    """Return the cycle and depth problems found in a role declaration.

    The returned strings are human-readable issue descriptions; an empty
    list means the declaration is acyclic and every role expands within
    ``policy["max_depth"]`` (default 3).
    """

    graph = _normalized_graph(roles)

    cycle = detect_cycle(graph)
    if cycle:
        # A cyclic graph cannot be safely expanded, so depth checks are
        # moot until the cycle is removed.
        return ["cycle detected: " + " -> ".join(cycle + [cycle[0]])]

    if isinstance(policy, Mapping):
        depth_limit = policy.get("max_depth", 3)
    else:
        depth_limit = 3
    _validate_depth_limit(depth_limit)

    issues: list[str] = []
    for role_name in sorted(graph):
        try:
            closure(graph, role_name, depth_limit)
        except DepthExceeded as exc:
            issues.append("depth exceeded for role %s: %s" % (role_name, exc))
    return issues


def _subject_pairs(subjects: Any) -> list[tuple[Any, str]]:
    """Normalize the subjects input into ``(subject, role)`` pairs."""

    if isinstance(subjects, Mapping):
        return list(subjects.items())
    return [tuple(pair) for pair in subjects]


def role_batch(
    subjects: Any,
    roles: Mapping[str, Iterable[str]],
    perms: Mapping[str, Iterable[str]],
    policy: Mapping[str, Any],
) -> dict[Any, set[str]]:
    """Return a permission set for every subject to role assignment.

    Each subject is expanded independently through
    :func:`permissions_of` and receives a freshly allocated set, so one
    subject's closure or result can never contaminate another subject.
    """

    if isinstance(policy, Mapping):
        depth_limit = policy.get("max_depth", 3)
    else:
        depth_limit = 3

    result: dict[Any, set[str]] = {}
    for subject, role in _subject_pairs(subjects):
        result[subject] = permissions_of(roles, perms, role, depth_limit)
    return result
