from rolehier.expand import (
    closure,
    detect_cycle,
    parse_roles,
    permissions_of,
    role_batch,
)


def test_parse_roles_normalizes_and_collects():
    roles = parse_roles({"Admin": ["Staff"], "staff": []})
    assert roles == {"admin": {"staff"}, "staff": set()}


def test_single_level_inheritance():
    roles = {
        "admin": set(),
        "admin.audit": {"staff"},
        "staff": set(),
    }
    assert closure(roles, "admin.audit", 5) == {"admin.audit", "staff"}


def test_permissions_are_unioned():
    roles = {"admin": {"staff"}, "staff": set()}
    perms = {"admin": {"user.delete"}, "staff": {"user.read"}}
    assert permissions_of(roles, perms, "admin", 5) == {
        "user.delete",
        "user.read",
    }


def test_cycle_detection():
    roles = {"a": {"b"}, "b": {"c"}, "c": {"a"}}
    cycle = detect_cycle(roles)
    assert cycle
    assert set(cycle) == {"a", "b", "c"}


def test_batch_keeps_subjects_independent():
    roles = {"admin": {"staff"}, "staff": set()}
    perms = {"admin": {"user.delete"}, "staff": {"user.read"}}
    result = role_batch(
        {"alice": "admin", "bob": "staff"},
        roles,
        perms,
        {"max_depth": 5},
    )
    assert result["alice"] == {"user.delete", "user.read"}
    assert result["bob"] == {"user.read"}