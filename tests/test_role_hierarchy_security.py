"""Regression tests for the role-hierarchy security contract.

Maps one test group to each requirement in the task statement:

1. inheritance is only ever explicit (no dotted-name shortcuts);
2. cycles are found across the whole graph and rejected before expansion;
3. expansion depth is bounded; any over-long chain fails closed;
4. effective permissions are the union inside the role closure;
5. role names (and permission-table keys) normalize before comparison;
6. batch subjects are independent;
7. repeated expansion is deterministic and cache-mutation safe.
"""

import pytest

from rolehier.expand import (
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


# --- 1. explicit declaration only ---------------------------------------

def test_dotted_name_does_not_create_implicit_inheritance():
    roles = {"admin": set(), "admin.audit": {"staff"}, "staff": set()}
    assert closure(roles, "admin.audit", 5) == {"admin.audit", "staff"}
    assert check_declared("admin.audit", "admin", roles) is False
    assert check_declared("admin.audit", "staff", roles) is True


def test_only_declared_edges_are_walked():
    roles = {"a": {"b"}, "b": set(), "c": set()}
    assert closure(roles, "a", 5) == {"a", "b"}


def test_check_declared_requires_edge_and_normalizes():
    roles = {"Admin": {" Staff "}}
    assert check_declared(" admin ", "staff", roles) is True
    assert check_declared("admin", "nobody", roles) is False
    assert check_declared("ghost", "staff", roles) is False
    assert check_declared("admin", "admin", roles) is False
    # Coexistence without an edge is not a declaration.
    assert check_declared("staff", "admin", {"admin": set(), "staff": set()}) is False


# --- 2. cycles -----------------------------------------------------------

def test_detect_cycle_returns_cycle_members_in_order():
    roles = {"a": {"b"}, "b": {"c"}, "c": {"a"}, "d": {"a"}}
    assert detect_cycle(roles) == ["a", "b", "c"]


def test_self_loop_is_a_cycle():
    assert detect_cycle({"a": {"a"}}) == ["a"]


def test_acyclic_graph_reports_no_cycle():
    assert detect_cycle({"a": {"b"}, "b": {"c"}, "c": set()}) == []


def test_detect_cycle_is_case_insensitive():
    assert set(detect_cycle({"A": {"b"}, "b": {"a"}})) == {"a", "b"}


def test_closure_rejects_whole_graph_cycles_before_expansion():
    with pytest.raises(CycleError):
        closure({"a": {"b"}, "b": {"a"}}, "a", 5)
    # A root outside the cycle is refused too: the graph is validated whole.
    roles = {"innocent": set(), "a": {"b"}, "b": {"a"}}
    with pytest.raises(CycleError):
        closure(roles, "innocent", 5)


def test_role_batch_refuses_cyclic_graph():
    with pytest.raises(CycleError):
        role_batch(
            {"u": "a"},
            {"a": {"b"}, "b": {"a"}},
            {"a": {"x"}},
            {"max_depth": 5},
        )


# --- 3. depth limit ------------------------------------------------------

def test_depth_zero_keeps_only_root_and_refuses_edges():
    assert closure({"a": set()}, "a", 0) == {"a"}
    with pytest.raises(DepthExceeded):
        closure({"a": {"b"}, "b": set()}, "a", 0)


def test_chain_at_exactly_the_limit_is_allowed():
    roles = {"a": {"b"}, "b": {"c"}, "c": set()}
    assert closure(roles, "a", 2) == {"a", "b", "c"}


def test_chain_beyond_the_limit_is_refused():
    roles = {"a": {"b"}, "b": {"c"}, "c": {"d"}, "d": set()}
    with pytest.raises(DepthExceeded):
        closure(roles, "a", 2)


def test_negative_depth_limit_is_refused():
    with pytest.raises(DepthExceeded):
        closure({"a": set()}, "a", -1)


def test_overdeep_branch_through_shared_role_is_always_refused():
    # root -> jia -> bing -> ding -> wu is four edges; ding is also
    # reachable in two edges through yi. Whichever branch is visited first,
    # the four-edge chain must be discovered and refused.
    def graph(root_parents):
        return {
            "root": set(root_parents),
            "jia": {"bing"},
            "bing": {"ding"},
            "yi": {"ding"},
            "ding": {"wu"},
            "wu": set(),
        }

    for order in (("jia", "yi"), ("yi", "jia")):
        with pytest.raises(DepthExceeded):
            closure(graph(order), "root", 3)

    # A limit covering the longest chain admits the complete closure.
    assert closure(graph(("jia", "yi")), "root", 4) == {
        "root", "jia", "yi", "bing", "ding", "wu",
    }


def test_validate_roles_flags_depth_and_cycles():
    roles = {"a": {"b"}, "b": {"c"}, "c": set()}
    issues = validate_roles(roles, {"max_depth": 1})
    assert any("depth exceeded" in issue for issue in issues)
    assert validate_roles(roles, {"max_depth": 3}) == []
    cycle_issues = validate_roles({"a": {"b"}, "b": {"a"}}, {"max_depth": 3})
    assert any("cycle" in issue for issue in cycle_issues)


# --- 4. union of permissions inside the closure --------------------------

def test_permissions_are_unioned_not_intersected():
    roles = {"admin": {"staff"}, "staff": set()}
    perms = {
        "admin": {"user.delete", "user.read"},
        "staff": {"user.read", "ticket.reply"},
    }
    assert permissions_of(roles, perms, "admin", 5) == {
        "user.delete", "user.read", "ticket.reply",
    }


def test_unrelated_roles_never_contribute_permissions():
    roles = {"staff": set(), "admin": set()}
    perms = {"staff": {"user.read"}, "admin": {"user.delete"}}
    assert permissions_of(roles, perms, "staff", 5) == {"user.read"}
    assert permissions_of(
        {"admin": {"staff"}, "staff": set()},
        {"admin": {"user.delete"}, "staff": {"user.read"}, "other": {"x"}},
        "admin",
        5,
    ) == {"user.delete", "user.read"}


def test_unknown_and_blank_role_grant_nothing():
    roles = {"admin": {"staff"}, "staff": set()}
    perms = {"admin": {"user.delete"}, "staff": {"user.read"}}
    assert permissions_of(roles, perms, "stranger", 5) == set()
    assert permissions_of(roles, perms, "", 5) == set()
    assert permissions_of(roles, perms, "   ", 5) == set()


def test_blank_role_closure_is_rejected_but_permissions_fail_closed():
    with pytest.raises(RoleHierarchyError):
        closure({"a": set()}, "  ", 3)
    assert permissions_of({"a": set()}, {"a": {"x"}}, "  ", 3) == set()


def test_diamond_inheritance_unions_without_duplication():
    roles = {
        "top": {"left", "right"},
        "left": {"base"},
        "right": {"base"},
        "base": set(),
    }
    perms = {"top": {"t"}, "left": {"l"}, "right": {"r"}, "base": {"b"}}
    assert permissions_of(roles, perms, "top", 5) == {"t", "l", "r", "b"}


# --- 5. name normalization ----------------------------------------------

def test_normalize_role_strips_and_lowercases():
    assert normalize_role("  AdMiN ") == "admin"
    with pytest.raises(TypeError):
        normalize_role(123)  # type: ignore[arg-type]


def test_parse_roles_merges_case_and_whitespace_variants():
    assert parse_roles({"Admin": ["Staff"], "staff": []}) == {
        "admin": {"staff"},
        "staff": set(),
    }
    assert parse_roles({" A ": ["b", " B "], "b": []}) == {
        "a": {"b"},
        "b": set(),
    }


def test_parse_string_form_normalizes_names():
    assert parse_roles("Admin = Staff\nstaff =\n") == {
        "admin": {"staff"},
        "staff": set(),
    }


def test_closure_treats_case_variants_as_one_role():
    roles = {"admin": {"staff"}, "staff": set()}
    assert closure(roles, " ADMIN ", 5) == {"admin", "staff"}


def test_permission_table_keys_are_normalized():
    roles = parse_roles({"Admin": ["Staff"], "staff": []})
    perms = {"Admin": {"user.delete"}, " Staff ": {"user.read"}}
    assert permissions_of(roles, perms, "ADMIN", 5) == {
        "user.delete", "user.read",
    }


def test_blank_role_name_in_declaration_is_rejected():
    with pytest.raises(RoleHierarchyError):
        parse_roles({"   ": ["staff"]})


# --- 6. batch independence ----------------------------------------------

def test_batch_uses_each_subject_own_closure():
    roles = {"admin": {"staff"}, "staff": set()}
    perms = {"admin": {"user.delete"}, "staff": {"user.read"}}
    result = role_batch(
        {"alice": "admin", "bob": "staff", "carol": "staff"},
        roles,
        perms,
        {"max_depth": 5},
    )
    assert result["alice"] == {"user.delete", "user.read"}
    assert result["bob"] == {"user.read"}
    assert result["carol"] == {"user.read"}


def test_batch_results_are_independent_mutable_sets():
    result = role_batch(
        {"u1": "a", "u2": "b"},
        {"a": {"b"}, "b": set()},
        {"a": {"x"}, "b": {"y"}},
        {"max_depth": 5},
    )
    result["u1"].add("poison")
    assert "poison" not in result["u2"]


def test_batch_order_does_not_change_results():
    roles = {"admin": {"staff"}, "staff": set()}
    perms = {"admin": {"user.delete"}, "staff": {"user.read"}}
    first = role_batch(
        {"alice": "admin", "bob": "staff"}, roles, perms, {"max_depth": 5}
    )
    second = role_batch(
        {"bob": "staff", "alice": "admin"}, roles, perms, {"max_depth": 5}
    )
    assert first == second


def test_batch_accepts_pair_sequences():
    result = role_batch(
        [("u1", "a"), ("u2", "b")],
        {"a": {"b"}, "b": set()},
        {"a": {"x"}, "b": {"y"}},
        {"max_depth": 5},
    )
    assert result == {"u1": {"x", "y"}, "u2": {"y"}}


# --- 7. determinism and cache safety -------------------------------------

def test_repeated_expansion_is_identical_but_distinct():
    roles = {"a": {"b", "c"}, "b": {"d"}, "c": {"d"}, "d": set()}
    first = closure(roles, "a", 5)
    second = closure(roles, "a", 5)
    assert first == second == {"a", "b", "c", "d"}
    assert first is not second


def test_declaration_mutation_is_observed_without_stale_cache():
    roles = {"admin": {"staff"}, "staff": set()}
    assert closure(roles, "admin", 5) == {"admin", "staff"}
    roles["admin"].add("auditor")
    roles["auditor"] = set()
    assert closure(roles, "admin", 5) == {"admin", "staff", "auditor"}


def test_id_reuse_cannot_serve_stale_expansion():
    def build():
        return {"a": {"b"}, "b": set()}

    first_graph = build()
    first = closure(first_graph, "a", 5)
    del first_graph
    second_graph = build()
    assert closure(second_graph, "a", 5) == first
    assert closure({"a": set()}, "a", 5) == {"a"}


def test_returned_set_does_not_alias_inputs_or_cache():
    roles = {"a": {"b"}, "b": set()}
    expanded = closure(roles, "a", 5)
    expanded.add("poison")
    assert roles == {"a": {"b"}, "b": set()}
    assert closure(roles, "a", 5) == {"a", "b"}
