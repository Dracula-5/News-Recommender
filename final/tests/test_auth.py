"""
Regression tests for two auth bugs fixed in this pass:

  1. verify_password() compared hashes with a plain `==`, which is not
     constant-time and leaks a timing side-channel.
  2. No endpoint actually checked the bearer token against the user_id in
     the request — any client could call /recommend, /feedback, or read
     /user/{id} for *any* user_id with no token at all, or with a token
     belonging to a completely different account (IDOR). Guest mode is
     intentionally tokenless and must keep working unauthenticated.
  3. There was no server-side logout — a token stayed valid until its
     30-day expiry regardless of client-side localStorage.clear().
"""
from __future__ import annotations

import hmac

from auth import hash_password, verify_password


def test_verify_password_uses_constant_time_comparison(monkeypatch):
    calls = []
    real_compare = hmac.compare_digest

    def spy(a, b):
        calls.append((a, b))
        return real_compare(a, b)

    monkeypatch.setattr(hmac, "compare_digest", spy)
    h = hash_password("correct-horse-battery-staple")
    verify_password("correct-horse-battery-staple", h)
    assert calls, "verify_password should call hmac.compare_digest, not `==`"


def test_verify_password_accepts_correct_and_rejects_wrong():
    h = hash_password("s3cret-passw0rd")
    assert verify_password("s3cret-passw0rd", h) is True
    assert verify_password("wrong-password", h) is False


def _signup(api_client, email, username, password="testpass123"):
    resp = api_client.post(
        "/auth/signup",
        json={"email": email, "username": username, "password": password},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_guest_requests_without_a_token_still_work(api_client, unique_user_id):
    # No Authorization header at all — must keep working (guest mode).
    resp = api_client.post("/recommend", json={"user_id": unique_user_id, "k": 8})
    assert resp.status_code == 200


def test_token_holder_can_act_as_themselves(api_client):
    account = _signup(api_client, "alice@example.com", "alice_test_user")
    headers = {"Authorization": f"Bearer {account['token']}"}
    resp = api_client.post(
        "/recommend",
        json={"user_id": account["user_id"], "k": 8},
        headers=headers,
    )
    assert resp.status_code == 200


def test_token_holder_cannot_act_as_a_different_user(api_client):
    account = _signup(api_client, "bob@example.com", "bob_test_user")
    headers = {"Authorization": f"Bearer {account['token']}"}

    resp = api_client.post(
        "/recommend",
        json={"user_id": "someone_elses_user_id", "k": 8},
        headers=headers,
    )
    assert resp.status_code == 403


def test_token_holder_cannot_read_a_different_users_profile(api_client):
    victim = _signup(api_client, "victim@example.com", "victim_test_user")
    attacker = _signup(api_client, "attacker@example.com", "attacker_test_user")
    headers = {"Authorization": f"Bearer {attacker['token']}"}

    resp = api_client.get(f"/user/{victim['user_id']}", headers=headers)
    assert resp.status_code == 403


def test_invalid_token_is_rejected(api_client, unique_user_id):
    headers = {"Authorization": "Bearer not-a-real-token"}
    resp = api_client.post(
        "/recommend",
        json={"user_id": unique_user_id, "k": 8},
        headers=headers,
    )
    assert resp.status_code == 401


def test_logout_revokes_the_token(api_client):
    account = _signup(api_client, "carol@example.com", "carol_test_user")
    headers = {"Authorization": f"Bearer {account['token']}"}

    ok = api_client.post(
        "/recommend", json={"user_id": account["user_id"], "k": 8}, headers=headers,
    )
    assert ok.status_code == 200

    logout = api_client.post("/auth/logout", headers=headers)
    assert logout.status_code == 200
    assert logout.json()["revoked"] is True

    after_logout = api_client.post(
        "/recommend", json={"user_id": account["user_id"], "k": 8}, headers=headers,
    )
    assert after_logout.status_code == 401
