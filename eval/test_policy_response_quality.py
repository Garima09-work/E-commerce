import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["MOCK_LLM"] = "1"
os.environ["USE_REAL_LLM"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from agent.graph import run_agent


def test_1_what_is_nykaas_return_policy():
    print("\n--- Test 1: 'What is Nykaa\'s return policy?' ---")
    res = run_agent("What is Nykaa's return policy?", return_state=True)

    route = res.get("route")
    ans = res["response"]["answer"]
    sources = res["response"]["sources"]
    v_status = res.get("verification_status")

    print(f"Route: {route}")
    print(f"Sources: {sources}")
    print(f"Verification: {v_status}")
    print(f"Answer: {ans}")

    assert route == "policy", f"Expected policy route, got {route}"
    assert "return_window.md" in sources, f"Expected return_window.md in sources, got {sources}"
    assert not ans.lower().startswith("safety reasons"), f"Answer should not start mid-sentence with 'safety reasons': {ans}"
    assert "15 days" in ans, f"Answer should mention 15 days: {ans}"
    assert "Customers may initiate a return request" in ans or "within 15 days" in ans, f"Answer missing main policy rule: {ans}"
    assert v_status == "PASS", f"Expected PASS verification, got {v_status}"
    print("[PASSED] Test 1 passed!")


def test_2_hi_what_is_the_return_policy():
    print("\n--- Test 2: 'Hi, what is the return policy?' ---")
    res = run_agent("Hi, what is the return policy?", return_state=True)

    route = res.get("route")
    ans = res["response"]["answer"]
    sources = res["response"]["sources"]
    v_status = res.get("verification_status")

    print(f"Route: {route}")
    print(f"Sources: {sources}")
    print(f"Verification: {v_status}")
    print(f"Answer: {ans}")

    assert route == "policy", f"Expected policy route (not conversational), got {route}"
    assert res.get("is_conversational") is not True, "Mixed query should not be flagged as purely conversational"
    assert "return_window.md" in sources, f"Expected return_window.md in sources, got {sources}"
    assert not ans.lower().startswith("safety reasons"), f"Answer must not start with fragment 'safety reasons': {ans}"
    assert ans.startswith("Customers may initiate") or "15 days" in ans, f"Answer should start with main policy rule: {ans}"
    assert "strictly non-returnable" in ans, f"Exclusion information should follow main rule: {ans}"
    assert v_status == "PASS", f"Expected PASS verification, got {v_status}"
    print("[PASSED] Test 2 passed!")


def test_3_can_i_return_an_opened_fragrance():
    print("\n--- Test 3: 'Can I return an opened fragrance?' ---")
    res = run_agent("Can I return an opened fragrance?", return_state=True)

    route = res.get("route")
    ans = res["response"]["answer"]
    sources = res["response"]["sources"]

    print(f"Route: {route}")
    print(f"Sources: {sources}")
    print(f"Answer: {ans}")

    assert route == "policy", f"Expected policy route, got {route}"
    assert "return_window.md" in sources, f"Expected return_window.md in sources, got {sources}"
    assert "opened fragrances" in ans.lower(), f"Expected opened fragrances mentioned: {ans}"
    assert "non-returnable" in ans.lower(), f"Expected non-returnable exclusion grounded: {ans}"
    print("[PASSED] Test 3 passed!")


def test_4_how_many_days_to_return_eligible_product():
    print("\n--- Test 4: 'How many days do I have to return an eligible product?' ---")
    res = run_agent("How many days do I have to return an eligible product?", return_state=True)

    route = res.get("route")
    ans = res["response"]["answer"]
    sources = res["response"]["sources"]

    print(f"Route: {route}")
    print(f"Sources: {sources}")
    print(f"Answer: {ans}")

    assert route == "policy", f"Expected policy route, got {route}"
    assert "return_window.md" in sources, f"Expected return_window.md in sources, got {sources}"
    assert "15 days" in ans, f"Expected supported 15-day return window in answer: {ans}"
    print("[PASSED] Test 4 passed!")


def test_5_return_policy_for_beauty_products():
    print("\n--- Test 5: 'What is the return policy for Beauty products?' ---")
    res = run_agent("What is the return policy for Beauty products?", return_state=True)

    route = res.get("route")
    ans = res["response"]["answer"]
    sources = res["response"]["sources"]

    print(f"Route: {route}")
    print(f"Sources: {sources}")
    print(f"Answer: {ans}")

    assert route == "policy", f"Expected policy route, got {route}"
    assert "return_window.md" in sources, f"Expected return_window.md in sources, got {sources}"
    assert "beauty" in ans.lower() or "cosmetic" in ans.lower(), f"Expected beauty-specific policy evidence: {ans}"
    assert "15 days" in ans, f"Expected 15-day window for Beauty: {ans}"
    print("[PASSED] Test 5 passed!")


def test_6_what_are_the_return_restrictions():
    print("\n--- Test 6: 'What are the return restrictions?' ---")
    res = run_agent("What are the return restrictions?", return_state=True)

    route = res.get("route")
    ans = res["response"]["answer"]
    sources = res["response"]["sources"]

    print(f"Route: {route}")
    print(f"Sources: {sources}")
    print(f"Answer: {ans}")

    assert route == "policy", f"Expected policy route, got {route}"
    assert "return_window.md" in sources, f"Expected return_window.md in sources, got {sources}"
    assert "non-returnable" in ans.lower() or "hygiene" in ans.lower() or "restrictions" in ans.lower(), f"Expected exclusion/restriction details: {ans}"
    print("[PASSED] Test 6 passed!")


def test_7_hi_pure_greeting():
    print("\n--- Test 7: 'Hi' ---")
    res = run_agent("Hi", return_state=True)

    route = res.get("route")
    ans = res["response"]["answer"]
    resp_type = res["response"]["response_type"]
    sources = res["response"]["sources"]

    print(f"Route: {route}")
    print(f"Type: {resp_type}")
    print(f"Sources: {sources}")
    print(f"Answer: {ans.encode('ascii', 'backslashreplace').decode('ascii')}")

    assert route == "conversational", f"Expected conversational route for 'Hi', got {route}"
    assert resp_type == "conversational", f"Expected conversational response type, got {resp_type}"
    assert sources == [], f"Expected no sources for pure greeting, got {sources}"
    assert "help" in ans.lower() or "how can i" in ans.lower() or "good" in ans.lower(), f"Unexpected greeting answer: {ans}"
    print("[PASSED] Test 7 passed!")


def main():
    print("================================================================================")
    print("       NYKAA ASSIST — POLICY RESPONSE QUALITY & COMPLETENESS TESTS             ")
    print("================================================================================")
    test_1_what_is_nykaas_return_policy()
    test_2_hi_what_is_the_return_policy()
    test_3_can_i_return_an_opened_fragrance()
    test_4_how_many_days_to_return_eligible_product()
    test_5_return_policy_for_beauty_products()
    test_6_what_are_the_return_restrictions()
    test_7_hi_pure_greeting()
    print("\n================================================================================")
    print("                 ALL 7 POLICY QUALITY TESTS PASSED (100%)                       ")
    print("================================================================================")


if __name__ == "__main__":
    main()
