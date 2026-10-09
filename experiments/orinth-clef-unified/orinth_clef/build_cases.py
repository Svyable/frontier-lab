"""Deterministic *synthetic* 120-case routing corpus, not human evaluation gold.

Thirty distinct scenario families, four controlled phrasings each. Families
are assigned as group_id so held-out variants never cross split boundaries.
Construction labels are recorded separately and are not expert validated.
"""
import argparse
import json
from pathlib import Path

from .data import write_jsonl

# group, scenario, intended route, intended urgency. These are hand-authored
# construction intents only; an independent audit is required for true gold.
SCENARIOS = [
    ("payment-gateway-down", "The payment gateway is returning errors for every card transaction.", "technical", True),
    ("duplicate-charge", "A customer sees two successful charges for a single order.", "billing", False),
    ("login-error", "Hundreds of users are unable to log in following the latest deployment.", "technical", True),
    ("refund-status", "A customer requests the status of a refund issued last week.", "billing", False),
    ("invoice-tax-id", "An account manager needs the tax identification number corrected on their invoice.", "billing", False),
    ("regional-api-outage", "API endpoints across one region have failed all health checks.", "technical", True),
    ("subscription-proration", "An account owner is confused by prorated plan-upgrade charges.", "billing", False),
    ("dashboard-blank", "The analytics dashboard is blank for a subset of users after an update.", "technical", False),
    ("mass-overbilling", "Every enterprise customer was charged ten times their contracted amount today.", "billing", True),
    ("password-reset-email", "Password reset emails have stopped arriving for all customers.", "technical", True),
    ("receipt-copy", "A customer asks for a copy of their last paid receipt.", "billing", False),
    ("app-crash", "The iOS application crashes immediately on launch for all users.", "technical", True),
    ("coupon-validity", "A customer wants clarification about a coupon expiration date.", "billing", False),
    ("slow-search", "Product search queries take eight seconds after an index change.", "technical", False),
    ("bank-transfer", "An account asks why a completed bank transfer is not marked paid.", "billing", False),
    ("database-data-loss", "The primary database lost recent writes and some user records are missing.", "technical", True),
    ("plan-downgrade", "A customer wants to downgrade to a lower-cost monthly plan.", "billing", False),
    ("timezone-bug", "Calendar reminders fire at the wrong hour after a timezone conversion.", "technical", False),
    ("fraud-charge-spike", "Hundreds of accounts are reporting unauthorized recurring charges.", "billing", True),
    ("export-button", "CSV exports fail when users press the download button.", "technical", False),
    ("credit-note", "The finance contact requests a formal credit note for a corrected invoice.", "billing", False),
    ("webhook-queue", "Webhooks are stuck in the queue and third-party integrations have stopped updating.", "technical", True),
    ("pricing-question", "A prospect asks whether the annual tier includes a setup charge.", "billing", False),
    ("email-notification-bug", "Product notification emails arrive with broken links for all users.", "technical", False),
    ("enterprise-contract-breach", "A billing configuration caused immediate incorrect invoices for all enterprise contracts.", "billing", True),
    ("mobile-sync", "A phone and desktop client display conflicting saved preferences.", "technical", False),
    ("tax-exemption", "A nonprofit asks to have its tax-exempt status applied to the next invoice.", "billing", False),
    ("file-uploads-down", "Every user-uploaded file is failing validation with a server error.", "technical", True),
    ("cancel-policy", "A subscriber asks whether cancellation produces an automatic refund.", "billing", False),
    ("authentication-bypass", "A security engineer reports a reproducible authentication bypass in production.", "technical", True),
]

FRAMES = (
    "{scenario}",
    "Customer support ticket: {scenario}",
    "Incident triage summary from the service desk: {scenario}",
    "A teammate forwarded this issue for routing: {scenario}",
)


def create_cases() -> tuple[list[dict], list[dict]]:
    cases = []
    references = []
    for group, scenario, expected_team, expected_urgent in SCENARIOS:
        for index, frame in enumerate(FRAMES):
            ident = f"{group}-{index:02d}"
            # Two variants of field names make rigid fixed-key memorization harder.
            team_id = "team" if index % 2 == 0 else "responsible_team"
            urgent_id = "urgent" if index % 2 == 0 else "needs_immediate_attention"
            request = {
                "model": "clef-flash",
                "state": frame.format(scenario=scenario),
                "questions": {
                    team_id: {
                        "type": "choice",
                        "instructions": "Which support team should own this issue?",
                        "criteria": {
                            "billing": "Billing: invoices, subscriptions, charges, refunds, pricing and payments disputes",
                            "technical": "Technical: application failures, security faults, outages, software and integration bugs",
                        },
                    },
                    urgent_id: {
                        "type": "noul",
                        "instructions": "Does this incident require immediate attention due to widespread failure or substantial harm?",
                    },
                },
            }
            cases.append({
                "id": ident,
                "group_id": group,
                "source": "deterministic_handwritten_synthetic_scenario",
                "request": request,
            })
            references.append({
                "id": ident,
                "group_id": group,
                "reference_kind": "construction_intent_unverified",
                "intent_decisions": {team_id: expected_team, urgent_id: expected_urgent},
            })
    return cases, references


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cases", type=Path, default=Path("data/cases_120.jsonl"))
    p.add_argument("--references", type=Path, default=Path("data/construction_intent_120.jsonl"))
    args = p.parse_args()
    cases, refs = create_cases()
    write_jsonl(args.cases, cases)
    write_jsonl(args.references, refs)
    print(json.dumps({"scenario_families": len(SCENARIOS), "cases": len(cases),
                      "unverified_reference_intents": len(refs)}))


if __name__ == "__main__":
    main()
