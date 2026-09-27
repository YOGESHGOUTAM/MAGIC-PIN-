"""
Generates submission.jsonl containing compositions for the 30 canonical test pairs.
Validates each composition against challenge requirements.
"""

from __future__ import annotations

import json
from pathlib import Path

from bot import compose

DATASET_DIR = Path(__file__).parent / "dataset"
EXPANDED_DIR = DATASET_DIR / "expanded"


def load_dataset():
    """Load categories, merchants, customers, triggers."""
    categories = {}
    merchants = {}
    customers = {}
    triggers = {}

    # Categories
    cat_dir = DATASET_DIR / "categories"
    if cat_dir.exists():
        for f in cat_dir.glob("*.json"):
            data = json.load(open(f, encoding="utf-8"))
            categories[data.get("slug", f.stem)] = data

    # Merchants (expanded preferred)
    m_dir = EXPANDED_DIR / "merchants"
    if m_dir.exists():
        for f in m_dir.glob("*.json"):
            m = json.load(open(f, encoding="utf-8"))
            merchants[m["merchant_id"]] = m
    else:
        for m in json.load(open(DATASET_DIR / "merchants_seed.json", encoding="utf-8")).get("merchants", []):
            merchants[m["merchant_id"]] = m

    # Customers (expanded preferred)
    c_dir = EXPANDED_DIR / "customers"
    if c_dir.exists():
        for f in c_dir.glob("*.json"):
            c = json.load(open(f, encoding="utf-8"))
            customers[c["customer_id"]] = c
    else:
        for c in json.load(open(DATASET_DIR / "customers_seed.json", encoding="utf-8")).get("customers", []):
            customers[c["customer_id"]] = c

    # Triggers (expanded preferred)
    t_dir = EXPANDED_DIR / "triggers"
    if t_dir.exists():
        for f in t_dir.glob("*.json"):
            t = json.load(open(f, encoding="utf-8"))
            triggers[t["id"]] = t
    else:
        for t in json.load(open(DATASET_DIR / "triggers_seed.json", encoding="utf-8")).get("triggers", []):
            triggers[t["id"]] = t

    return categories, merchants, customers, triggers


def main():
    test_pairs_path = EXPANDED_DIR / "test_pairs.json"
    if not test_pairs_path.exists():
        raise FileNotFoundError(f"{test_pairs_path} not found. Run generate_dataset.py first.")

    test_pairs_data = json.load(open(test_pairs_path, encoding="utf-8"))
    pairs = test_pairs_data.get("pairs", [])
    print(f"Loaded {len(pairs)} test pairs from {test_pairs_path}")

    categories, merchants, customers, triggers = load_dataset()
    print(f"Loaded: {len(categories)} categories, {len(merchants)} merchants, {len(customers)} customers, {len(triggers)} triggers")

    output_lines = []

    for pair in pairs:
        test_id = pair["test_id"]
        trg_id = pair["trigger_id"]
        mid = pair["merchant_id"]
        cid = pair.get("customer_id")

        trg = triggers.get(trg_id)
        if not trg:
            raise KeyError(f"Trigger {trg_id} not found in dataset")

        merchant = merchants.get(mid)
        if not merchant:
            raise KeyError(f"Merchant {mid} not found in dataset")

        cat_slug = merchant.get("category_slug", "")
        category = categories.get(cat_slug, {"slug": cat_slug, "voice": {}})

        customer = customers.get(cid) if cid else None

        # Compose message
        composed = compose(category, merchant, trg, customer)

        submission_entry = {
            "test_id": test_id,
            "body": composed["body"],
            "cta": composed["cta"],
            "send_as": composed["send_as"],
            "suppression_key": composed["suppression_key"],
            "rationale": composed["rationale"],
        }
        output_lines.append(submission_entry)

    out_file = Path(__file__).parent / "submission.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for entry in output_lines:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Successfully generated {len(output_lines)} lines in {out_file}")

    # Validation
    lines = open(out_file, encoding="utf-8").read().strip().split("\n")
    assert len(lines) == 30, f"Expected 30 lines, got {len(lines)}"
    for idx, l in enumerate(lines, 1):
        obj = json.loads(l)
        assert "test_id" in obj and obj["test_id"] == f"T{idx:02d}"
        assert "body" in obj and len(obj["body"]) > 20
        assert "cta" in obj
        assert "send_as" in obj
        assert "suppression_key" in obj
        assert "rationale" in obj
        # Assert no taboos
        assert "guaranteed" not in obj["body"].lower()

    print("[PASS] submission.jsonl passed all structure and content validations!")


if __name__ == "__main__":
    main()
