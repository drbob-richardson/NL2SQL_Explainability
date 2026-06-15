"""Generate a larger airbnb eval set by templating (question, gold) pairs.

Deterministic. Extends the 31 hand-curated cheat-sheet items (ids 1-31, kept verbatim with
their ids so their cached samples are reused) with templated items (ids 100+) spanning
projection / filter / aggregate / group / order / limit / having / distinct / between, plus
a few deliberately AMBIGUOUS phrasings that induce genuine model disagreement.

Writes data/airbnb_eval_large.json.  Run:  ./.venv/bin/python scripts/gen_eval.py
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, "..", "data", "airbnb_eval.json")
OUT = os.path.join(HERE, "..", "data", "airbnb_eval_large.json")

NUM = ["number_of_rooms", "year_listed"]
TEXT = ["city", "country"]
AGG = {"average": "AVG", "total": "SUM", "maximum": "MAX", "minimum": "MIN"}
CMP = {"greater than": ">", "less than": "<", "at least": ">=", "at most": "<=",
       "equal to": "="}


def build():
    items = []
    nid = 100

    def add(q, gold):
        nonlocal nid
        items.append({"id": nid, "question": q, "gold": gold})
        nid += 1

    # global aggregates
    for phrase, fn in AGG.items():
        add(f"What is the {phrase} number_of_rooms across all listings?",
            f"SELECT {fn}(number_of_rooms) FROM airbnb_listings")
        add(f"What is the {phrase} year_listed across all listings?",
            f"SELECT {fn}(year_listed) FROM airbnb_listings")

    # group-by aggregates
    for phrase, fn in AGG.items():
        for col in TEXT:
            add(f"What is the {phrase} number_of_rooms for each {col}?",
                f"SELECT {col}, {fn}(number_of_rooms) FROM airbnb_listings GROUP BY {col}")

    # numeric filters
    for col in NUM:
        for phrase, op in CMP.items():
            add(f"Get all listings where {col} is {phrase} 4",
                f"SELECT * FROM airbnb_listings WHERE {col} {op} 4")

    # text equality / membership
    for val in ["France", "Japan", "USA"]:
        add(f"Get all listings in {val}",
            f"SELECT * FROM airbnb_listings WHERE country = '{val}'")
    add("Get all listings in France or Japan",
        "SELECT * FROM airbnb_listings WHERE country IN ('France', 'Japan')")

    # ordering + limit
    for col in NUM:
        for d, kw in [("highest", "DESC"), ("lowest", "ASC")]:
            add(f"List all listings ordered from {d} {col} to the other",
                f"SELECT * FROM airbnb_listings ORDER BY {col} {kw}")
    for n in (3, 10):
        add(f"Get the first {n} listings", f"SELECT * FROM airbnb_listings LIMIT {n}")

    # distinct
    for col in TEXT:
        add(f"Get the distinct {col} values among listings",
            f"SELECT DISTINCT {col} FROM airbnb_listings")

    # between
    add("Get listings with number_of_rooms between 2 and 5",
        "SELECT * FROM airbnb_listings WHERE number_of_rooms BETWEEN 2 AND 5")
    add("Get listings listed between 2015 and 2020",
        "SELECT * FROM airbnb_listings WHERE year_listed BETWEEN 2015 AND 2020")

    # having / counts per group
    add("Which countries have more than 5 listings?",
        "SELECT country FROM airbnb_listings GROUP BY country HAVING COUNT(id) > 5")
    add("Which years had more than 3 listings?",
        "SELECT year_listed FROM airbnb_listings GROUP BY year_listed HAVING COUNT(id) > 3")
    add("How many listings are there per country?",
        "SELECT country, COUNT(*) FROM airbnb_listings GROUP BY country")
    add("How many listings are there per city?",
        "SELECT city, COUNT(*) FROM airbnb_listings GROUP BY city")

    # combined filters
    add("Get listings in France with number_of_rooms greater than 3",
        "SELECT * FROM airbnb_listings WHERE country = 'France' AND number_of_rooms > 3")
    add("Get listings in Japan or listed after 2018",
        "SELECT * FROM airbnb_listings WHERE country = 'Japan' OR year_listed > 2018")

    # deliberately ambiguous (induce model disagreement; gold is one reasonable reading)
    add("Show me the busiest country by listings",
        "SELECT country FROM airbnb_listings GROUP BY country ORDER BY COUNT(*) DESC LIMIT 1")
    add("What's a typical number of rooms?",
        "SELECT AVG(number_of_rooms) FROM airbnb_listings")
    add("Find recent listings",
        "SELECT * FROM airbnb_listings WHERE year_listed >= 2020")
    add("Which cities are popular?",
        "SELECT city FROM airbnb_listings GROUP BY city ORDER BY COUNT(*) DESC LIMIT 3")
    return items


def main():
    with open(SRC) as f:
        base = json.load(f)
    extra = build()
    merged = dict(base)
    merged["questions"] = base["questions"] + extra
    merged["source"] = base["source"] + " + templated generator"
    with open(OUT, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"wrote {OUT}")
    print(f"  {len(base['questions'])} curated + {len(extra)} generated "
          f"= {len(merged['questions'])} questions")


if __name__ == "__main__":
    main()
