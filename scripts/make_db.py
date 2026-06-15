"""Create a small seeded SQLite airbnb_listings DB for execution-accuracy evaluation.

Deterministic (fixed seed). Includes duplicate cities within a country (so COUNT vs
COUNT DISTINCT genuinely differ) and some NULL number_of_rooms (for the IS NULL questions).

Run:  ./.venv/bin/python scripts/make_db.py
"""

from __future__ import annotations

import os
import random
import sqlite3

DB = os.path.join(os.path.dirname(__file__), "..", "data", "airbnb.sqlite")

CITY_COUNTRY = [
    ("Paris", "France"), ("Lyon", "France"), ("Tokyo", "Japan"), ("Osaka", "Japan"),
    ("New York", "USA"), ("Austin", "USA"), ("Miami", "USA"), ("London", "UK"),
    ("Berlin", "Germany"), ("Madrid", "Spain"), ("Rome", "Italy"), ("Lisbon", "Portugal"),
]


def main():
    rng = random.Random(20240613)
    rows = []
    for i in range(1, 61):
        city, country = rng.choice(CITY_COUNTRY)
        rooms = None if rng.random() < 0.1 else rng.randint(1, 8)
        year = rng.randint(2010, 2023)
        rows.append((i, city, country, rooms, year))

    os.makedirs(os.path.dirname(DB), exist_ok=True)
    if os.path.exists(DB):
        os.remove(DB)
    con = sqlite3.connect(DB)
    con.execute(
        "CREATE TABLE airbnb_listings ("
        "id INTEGER, city TEXT, country TEXT, number_of_rooms INTEGER, year_listed INTEGER)"
    )
    con.executemany("INSERT INTO airbnb_listings VALUES (?,?,?,?,?)", rows)
    con.commit()
    n = con.execute("SELECT COUNT(*) FROM airbnb_listings").fetchone()[0]
    nnull = con.execute(
        "SELECT COUNT(*) FROM airbnb_listings WHERE number_of_rooms IS NULL"
    ).fetchone()[0]
    ncountry = con.execute("SELECT COUNT(DISTINCT country) FROM airbnb_listings").fetchone()[0]
    con.close()
    print(f"wrote {DB}")
    print(f"  {n} rows, {nnull} with NULL rooms, {ncountry} distinct countries")


if __name__ == "__main__":
    main()
