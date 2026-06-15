"""BIRD data-dictionary loader + gold-column extraction (no API).

Builds, per database, the candidate column set with two text representations:
  BARE: 'table.original_column_name'                (Spider-style, names only)
  RICH: original + human column_name + column_description + value_description  (BIRD data dict)
and extracts, per question, the set of columns referenced by the gold SQL (the positive set
for the column-inclusion model). The match rate between gold columns and the known schema is
the feasibility gate.
"""
from __future__ import annotations
import csv, glob, os
import sqlglot
from sqlglot import exp

ROOT = os.path.join(os.path.dirname(__file__), "..")
DESC = os.path.join(ROOT, "data", "bird", "desc")


def load_schema(db):
    """-> {table_lower: {col_lower: {'orig','table','bare','rich'}}}"""
    out = {}
    for f in glob.glob(os.path.join(DESC, db, "*.csv")):
        table = os.path.basename(f)[:-4]
        cols = {}
        text = None
        for enc in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                text = open(f, encoding=enc).read()
                break
            except UnicodeDecodeError:
                continue
        for r in csv.DictReader(text.splitlines()):
            orig = (r.get("original_column_name") or "").strip()
            if not orig:
                continue
            name = (r.get("column_name") or "").strip()
            desc = (r.get("column_description") or "").strip()
            vald = (r.get("value_description") or "").strip()
            bare = f"{table}.{orig}"
            nm = name if (name and name.lower() != orig.lower()) else orig
            name_rep = f"{table}.{orig} ({nm})" if nm != orig else bare
            desc_rep = (f"{table}.{orig}: {desc}"[:600]) if desc and desc.lower() != orig.lower() else bare
            rich_parts = [f"Table {table}, column {orig}"]
            if name and name.lower() != orig.lower():
                rich_parts.append(f"({name})")
            if desc and desc.lower() != orig.lower():
                rich_parts.append(f": {desc}")
            if vald:
                rich_parts.append(f". Values: {vald}")
            rich = " ".join(rich_parts)
            cols[orig.lower()] = {"orig": orig, "table": table, "bare": bare,
                                  "name": name_rep, "desc": desc_rep, "rich": rich[:600]}
        out[table.lower()] = cols
    return out


def gold_columns(sql, schema):
    """Set of (table_lower, col_lower) referenced in the gold SQL, resolved against schema.

    BIRD SQL uses backtick / bracket quoting; we extract bare column names via sqlglot and
    resolve each to whatever table in this db owns that column name (most BIRD columns are
    unambiguous across the small per-db schema)."""
    col_owner = {}  # col_lower -> list of tables that have it
    for t, cols in schema.items():
        for c in cols:
            col_owner.setdefault(c, []).append(t)
    try:
        tree = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        try:
            tree = sqlglot.parse_one(sql)
        except Exception:
            return set(), 0, 0
    refs, hits = set(), 0
    seen = 0
    for c in tree.find_all(exp.Column):
        seen += 1
        name = c.name.lower()
        owners = col_owner.get(name, [])
        if owners:
            hits += 1
            # if the column qualified with a table, prefer it; else attach to all owners
            tbl = (c.table or "").lower()
            if tbl in schema and name in schema[tbl]:
                refs.add((tbl, name))
            else:
                for t in owners:
                    refs.add((t, name))
    return refs, hits, seen


def select_columns(sql, schema):
    """Columns appearing ONLY in the top-level SELECT projection list (what the query returns)."""
    col_owner = {}
    for t, cols in schema.items():
        for c in cols:
            col_owner.setdefault(c, []).append(t)
    try:
        tree = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        try:
            tree = sqlglot.parse_one(sql)
        except Exception:
            return set()
    sel = tree.find(exp.Select)
    if sel is None:
        return set()
    refs = set()
    for proj in sel.expressions:                 # projection expressions only
        for c in proj.find_all(exp.Column):
            name = c.name.lower()
            owners = col_owner.get(name, [])
            if not owners:
                continue
            tbl = (c.table or "").lower()
            if tbl in schema and name in schema[tbl]:
                refs.add((tbl, name))
            else:
                for t in owners:
                    refs.add((t, name))
    return refs


def load_questions():
    import json
    return json.load(open(os.path.join(ROOT, "data", "bird", "dev.json")))


if __name__ == "__main__":
    from collections import Counter
    qs = load_questions()
    schemas = {}
    tot_hits = tot_seen = 0
    base_num = base_den = 0
    cols_per_db = {}
    no_pos = 0
    for q in qs:
        db = q["db_id"]
        if db not in schemas:
            schemas[db] = load_schema(db)
            cols_per_db[db] = sum(len(c) for c in schemas[db].values())
        refs, hits, seen = gold_columns(q["SQL"], schemas[db])
        tot_hits += hits
        tot_seen += seen
        if not refs:
            no_pos += 1
        ncand = cols_per_db[db]
        base_num += len(refs)
        base_den += ncand
    print(f"questions: {len(qs)}")
    print(f"gold-column resolution rate: {tot_hits}/{tot_seen} = {tot_hits/tot_seen:.3f}")
    print(f"questions with ZERO resolved gold columns: {no_pos}")
    print(f"candidate columns per db: min {min(cols_per_db.values())}, "
          f"max {max(cols_per_db.values())}, mean {sum(cols_per_db.values())/len(cols_per_db):.0f}")
    print(f"base rate pi (gold cols / candidate cols): {base_num}/{base_den} = {base_num/base_den:.4f}")
    print(f"total unique candidate columns across dbs: {sum(cols_per_db.values())}")
