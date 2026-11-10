"""Self-check for the parsers and the dedupe logic. Run: python test_collect.py"""
import tempfile
from pathlib import Path

from collect import CHANGE_COLS, PLAN_COLS, latest_state, new_rows, parse_changes, parse_plan, write_rows

PLAN_XML = b"""<timetable station="Frankfurt(Main)Hbf">
<s id="-123-2611101200-5"><tl f="F" t="p" o="80" c="ICE" n="577"/>
<ar pt="2611101155" pp="7" l="" ppth="Hamburg Hbf|Kassel"/><dp pt="2611101200" pp="7" ppth="Mannheim|M\xc3\xbcnchen Hbf"/></s>
<s id="-9-2611101210-1"><tl c="S" n="35"/><dp pt="2611101210" pp="101" l="8" ppth="Wiesbaden"/></s>
</timetable>"""

CHANGE_XML = b"""<timetable eva="8000105">
<s id="-123-2611101200-5" eva="8000105"><m t="h" c="2"/>
<ar ct="2611101203"><m t="d" c="43"/></ar><dp ct="2611101208" cp="8"><m t="d" c="43"/></dp></s>
<s id="-9-2611101210-1" eva="8000105"><dp cs="c" clt="2611101100"/></s>
</timetable>"""

plan = parse_plan(PLAN_XML, "8000105")
assert len(plan) == 2
assert plan[0]["category"] == "ICE" and plan[0]["number"] == "577" and plan[0]["eva"] == "8000105"
assert plan[0]["origin"] == "Hamburg Hbf" and plan[0]["destination"] == "München Hbf"
assert plan[1]["line"] == "8" and plan[1]["arr_pt"] == "" and plan[1]["origin"] == ""

changes = parse_changes(CHANGE_XML, "8000105")
assert changes[0]["dep_ct"] == "2611101208" and changes[0]["dep_cp"] == "8"
assert changes[0]["messages"] == "d:43 h:2"  # deduped and sorted across stop, arrival and departure
assert changes[1]["dep_cs"] == "c" and changes[1]["dep_clt"] == "2611101100"

with tempfile.TemporaryDirectory() as tmp:
    folder, days = Path(tmp) / "changes", ["2026-11-09", "2026-11-10"]
    first = new_rows(changes, latest_state(folder, days, CHANGE_COLS), CHANGE_COLS)
    assert len(first) == 2
    for row in first:
        row["observed_at"] = "2026-11-10T10:00:00Z"
    write_rows(folder / "2026-11-10" / "100000.csv.gz", first, CHANGE_COLS + ["observed_at"])

    # Same snapshot again: nothing new. Then one train's time changes: only that row is new.
    assert new_rows(parse_changes(CHANGE_XML, "8000105"), latest_state(folder, days, CHANGE_COLS), CHANGE_COLS) == []
    later = parse_changes(CHANGE_XML.replace(b"2611101208", b"2611101212"), "8000105")
    fresh = new_rows(later, latest_state(folder, days, CHANGE_COLS), CHANGE_COLS)
    assert [r["dep_ct"] for r in fresh] == ["2611101212"]

    # Duplicates inside one run are written once.
    assert len(new_rows(plan + plan, {}, PLAN_COLS)) == 2

    # Plans keep the first row per stop: a second copy with other values is not written (main() passes cols=[]).
    other = [dict(plan[0], destination="Elsewhere")]
    assert new_rows(other, {(plan[0]["stop_id"], plan[0]["eva"]): ()}, []) == []

# Special trains come only in fchg, with their own tl and pt; parse_plan picks them up there.
SPECIAL = b'<timetable><s id="-7-2611101300-2" eva="8000105"><tl t="s" c="ICE" n="9551"/><dp pt="2611101300" ct="2611101305"/></s></timetable>'
row = parse_plan(SPECIAL, "8000105")[0]
assert row["trip_type"] == "s" and row["dep_pt"] == "2611101300" and row["number"] == "9551"

print("ok")
