#!/usr/bin/env python3
"""Campaign engine for the CATLASS A2/A3 -> Ascend950 migration workflow.

State is an append-only event log. Phase is a fold over it, never a mutable field, so every
command is idempotent and no command can lose a decision another one recorded.

Subcommands, in workflow order:

    init      validate plan.json, create the run directory
    refs      resolve the CATLASS and asc-devkit reference checkouts at their pins
    profile   validate the discovered target profile -- four sections, all answered
    check     validate a phase's artifacts and record the promotion
    gate      render a gate packet -- `--phase migrate` (default) or `optimize`; exits 2
    confirm   record the human's decision at either gate
    prove     execute the discovered build+run and parse the accuracy evidence
    remote    reach the declared remote side: push files, run a command, fetch results
    park      record a unit as blocked, with a reason
    status    fold the log and print the exact next command per unit
    report    generate report.md from the log and the artifacts

Every command is safe to re-run: promotions take the highest rank reached, so a repeated
event is a no-op rather than a state change.
"""

import argparse
import glob
import hashlib
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone

# --- vocabulary --------------------------------------------------------------------------

# Ascending change surface. These are *analysis vocabulary*: the analyst must exclude the
# cheaper classes with evidence before proposing a dearer one. Deliberately, no code branches
# on the value -- gating on a label the agent assigns itself would be theatre.
ROUTES = ("retarget", "unblock", "reimplement", "redesign")

# `redesign` changes the external contract, so it is not a migration. A unit that lands there
# is reported and terminated rather than implemented.
ROUTE_TERMINAL = "redesign"

# Shared components either generalize an existing symbol or add one. Consumers of a
# generalization are derived from the tree, not asserted in the analysis.
SHARED_KINDS = ("generalize", "add")

# Rank orders promotion. `COMPILED` is recorded before device execution; optimization
# phases follow PROVEN, and a re-proof after APPLY is absorbed by the fold.
PHASES = ("INTAKE", "ANALYZED", "AUTHORIZED", "IMPLEMENTED", "COMPILED", "PROVEN",
          "OPT_SCREENED", "OPT_AUTHORIZED", "OPTIMIZED")
RANK = {p: i for i, p in enumerate(PHASES)}

# Which event promotes to which phase. Anything else in the log is campaign-level fact.
PROMOTES = {"unit.analyzed": "ANALYZED", "unit.authorized": "AUTHORIZED",
            "unit.implemented": "IMPLEMENTED", "unit.compiled": "COMPILED",
            "unit.proven": "PROVEN",
            "unit.opt_screened": "OPT_SCREENED", "unit.opt_authorized": "OPT_AUTHORIZED",
            "unit.optimized": "OPTIMIZED"}

# `unit.applied` is deliberately not here. That the rewrite landed is a fact about the tree,
# not a state the campaign resumes from: between the landing and its re-prove the unit is
# still OPT_AUTHORIZED, and `status` must go on asking for the re-prove. A rank of its own
# would name a unit whose rewritten path has no accuracy record at all, one step below
# OPTIMIZED -- which is exactly the claim this event exists to stop being made.
#
# `unit.prove_failed` is not here either, and for the opposite reason: it must never *raise* a
# rank, but it must not be invisible. It sets an overlay flag the way `unit.parked` does, so a
# unit whose most recent proof attempt failed still folds to whatever rank it earned, while
# `status` and `report` say plainly that the last attempt failed. Without it a failing re-prove
# wrote `proof.json` and appended nothing, and the report went on printing `errors=0` for a
# unit whose last run disagreed.

# `applied` records that the rewrite landed and is the event `check --phase optimized` orders
# the re-prove against; it promotes nothing.
CHECKABLE = ("analyzed", "implemented", "compiled", "screened", "applied", "optimized")

# The two gates. Each is rendered by `gate` and recorded by `confirm`, and each sits immediately
# before the writes it authorizes: `migrate` before any migration write, `optimize` before any
# byte of the L0C->UB rewrite.
GATES = ("migrate", "optimize")

# The two data-path strategies for the rewrite, and the only two. `coexist` keeps the proven
# baseline compilable behind the example's opt-in switch; `replace` removes the old path, which
# is the one irreversible act in this workflow -- nothing here commits, so a failed re-prove
# after a `replace` leaves nothing to restore. A human picks between them, never this tool.
STRATEGIES = ("coexist", "replace")

# The two Fixpipe L0C->UB modes. `SPLIT_M` (dualDstCtl=1) halves M across the two AIV sub-blocks
# and needs the dual-flag cross-core protocol; `NO_SPLIT` (dualDstCtl=0) gives one sub-block the
# whole M x N tile and uses single flags. The UB budget is what decides between them.
COPY_MODES = ("SPLIT_M", "NO_SPLIT")

# The performance case table's one home. `init` stages either a supplied table or the
# bundled template here, and gate packets and reports read this path.
PERF_CASES = "perf_cases.md"
PERF_CASE_TEMPLATE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "assets", "perf_case_template.md")

REF_REPOS = {"catlass": "https://gitcode.com/cann/catlass",
             "asc-devkit": "https://gitcode.com/cann/asc-devkit"}

# Canonical reference pins. A campaign can resolve only these immutable full revisions.
REF_PINS = {"catlass": "89e1fc39881a715882b9b47459add06ba270105c",
            "asc-devkit": "512674e996da0feaee6e7f435e4efc1cad1d74fb"}

WORK_DIR = ".agents-work"
# Matches this skill's directory name. It names the run-directory segment under
# `.agents-work/`; the reference cache deliberately does not carry it, because `<name>@<rev>`
# is already content-addressed and two skills wanting the same pin want the same directory.
SKILL_NAME = "catlass-ascend950-migration"

# The migrated example prints exactly one of these, and `prove` counts the matches rather than
# taking the first: two marked lines are two claims, and the first one silently hid the second.
# Parsing a JSON object beats regexing a success string: `Compare success.` carries no numbers,
# so it cannot distinguish a real pass from a comparison that never ran.
EVIDENCE_RE = re.compile(r"^CATLASS_EVIDENCE\s+(\{.*\})\s*$", re.M)

# An append is atomic only while the whole line lands in one write. Keep payloads small and
# put bulk in artifact files; this bound is asserted rather than hoped for.
MAX_EVENT_BYTES = 3500

PROBE_CONCERNS = ("build", "golden", "registration", "arch-gating")


class Fail(Exception):
    """A contract violation to report to the user, never a traceback."""


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError as exc:
        raise Fail(f"cannot read {path}: {exc}")


def write(path, text):
    """Atomic replace. A torn write would corrupt an artifact the next phase reads, and
    os.replace on the same filesystem is one line more than the naive open()."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text if text.endswith("\n") else text + "\n")
    os.replace(tmp, path)


def load_json(path, where):
    try:
        return json.loads(read(path))
    except json.JSONDecodeError as exc:
        raise Fail(f"{where} is not valid JSON ({path}): {exc}")


def dump_json(path, obj):
    write(path, json.dumps(obj, indent=2, ensure_ascii=False))


def closed(obj, allowed, where):
    """Reject unknown keys. A typo in an artifact key is otherwise a silently missing field."""
    if not isinstance(obj, dict):
        raise Fail(f"{where} must be a JSON object")
    extra = sorted(set(obj) - set(allowed))
    if extra:
        raise Fail(f"unknown key(s) in {where}: {', '.join(extra)}; allowed: "
                   f"{', '.join(sorted(allowed))}")


def need_str(obj, key, where):
    val = obj.get(key)
    if not isinstance(val, str) or not val.strip():
        raise Fail(f"{where}.{key} must be a non-empty string")
    return val


def need_list(obj, key, where, of=str, allow_empty=True):
    val = obj.get(key, [])
    if not isinstance(val, list):
        raise Fail(f"{where}.{key} must be a list")
    if not allow_empty and not val:
        raise Fail(f"{where}.{key} must be a non-empty list")
    for i, item in enumerate(val):
        if not isinstance(item, of):
            raise Fail(f"{where}.{key}[{i}] must be {of.__name__}")
    return val


def clip(text, width):
    """One packet cell, bounded. A `dtype` field that is a sentence is legitimate content --
    the analyst is asked to record stored dtype *and* Cube-operand dtype -- but a tensor table
    laid out from it pushed the header past 1000 characters, and the packet is the one place
    in this workflow where readability is load-bearing. Clipped here, whole in findings.json.
    """
    text = str(text)
    return text if len(text) <= width else text[:width - 1] + "\u2026"


# --- paths -------------------------------------------------------------------------------

def contained(root, rel, where):
    """Resolve `rel` under `root`, refusing an escape. Symlinks are resolved, so a link out
    of the tree is caught rather than followed."""
    if os.path.isabs(rel):
        raise Fail(f"{where} must be relative, got {rel!r}")
    root_real = os.path.realpath(root)
    resolved = os.path.realpath(os.path.join(root_real, rel))
    if resolved != root_real and not resolved.startswith(root_real + os.sep):
        raise Fail(f"{where} escapes the target root: {rel!r} -> {resolved}")
    return resolved


def cache_root(target_root):
    """Return the per-tree reference cache under `.agents-work/`."""
    return os.path.join(os.path.abspath(target_root), WORK_DIR, ".cache", "refs")


# --- event log ---------------------------------------------------------------------------

def log_path(run_dir):
    return os.path.join(run_dir, "events.jsonl")


def append_event(run_dir, event, unit=None, actor="agent", **payload):
    """One line, one write, O_APPEND. Concurrent appenders interleave lines but never
    interleave within a line, so no lock is needed and no update is lost."""
    rec = {"ts": now(), "actor": actor, "event": event, "unit": unit, "payload": payload}
    line = json.dumps(rec, separators=(",", ":"), ensure_ascii=False) + "\n"
    if len(line.encode("utf-8")) > MAX_EVENT_BYTES:
        raise Fail(f"event '{event}' payload is {len(line)} bytes, over the {MAX_EVENT_BYTES} "
                   "byte single-write bound. Put the bulk in an artifact file and reference "
                   "its path in the payload.")
    with open(log_path(run_dir), "a", encoding="utf-8") as fh:
        fh.write(line)
    return rec


def read_events(run_dir):
    """Every event, oldest first. A malformed final line is a torn write from a killed
    process and is dropped with a warning; a malformed line anywhere else is real corruption
    and is an error, because silently skipping it would lose a recorded decision."""
    path = log_path(run_dir)
    if not os.path.exists(path):
        return []
    lines = [ln for ln in read(path).split("\n") if ln.strip()]
    out = []
    for i, line in enumerate(lines):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                print(f"warning: dropped a torn final line in {path} (killed mid-write)",
                      file=sys.stderr)
                break
            raise Fail(f"{path} line {i + 1} is not valid JSON. This is not a torn write "
                       "(it is not the last line), so it is corruption: recover the line or "
                       "truncate the log at it deliberately.")
    return out


def fold(run_dir, plan):
    """Derive current state from the log. This is the only place phase is computed."""
    units = {u["id"]: {"id": u["id"], "source": u["source"], "target": u["target"],
                       "phase": "INTAKE", "route": None, "parked": None, "proof": None,
                       "prove_failed": None, "opt_skipped": False,
                       "excluded": False, "opt_excluded": False, "strategy": None,
                       "opt_manifest": None, "authorization": None,
                       "authorization_invalidated": False, "authorization_reanalyzed": False}
             for u in plan["units"]}
    # Two gates, so two of everything they record. A unit set aside at the migration gate never
    # gets built; one set aside at the optimization gate keeps its migration and simply is not
    # rewritten -- different outcomes that a single flag would have conflated into "excluded".
    camp = {"refs": None, "profile": False, "skill_head": None,
            "gate": None, "confirmed": None, "opt_gate": None, "opt_confirmed": None}
    for ev in read_events(run_dir):
        name, uid, pay = ev["event"], ev.get("unit"), ev.get("payload") or {}
        if uid is not None and uid not in units:
            continue  # an event for a unit no longer in the plan; the plan is the scope
        if name in PROMOTES:
            row = units[uid]
            if RANK[PROMOTES[name]] > RANK[row["phase"]]:
                row["phase"] = PROMOTES[name]
            if pay.get("route"):
                row["route"] = pay["route"]
            if name == "unit.analyzed" and row["authorization_invalidated"]:
                # Re-analysis is evidence for a replacement decision, not a way to revive the
                # old grant. Only `unit.authorized` below clears the invalidation.
                row["authorization_reanalyzed"] = True
            if name == "unit.authorized":
                row["authorization"] = {
                    key: pay.get(key) for key in ("plan", "profile", "findings")}
                row["authorization_invalidated"] = False
                row["authorization_reanalyzed"] = False
            if name == "unit.proven":
                row["proof"] = pay
                # A passing proof clears any earlier failure: the last attempt is what counts,
                # in both directions.
                row["prove_failed"] = None
            if name == "unit.opt_authorized":
                # The strategy AND the file list the human authorized, snapshotted out of
                # `screen.json` at the moment of confirmation and carried in the append-only
                # log, which is the authority. APPLY compares the screen file against both: an
                # artifact edited after the packet was approved no longer describes what was
                # approved. A re-authorization overwrites these, which is the point -- the
                # LATEST confirmation is the one in force.
                row["strategy"] = pay.get("strategy")
                row["opt_manifest"] = pay.get("manifest")
        elif name == "plan.frozen":
            camp["skill_head"] = pay.get("skill_head")
        elif name == "unit.authorization_invalidated":
            units[uid]["authorization_invalidated"] = True
            units[uid]["authorization_reanalyzed"] = False
        elif name == "unit.parked":
            units[uid]["parked"] = pay.get("reason") or "no reason recorded"
            # Interrupted pre-snapshot campaigns can have the park but no separate invalidation.
            # Its recorded phase is still enough to prevent that contradicted grant reviving.
            if pay.get("at_phase") == "AUTHORIZED":
                units[uid]["authorization_invalidated"] = True
                units[uid]["authorization_reanalyzed"] = False
        elif name == "unit.unparked":
            units[uid]["parked"] = None
        elif name == "unit.prove_failed":
            # An overlay, never a rank change. A unit that once proved keeps the rank it
            # earned -- demoting it would erase the fact that it did -- but the report and
            # `status` must not go on presenting a stale pass as the current state.
            units[uid]["prove_failed"] = pay
        elif name == "unit.opt_skipped":
            units[uid]["opt_skipped"] = True
        elif name == "refs.resolved":
            camp["refs"] = pay
        elif name == "profile.recorded":
            camp["profile"] = True
        elif name == "gate.presented":
            camp["opt_gate" if pay.get("gate") == "optimize" else "gate"] = pay
        elif name == "campaign.confirmed":
            # `actor` is stamped by `append_event` at the event's top level, not inside the
            # payload — it is provenance, so the command owns it and no caller can pass it in.
            optimize = pay.get("gate") == "optimize"
            camp["opt_confirmed" if optimize else "confirmed"] = dict(pay, actor=ev.get("actor"))
            # The latest confirmation defines this gate's exclusions. Intentional migration
            # skips are additive: previously authorized units are not re-presented, so a later
            # decision for an excluded unit cannot revoke their recorded stop.
            key = "opt_excluded" if optimize else "excluded"
            for row in units.values():
                row[key] = False
            for ex in pay.get("excluded") or []:
                if ex in units:
                    units[ex][key] = True
            if not optimize:
                for uid in pay.get("skip_optimize") or []:
                    if uid in units:
                        units[uid]["opt_skipped"] = True
    return units, camp


# --- plan --------------------------------------------------------------------------------

PLAN_KEYS = ("version", "request", "target_root", "refs", "optimize", "units")
UNIT_KEYS = ("id", "source", "target", "note")


# Cube operands stored as `AscendC::int4b_t` are outside this migration's supported
# backend. A unit may store int4 only when it casts to int8 before Cube; the Cube operands
# in that form are int8 and are not refused. The exact operand token keeps unrelated int4
# forms out of this rule.
BLOCKED_OPERAND_DTYPE = "AscendC::int4b_t"
BLOCKED_OPERAND_RE = re.compile(r"\bint4b_t\b")

# The scan matches `ElementA` or `ElementB` declarations in source code. It is a textual
# backstop; aliases remain an ANALYZE finding.
SOURCE_SUFFIXES = (".cpp", ".cc", ".cxx", ".hpp", ".h", ".cuh", ".inc")
CUBE_OPERAND_RE = re.compile(r"\bElement[AB]\b")


def scan_blocked_dtype(root, rel, limit=8):
    """Cube-operand declarations naming the refused dtype, as `path:line` strings."""
    base = os.path.join(os.path.realpath(root), rel)
    files = [base] if os.path.isfile(base) else [
        os.path.join(d, fn) for d, _sub, names in os.walk(base) for fn in sorted(names)]
    hits = []
    for p in sorted(f for f in files if f.lower().endswith(SOURCE_SUFFIXES)):
        try:
            with open(p, encoding="utf-8", errors="ignore") as fh:
                for i, line in enumerate(fh, 1):
                    if BLOCKED_OPERAND_RE.search(line) and CUBE_OPERAND_RE.search(line):
                        hits.append(f"{os.path.relpath(p, os.path.realpath(root))}:{i}")
                        if len(hits) >= limit:
                            return hits
        except OSError:
            continue
    return hits


def refuse_blocked_dtype(blocked):
    """One message for both gates. `blocked` is [(unit_id, source, [hits])]."""
    lines = [f"{BLOCKED_OPERAND_DTYPE} is not migratable to this target, and these unit(s) use it:"]
    for uid, src, hits in blocked:
        lines.append(f"  unit {uid}  ({src})")
        for h in hits:
            lines.append(f"      {h}")
        if len(hits) >= 8:
            lines.append("      ... (list truncated)")
    lines += [
        "",
        "  Direct Cube int4 operands require new-backend work, not a migration. There is no "
        "  override or migration route for this refusal.",
        "",
        "  Tell the user which units were refused and that a separate backend decision is "
        "  required.",
        "",
        "  Storing int4 and casting to int8 before Cube is allowed: its Cube operands are int8. "
        "  This refusal applies only when `ElementA` or `ElementB` is the s4 type itself.",
    ]
    return Fail("\n".join(lines))


def load_plan(path):
    raw = load_json(path, "plan")
    closed(raw, PLAN_KEYS, "plan")
    if raw.get("version") != 1 or type(raw.get("version")) is not int:
        raise Fail("plan 'version' must be the integer 1")
    # The verbatim request is kept so the report can be checked against what was actually
    # asked, and so a resumed run cannot drift from the original intent.
    request = need_str(raw, "request", "plan")
    given_root = raw.get("target_root")
    if given_root is not None and not isinstance(given_root, str):
        raise Fail(f"plan.target_root must be a string path, got {type(given_root).__name__}")
    root = os.path.abspath(given_root or ".")
    if not os.path.isdir(root):
        raise Fail(f"plan.target_root is not an existing directory: {root}")

    refs = dict(REF_PINS)
    given = raw.get("refs")
    if given is not None and not isinstance(given, dict):
        raise Fail("plan.refs must be a JSON object mapping a reference name to a revision")
    given = given or {}
    closed(given, REF_REPOS, "plan.refs")
    for name, rev in given.items():
        if rev != REF_PINS[name]:
            raise Fail(f"plan.refs.{name} must exactly match the canonical full pin "
                       f"{REF_PINS[name]}")

    opt = raw.get("optimize")
    if opt is not None and not isinstance(opt, dict):
        raise Fail("plan.optimize must be a JSON object with a single key, 'enabled'")
    opt = opt or {}
    closed(opt, ("enabled",), "plan.optimize")
    # Optimization is enabled unless FRAME freezes an explicit opt-out in the plan.
    if not isinstance(opt.get("enabled", True), bool):
        raise Fail("plan.optimize.enabled must be a JSON boolean")

    raw_units = raw.get("units")
    if not isinstance(raw_units, list) or not raw_units:
        raise Fail("plan 'units' must be a non-empty list")
    units, seen, blocked = [], set(), []
    for i, u in enumerate(raw_units):
        where = f"plan.units[{i}]"
        closed(u, UNIT_KEYS, where)
        uid = need_str(u, "id", where)
        # The id becomes a directory name under units/ and appears in every CLI invocation, so
        # it has to be a single safe path segment. Unconstrained, an id containing '/' silently
        # nested the unit directory (units/a/b/logs) and put a separator into every --unit
        # argument; '.' and '..' would be worse.
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", uid) or uid in (".", ".."):
            raise Fail(f"{where}.id must be a single path-safe token matching "
                       f"[A-Za-z0-9][A-Za-z0-9._-]* (it names a directory under units/ and is "
                       f"passed to --unit), got {uid!r}")
        if uid in seen:
            raise Fail(f"duplicate unit id {uid!r}")
        seen.add(uid)
        src = need_str(u, "source", where)
        tgt = need_str(u, "target", where)
        # A placeholder copied out of a template is a legal relative path, so it passed every
        # check and got frozen into plan.json -- after which correcting it costs a fresh run
        # directory. Refuse the template's own punctuation instead.
        for key, val in (("source", src), ("target", tgt)):
            if "<" in val or ">" in val:
                raise Fail(f"{where}.{key} still contains a template placeholder: {val!r}. The "
                           f"plan templates print `<...>` for you to replace; `init` freezes "
                           f"whatever it is given, so fill it in from the naming row you "
                           f"measured (references/01-frame.md section 4).")
        # A unit's source must exist now; its target must be a legal place to write later.
        if not os.path.exists(contained(root, src, f"{where}.source")):
            raise Fail(f"{where}.source does not exist under {root}: {src}")
        tgt_abs = contained(root, tgt, f"{where}.target")
        # An existing target is the one candidate test that has no route class: a pre-existing
        # target-architecture directory means the migration may already exist, and an authorized
        # implementer would overwrite a working, registered, tested example.
        if os.path.exists(tgt_abs):
            raise Fail(f"{where}.target already exists: {tgt_abs}\n"
                       f"  A pre-existing target-architecture directory means this migration may "
                       f"already be done. That is not a unit: report the existing directory, its "
                       f"arch-list row, its registry case and whether it passes today, and ask "
                       f"whether re-verification is what is wanted "
                       f"(references/01-frame.md section 3).")
        blocked_hits = scan_blocked_dtype(root, src)
        if blocked_hits:
            blocked.append((uid, src, blocked_hits))
        units.append({"id": uid, "source": src, "target": tgt, "note": u.get("note")})

    # Collected rather than raised inside the loop, so a mixed plan reports every offending unit
    # in one pass instead of one per re-run.
    if blocked:
        raise refuse_blocked_dtype(blocked)

    return {"version": 1, "request": request, "target_root": root, "refs": refs,
            "optimize": {"enabled": bool(opt.get("enabled", True))},
            "units": units}


# --- findings / changes ------------------------------------------------------------------

FINDINGS_KEYS = ("unit", "route", "counterpart", "contract", "type_stack", "routes",
                 "shared_components", "prove", "writable_paths", "notes")
PROVE_KEYS = ("build", "run", "cwd", "shape", "device", "stage")

# The frozen contract the CONFIRM gate reads. Free-form prose let two analysts describe the same
# operator in shapes the gate could not compare, so the load-bearing fields are fixed here and
# anything else the analyst wants to record rides alongside them.
CONTRACT_TENSOR_KEYS = ("name", "role", "dtype", "layout", "storage", "alias_of")
CONTRACT_REQUIRED = ("tensors", "output_region", "supported_domain", "zero_work", "golden")
GOLDEN_REQUIRED = ("function", "comparator", "compared_tensor", "compared_dtype", "compute_num",
                   "compute_num_read_from")

# The bare function name out of a prose `comparator` string. The lead writes that field as a
# sentence -- entry point, overload set ruled out, rtol reasoning -- and the entry point is the
# first thing in it that looks like a call.
COMPARATOR_CALL_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_:]*)\s*(?:<[^<>]*>\s*)?\(")
# One key per layer of `references/03-analyze.md` section 4, each naming the file it was read
# from -- a layer asserted from memory is the failure this catches.
TYPE_STACK_LAYERS = ("host", "dispatch", "block", "tilecopy_ab", "tilecopy_cd", "epilogue",
                     "scheduler_tiles", "kernel", "adapter")


def not_applicable(val):
    """A field may be declared inapplicable, but only out loud and with a reason."""
    return isinstance(val, dict) and isinstance(val.get("not_applicable"), str) \
        and bool(val["not_applicable"].strip()) and set(val) == {"not_applicable"}


def comparator_name(comparator):
    m = COMPARATOR_CALL_RE.search(comparator)
    if not m:
        raise Fail("findings.contract.golden.comparator must name the entry point in call "
                   f"form -- `golden::CompareData(result, expect, computeNum)` -- because "
                   f"`compute_num_read_from` is checked against it. Got: {comparator[:80]!r}")
    return m.group(1).split("::")[-1]


def check_compute_num_citation(g, root):
    """`compute_num` decides the tolerance band, so a wrong one is a silent false pass.

    Nothing downstream can catch it: `prove` compares the migrated example's evidence line
    against this number, so a mis-read source value makes both sides agree on the wrong rtol
    and `errors=0` means nothing. There is no way to recompute it -- it is whatever local the
    source hands its comparator -- so what is checked is the citation: the analyst names the
    line, and this opens it and requires the comparator's own name to be there. A fabricated
    or stale citation is refused; a real one has been read by two parties.
    """
    cite = g.get("compute_num_read_from")
    path, _, lineno = (cite or "").rpartition(":")
    if not isinstance(cite, str) or not path or not lineno.isdigit():
        raise Fail("findings.contract.golden.compute_num_read_from must be `<path>:<line>` -- "
                   "the source line that hands compute_num to the comparator. It selects the "
                   f"tolerance band, so it is the one contract number this engine checks "
                   f"against the tree instead of taking on trust. Got: {cite!r}")
    dest = contained(root, path, "findings.contract.golden.compute_num_read_from")
    if not os.path.isfile(dest):
        raise Fail(f"findings.contract.golden.compute_num_read_from cites {path}, which is "
                   f"not a file under the target root.")
    lines = read(dest).splitlines()
    n = int(lineno)
    # A call can wrap; +-2 lines is enough for every form seen and small enough that a
    # citation pointing at the wrong function still fails.
    window = "\n".join(lines[max(0, n - 3):n + 2])
    name = comparator_name(g["comparator"])
    if name not in window:
        raise Fail(f"findings.contract.golden.compute_num_read_from is {cite}, but {name!r} "
                   f"-- the comparator this contract froze -- does not appear within two "
                   f"lines of it. compute_num is read off the comparator call; cite the call, "
                   f"or correct `comparator` if the entry point is a different one.\n"
                   f"  what is there: {(lines[n - 1] if 0 < n <= len(lines) else '<past EOF>').strip()[:120]}")


def check_contract(c, root):
    if not isinstance(c, dict) or not c:
        raise Fail("findings.contract must be a non-empty object")
    missing = [k for k in CONTRACT_REQUIRED if k not in c]
    if missing:
        raise Fail(f"findings.contract is missing: {', '.join(missing)}. These are the fields "
                   f"the gate compares and Phase 7 copies from; a field that genuinely does not "
                   f'apply is written {{"not_applicable": "<reason>"}}, never omitted. Extra keys '
                   f"of your own (cli, aliasing, frozen_from, ...) are welcome beside them.")
    tensors = c["tensors"]
    if not not_applicable(tensors):
        if not isinstance(tensors, list) or not tensors:
            raise Fail("findings.contract.tensors must be a non-empty list of tensor objects "
                       '(or {"not_applicable": "<reason>"})')
        for i, t in enumerate(tensors):
            where = f"findings.contract.tensors[{i}]"
            closed(t, CONTRACT_TENSOR_KEYS, where)
            for key in ("name", "role", "dtype", "layout", "storage"):
                need_str(t, key, where)
            if "alias_of" not in t:
                raise Fail(f"{where}.alias_of is required: name the allocation this one may "
                           f"overlap, or null when the source asserts no overlap. Silence here "
                           f"is how an aliasing contract gets lost.")
            if t["alias_of"] is not None and not isinstance(t["alias_of"], str):
                raise Fail(f"{where}.alias_of must be a tensor name or null")
    for key in ("output_region", "supported_domain", "zero_work"):
        if not not_applicable(c[key]):
            need_str(c, key, "findings.contract")
    g = c["golden"]
    if not not_applicable(g):
        if not isinstance(g, dict):
            raise Fail("findings.contract.golden must be an object with "
                       f"{', '.join(GOLDEN_REQUIRED)}")
        closed(g, GOLDEN_REQUIRED, "findings.contract.golden")
        need_str(g, "function", "findings.contract.golden")
        need_str(g, "comparator", "findings.contract.golden")
        compared_tensor = need_str(g, "compared_tensor", "findings.contract.golden")
        compared_dtype = need_str(g, "compared_dtype", "findings.contract.golden")
        if compared_dtype != compared_dtype.strip():
            raise Fail("findings.contract.golden.compared_dtype must be the exact canonical "
                       "CATLASS_EVIDENCE.dtype string, with no leading or trailing whitespace.")
        # The evidence has one dtype label, so the gate must freeze which contractual tensor it
        # describes. Do not infer that label from `tensor.dtype`: source dtype prose can describe
        # storage and Cube operands while the golden legitimately uses upgraded precision.
        if not isinstance(tensors, list) \
                or sum(t["name"] == compared_tensor for t in tensors) != 1:
            raise Fail("findings.contract.golden.compared_tensor must exactly name one "
                       "findings.contract.tensors[].name. It binds the canonical evidence dtype "
                       "to the buffer the frozen comparator actually compares.")
        # The comparator's third argument selects the tolerance band, so passing the wrong local
        # variable silently moves rtol and still reports zero errors. Freezing it here is what
        # lets `prove` compare it against the number the migrated example actually printed.
        if type(g.get("compute_num")) is not int:
            raise Fail("findings.contract.golden.compute_num must be the integer the source "
                       "hands to its comparator at the frozen shape. It selects the tolerance "
                       "band rather than counting elements, so `prove` checks the evidence "
                       "line against it.")
        check_compute_num_citation(g, root)


def check_type_stack(ts):
    if not isinstance(ts, dict) or not ts:
        raise Fail("findings.type_stack must be a non-empty object")
    missing = [k for k in TYPE_STACK_LAYERS if k not in ts]
    if missing:
        raise Fail(f"findings.type_stack is missing layer(s): {', '.join(missing)}. Every layer "
                   f"of references/03-analyze.md section 4 needs an entry; a layer that is "
                   f'genuinely absent -- `epilogue` on a pure-cube unit -- is written '
                   f'{{"not_applicable": "<reason>"}}.')
    for key in TYPE_STACK_LAYERS:
        val = ts[key]
        where = f"findings.type_stack.{key}"
        if not_applicable(val):
            continue
        if not isinstance(val, dict):
            raise Fail(f"{where} must be an object with 'type' (the expanded type) and "
                       f"'read_from' (the file you read it in), or "
                       f'{{"not_applicable": "<reason>"}}')
        closed(val, ("type", "read_from"), where)
        need_str(val, "type", where)
        need_str(val, "read_from", where)


COUNTERPART_KEYS = ("suspect", "verdict", "evidence")
COUNTERPART_VERDICTS = ("is-counterpart", "not-counterpart")


def check_counterpart(f):
    """Validate the required counterpart finding; `null` is a complete answer."""
    if "counterpart" not in f:
        raise Fail('findings.counterpart is required. `null` is the answer when no target-arch '
                   'directory plausibly implements this operator; otherwise it is '
                   '{"suspect": "<dir>", "verdict": "is-counterpart|not-counterpart", '
                   '"evidence": "<type stack and contract, field by field>"}. A same-slug '
                   'directory is not evidence of a pair -- a TLA A5 example is not the '
                   'counterpart of a non-TLA A2 source -- and the human cuts the unit at the '
                   'gate with this in front of them, never the agent beforehand.')
    cp = f["counterpart"]
    if cp is None:
        return
    closed(cp, COUNTERPART_KEYS, "findings.counterpart")
    need_str(cp, "suspect", "findings.counterpart")
    if need_str(cp, "verdict", "findings.counterpart") not in COUNTERPART_VERDICTS:
        raise Fail("findings.counterpart.verdict must be one of "
                   f"{', '.join(COUNTERPART_VERDICTS)}")
    # A name match is only a suspect; the contract and type-stack comparison is the evidence.
    need_str(cp, "evidence", "findings.counterpart")


def check_findings(path, unit_id, root, surfaces=()):
    f = load_json(path, "findings")
    closed(f, FINDINGS_KEYS, "findings")
    if f.get("unit") != unit_id:
        raise Fail(f"findings.unit is {f.get('unit')!r}, expected {unit_id!r}")
    route = f.get("route")
    if route not in ROUTES:
        raise Fail(f"findings.route must be one of {', '.join(ROUTES)}, got {route!r}")

    check_counterpart(f)
    check_contract(f.get("contract"), root)
    check_type_stack(f.get("type_stack"))

    # The recommended route and every cheaper route require an evidence-backed verdict.
    # A dearer route is invalid while a cheaper route remains eligible.
    adjudicated = {}
    for i, r in enumerate(need_list(f, "routes", "findings", of=dict, allow_empty=False)):
        where = f"findings.routes[{i}]"
        closed(r, ("route", "verdict", "evidence", "diagnostic", "conditional_on"), where)
        rt = need_str(r, "route", where)
        if rt not in ROUTES:
            raise Fail(f"{where}.route must be one of {', '.join(ROUTES)}")
        if need_str(r, "verdict", where) not in ("eligible", "excluded", "needs-diagnostic"):
            raise Fail(f"{where}.verdict must be eligible, excluded or needs-diagnostic")
        need_str(r, "evidence", where)
        # `needs-diagnostic` means "only instantiating something can settle this". Without the
        # command that would settle it, the verdict is indistinguishable from "not investigated".
        if r["verdict"] == "needs-diagnostic":
            need_str(r, "diagnostic", f"{where} (verdict is needs-diagnostic, so)")
        # A conditional route records the shared component it depends on.
        if r.get("conditional_on") is not None:
            need_str(r, "conditional_on", where)
        adjudicated[rt] = r["verdict"]

    if route not in ROUTES:
        raise Fail(f"findings.route must be one of {', '.join(ROUTES)}")
    rung = ROUTES.index(route)
    required = ROUTES[:rung + 1]
    missing = [r for r in required if r not in adjudicated]
    if missing:
        raise Fail(f"findings.routes does not adjudicate: {', '.join(missing)}. Recommending "
                   f"{route!r} requires a verdict with evidence for it and for every cheaper "
                   f"class it supersedes ({', '.join(required)}); the dearer classes need "
                   f"none, because excluding them decides nothing.")
    cheaper_open = [r for r in ROUTES[:rung] if adjudicated.get(r) == "eligible"]
    if cheaper_open:
        raise Fail(f"findings.route is {route!r}, but {', '.join(cheaper_open)} "
                   f"{'is' if len(cheaper_open) == 1 else 'are'} marked eligible. A dearer "
                   f"class is admissible only once the cheaper ones are excluded by precise "
                   f"source evidence -- either exclude "
                   f"{'it' if len(cheaper_open) == 1 else 'them'} with evidence, or recommend "
                   f"the cheapest eligible class.")
    # A recommendation ANALYZE cannot prove from source is admissible, because ANALYZE may not
    # build: `needs-diagnostic` plus the naming of the command that settles it is an honest
    # analysis, and forcing it to claim `eligible` instead is what produced false confidence at
    # the gate. `excluded` remains a contradiction.
    if adjudicated.get(route) not in ("eligible", "needs-diagnostic"):
        raise Fail(f"findings.route is {route!r} but findings.routes marks it "
                   f"{adjudicated.get(route)!r}; the recommended class must be eligible, or "
                   f"needs-diagnostic with the diagnostic command that settles it")

    # An empty shared-component list is valid; omitting the field is not.
    if "shared_components" not in f:
        raise Fail("findings.shared_components is required. An empty list is a legal answer -- "
                   "this unit lands no shared edit -- but the key must be there, because the "
                   "gate's blast-radius ledger is computed from it.")
    for i, s in enumerate(need_list(f, "shared_components", "findings", of=dict)):
        where = f"findings.shared_components[{i}]"
        closed(s, ("path", "symbol", "kind", "why", "consumers_of"), where)
        contained(root, need_str(s, "path", where), f"{where}.path")
        need_str(s, "symbol", where)
        kind = need_str(s, "kind", where)
        if kind not in SHARED_KINDS:
            raise Fail(f"{where}.kind must be one of {', '.join(SHARED_KINDS)}. A component "
                       f"this unit only consumes is not a row: the ledger derives consumers "
                       f"from the tree, so declaring reuse counted nothing it could not "
                       f"already measure.")
        # `why` is rendered at the migration gate.
        need_str(s, "why", where)
        # Name the existing symbol whose consumers must keep building.
        if kind == "generalize":
            need_str(s, "consumers_of", f"{where} (kind is generalize, so)")
        elif s.get("consumers_of") is not None:
            raise Fail(f"{where}.consumers_of must be absent on an `add` row: the symbol is "
                       f"new, so it has no existing consumers and no regression obligation.")

    # Every nonterminal unit must include every required registration surface discovered in
    # PROFILE in its shared-component ledger. The gate must present the complete write blast
    # radius and its executable order, so refuse any analysis that omits a required surface.
    if route != ROUTE_TERMINAL and surfaces:
        declared = {s["path"] for s in f["shared_components"]}
        missing = [p for p in surfaces if p not in declared]
        if missing:
            raise Fail("findings.shared_components does not declare this tree's required "
                       f"registration surface(s): {', '.join(missing)}.\n"
                       "  profile.registration.surfaces marks them required, so every unit "
                       "touches them and the lead lands each one AFTER that unit's target "
                       "directory exists. A unit that omits them reaches the gate with an "
                       "understated blast radius and an execution order that cannot be run.")

    # `redesign` terminates the unit, so it owes no build plan -- there is nothing to build.
    if route != ROUTE_TERMINAL:
        prove = f.get("prove")
        if not isinstance(prove, dict):
            raise Fail("findings.prove must be an object holding the discovered build and "
                       "run commands; `prove` cannot execute what nobody discovered")
        closed(prove, PROVE_KEYS, "findings.prove")
        need_list(prove, "build", "findings.prove", allow_empty=False)
        need_list(prove, "run", "findings.prove", allow_empty=False)
        # `cwd` defaulted to "." and `shape`/`device` were never required at all, so a findings
        # file could reach Phase 7 without saying where to run, at what shape, or whether a
        # capable device was ever seen -- and Phase 7 is where those three become the claim.
        need_str(prove, "cwd", "findings.prove")
        contained(root, prove["cwd"], "findings.prove.cwd")
        shape = prove.get("shape")
        if not isinstance(shape, list) or not shape or not all(type(v) is int for v in shape):
            raise Fail("findings.prove.shape must be a non-empty list of integers: the frozen "
                       "contract's own shape, which is the one PROVEN will be claimed for. "
                       "`prove` compares the evidence line's own `shape` against it elementwise, "
                       "so a debug case recorded here becomes the claim.")
        need_str(prove, "device", "findings.prove")
        # What has to exist on the device box before the run. Optional here and required at
        # `prove` time when that box is not this one: the build is local, so a remote device
        # has none of this unit's binary until something carries it over.
        stage = prove.get("stage")
        if stage is not None:
            if not isinstance(stage, list) or not stage \
                    or not all(isinstance(p, str) and p for p in stage):
                raise Fail("findings.prove.stage must be a non-empty list of tree-relative "
                           "paths: the artifact the profile discovered plus every input the "
                           "run argv reads. Omit the key entirely when the device is on this "
                           "machine; an empty list says nothing and reads like an oversight.")
            for i, p in enumerate(stage):
                contained(root, p, f"findings.prove.stage[{i}]")

        paths = need_list(f, "writable_paths", "findings", allow_empty=False)
        for i, p in enumerate(paths):
            contained(root, p, f"findings.writable_paths[{i}]")
    return f


# --- optimization artifacts ----------------------------------------------------------------

SCREEN_KEYS = ("unit", "applicable", "rows", "strategy", "manifest", "baseline", "notes")
# Five applicability rows, cheapest and most fundamental first. `gemm_family` comes before the
# rest because rows 4 and 5 have no subject without it: a Gemv/Conv/attention stack declares no
# `BlockMmad` to trace, and the whole rewrite binds to `L1TileShape::M`.
SCREEN_ROWS = ("gemm_family", "epilogue", "non_tla", "block_mmad", "block_epilogue")
MANIFEST_KEYS = ("path", "action", "tier")
MANIFEST_ACTIONS = ("add", "modify")
SAMPLE_KEYS = ("config", "task_us", "source")
SAMPLE_CONFIGS = ("baseline", "l0c_to_ub")


def us_stat(v):
    """Return sample count, range, and median for one value or a value list."""
    xs = sorted(v) if isinstance(v, list) else [v]
    n = len(xs)
    return {"n": n, "min": xs[0], "max": xs[-1],
            "median": xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2}


def us_text(v):
    """Format one sample or a median with its sample count and range."""
    s = us_stat(v)
    return f"{s['median']:g}" if s["n"] == 1 \
        else f"{s['median']:g} (n={s['n']}, {s['min']:g}-{s['max']:g})"


def check_sample(sample, where, udir):
    """Validate a profiler duration and the run-directory output that supplied it."""
    closed(sample, SAMPLE_KEYS, where)
    if need_str(sample, "config", where) not in SAMPLE_CONFIGS:
        raise Fail(f"{where}.config must be one of {', '.join(SAMPLE_CONFIGS)}")
    us = sample.get("task_us")
    vals = us if isinstance(us, list) else [us]
    if not vals or any(type(x) not in (int, float) or x <= 0 for x in vals):
        raise Fail(f"{where}.task_us must be a positive number -- the `Task Duration(us)` "
                   f"msprof reported for this configuration -- or a non-empty list of them, one "
                   "per repeat of the same measurement.")
    rel = need_str(sample, "source", where)
    dest = contained(udir, rel, f"{where}.source")
    hits = sorted(glob.glob(os.path.join(dest, "OPPROF_*", "OpBasicInfo.csv")))
    if len(hits) != len(vals):
        raise Fail(f"{where}.source is {rel!r}, which holds {len(hits)} "
                   f"`OPPROF_*/OpBasicInfo.csv` against {len(vals)} transcribed duration(s). "
                   f"One profiler output per number, in one directory: point every repeat's "
                   f"`msprof op --output=` at the same `units/<id>/logs/prof-<config>-<n>/` and "
                   f"record that directory here, relative to the unit directory.")


def check_screen(path, unit_id, root, udir):
    """Validate the SCREEN artifact: does the rewrite apply to this migrated unit, and how.

    The five rows are read from the **migrated** unit -- not from the A2 source, not from a
    plan. The rewrite is applied to what was built, so what was built is what gets screened.
    That is also why this is its own phase rather than part of ANALYZE: at ANALYZE the target
    directory does not exist, and `init` refuses a plan whose target already does.

    Validated: that every row carries its reading, and that an applicable unit names both a
    strategy and the file manifest that becomes its write grant once the gate authorizes it.
    Deliberately not validated: whether any of it is true. A fabricated row passes here. The
    optimization gate is what reads it, which is the same division of labour `check_findings`
    has with the migration gate.
    """
    s = load_json(path, "screen")
    closed(s, SCREEN_KEYS, "screen")
    if s.get("unit") != unit_id:
        raise Fail(f"screen.unit is {s.get('unit')!r}, expected {unit_id!r}")

    applicable = s.get("applicable")
    if not isinstance(applicable, bool):
        raise Fail("screen.applicable must be a JSON boolean: the verdict of the five rows")

    rows = s.get("rows")
    if not isinstance(rows, dict):
        raise Fail("screen.rows must be an object carrying all five applicability rows")
    closed(rows, SCREEN_ROWS, "screen.rows")
    for name in SCREEN_ROWS:
        # Every row carries its reading, including the ones that passed. A row recorded as a
        # bare verdict is the failure this prevents: at the gate a human authorizes a rewrite
        # of a working operator, and a verdict without the source line behind it is not a
        # finding, it is an assertion.
        need_str(rows, name, "screen.rows")

    strategy, manifest = s.get("strategy"), s.get("manifest")
    if applicable:
        if strategy not in STRATEGIES:
            raise Fail(f"screen.strategy must be one of {', '.join(STRATEGIES)} on an "
                       f"applicable unit: `coexist` keeps the proven baseline compilable "
                       f"behind the example's opt-in switch, `replace` removes the old path. "
                       f"What is written here is a proposal -- the human at the optimization "
                       f"gate decides, and `replace` is the one irreversible act in this "
                       f"workflow.")
        for i, m in enumerate(need_list(s, "manifest", "screen", of=dict, allow_empty=False)):
            where = f"screen.manifest[{i}]"
            closed(m, MANIFEST_KEYS, where)
            contained(root, need_str(m, "path", where), f"{where}.path")
            if need_str(m, "action", where) not in MANIFEST_ACTIONS:
                raise Fail(f"{where}.action must be one of {', '.join(MANIFEST_ACTIONS)}")
            if m.get("tier") not in (1, 2):
                raise Fail(f"{where}.tier must be 1 or 2. Tier 1 is operator-agnostic and the "
                           f"lead lands it once per tree; tier 2 is this unit's own and an "
                           f"Optimizer may write it.")
    else:
        if strategy is not None:
            raise Fail("screen.strategy must be null when the unit is not applicable: there "
                       "is no second data path to choose between.")
        if manifest:
            raise Fail("screen.manifest must be empty when the unit is not applicable: "
                       "nothing will be written, so nothing is granted.")

    base = s.get("baseline")
    if not isinstance(base, dict):
        raise Fail("screen.baseline must be an object holding this migrated unit's own "
                   "measured duration. Every migrated unit is profiled, applicable or not: "
                   "without it the report can say nothing about the performance of the thing "
                   "the campaign just built.")
    check_sample(base, "screen.baseline", udir)
    if base["config"] != "baseline":
        raise Fail('screen.baseline.config must be "baseline"')
    return s


OPTIMIZE_KEYS = ("unit", "mode", "paths", "profile", "notes")


def check_optimize(path, unit_id, root, udir, screen):
    """Validate the APPLY artifact: what the rewrite wrote, in which mode, and how it timed.

    What is absent is as deliberate as what is here. No accuracy field: the rewrite is the
    example's default build, so `prove` builds it and `prove` is the accuracy record -- a
    field here would be a second, weaker copy of a verdict the tool already produces from an
    execution it ran itself. No ranking and no adoption flag: which data path the tree keeps
    was settled by a human at the optimization gate before any of this was written, so
    nothing here may re-decide it on a number.
    """
    o = load_json(path, "optimize")
    closed(o, OPTIMIZE_KEYS, "optimize")
    if o.get("unit") != unit_id:
        raise Fail(f"optimize.unit is {o.get('unit')!r}, expected {unit_id!r}")

    if o.get("mode") not in COPY_MODES:
        raise Fail(f"optimize.mode must be one of {', '.join(COPY_MODES)}. The mode is not "
                   f"cosmetic: `SPLIT_M` halves M across the two AIV sub-blocks and requires "
                   f"the dual-flag protocol and a halved epilogue tile, `NO_SPLIT` requires "
                   f"neither. The UB budget is what decides it.")

    # The authorized manifest is the whole write grant, and the only one. A path outside it
    # was not on the packet the human approved, so it was not authorized.
    declared = [m["path"] for m in screen.get("manifest") or []]
    granted = {contained(root, p, "screen.manifest") for p in declared}
    for i, p in enumerate(need_list(o, "paths", "optimize", allow_empty=False)):
        dest = contained(root, p, f"optimize.paths[{i}]")
        if dest not in granted:
            raise Fail(f"optimize.paths[{i}] is {p!r}, which is not in the file manifest this "
                       f"unit declared at SCREEN and the human authorized at the optimization "
                       f"gate.\n  authorized: {', '.join(declared) or '<none>'}\n"
                       f"  Re-screen the unit and re-present the packet if the file is "
                       f"genuinely needed; a grant is not widened after the gate that read it.")
        # `contained` fences a path; it does not require one to be there. Without this, a unit
        # reached OPTIMIZED with no rewrite on disk at all -- every check above it validates a
        # declaration, so a declaration was all it took. The tree is read here because that is
        # the only place the fact lives, and this tool already reads the tree where it must:
        # `check_sample` globs the profiler's own output directory for the same reason.
        if not os.path.exists(dest):
            raise Fail(f"optimize.paths[{i}] is {p!r}, which does not exist under the target "
                       f"root ({dest}). The rewrite declared a file it did not write, or it "
                       f"wrote it somewhere else -- either way what landed is not what this "
                       f"artifact says landed. Fix the tree or fix the path; do not remove "
                       f"the entry to make the check pass.\n"
                       f"  Honest limit: existence is not authorship. A file that was already "
                       f"there passes this whether or not the rewrite touched it.")

    samples = need_list(o, "profile", "optimize", of=dict, allow_empty=False)
    for i, sample in enumerate(samples):
        check_sample(sample, f"optimize.profile[{i}]", udir)
    if not any(sample["config"] == "l0c_to_ub" for sample in samples):
        raise Fail("optimize.profile carries no `l0c_to_ub` sample. The rewritten path's own "
                   "duration is the one measurement this phase exists to produce.")
    if screen["strategy"] == "coexist" and \
            not any(sample["config"] == "baseline" for sample in samples):
        raise Fail("optimize.profile carries no `baseline` sample for the authorized `coexist` "
                   "strategy. Re-measure the still-buildable baseline in the same session as "
                   "the rewrite and record its non-empty profiler result beside `l0c_to_ub`.")
    return o


def check_apply_ready(row, root, udir, access):
    """What both APPLY-side checks stand on: the gate authorized this unit, and the screen it
    authorized still says what it said then.

    `check --phase applied` records that the rewrite landed and `check --phase optimized`
    records that the landed rewrite re-passed the golden. Both are refusals about the same
    authorization, and re-deriving it in two branches is how the two drift apart.
    """
    if RANK[row["phase"]] < RANK["OPT_AUTHORIZED"]:
        raise Fail(f"unit is at {row['phase']}; the rewrite lands only once the optimization "
                   f"gate has authorized it. Screen it with `check --phase screened`, then "
                   f"`gate --phase optimize` and `confirm --phase optimize`.")
    proof, issue = current_proof(row, access)
    if proof is None:
        raise Fail("accuracy is NOT ESTABLISHED: " + issue + ". Re-run `prove` on the "
                   "declared matching Ascend950 hardware before applying or optimizing.")
    s = check_screen(os.path.join(udir, "screen.json"), row["id"], root, udir)
    if not s["applicable"]:
        raise Fail("screen.json marks this unit not applicable, so there is nothing to apply. "
                   "Its result for these phases is the screen itself plus the baseline "
                   "measurement, and both are reported.")
    if s["strategy"] != row["strategy"]:
        raise Fail(f"screen.strategy is {s['strategy']!r}, but the optimization gate "
                   f"authorized {row['strategy']!r}. What the human approved is what the log "
                   f"records; re-present the packet and re-confirm rather than editing the "
                   f"artifact underneath the decision.")
    # The authorization binds both the selected strategy and its file manifest.
    authorized = row.get("opt_manifest")
    if authorized is None:
        print(f"  note: unit {row['id']} has no recorded authorization manifest, so its write "
              f"grant cannot be checked. Re-present with `gate --phase optimize` and re-confirm.")
    else:
        live = sorted(m["path"] for m in s.get("manifest") or [])
        if live != sorted(authorized):
            added = [p for p in live if p not in authorized]
            gone = [p for p in authorized if p not in live]
            raise Fail(
                f"screen.manifest no longer matches the file list the optimization gate "
                f"authorized for this unit"
                + (f"; added since: {', '.join(added)}" if added else "")
                + (f"; removed since: {', '.join(gone)}" if gone else "")
                + ". The manifest IS the write grant, so an edit after approval authorizes "
                  "writes nobody agreed to. Re-present the packet with `gate --phase "
                  "optimize`, have the human read the corrected list, and re-confirm.")
    return s


# --- tree diff -----------------------------------------------------------------------------

# Written by `init`: whatever was already dirty before this campaign existed. The scope check
# subtracts it, so a campaign started on a tree carrying unrelated local work reports that work
# as what it is rather than accusing an implementer of it.
TREE_BASELINE = "tree_baseline.txt"


def git_dirty(root):
    """Tree-relative paths `git status --porcelain` reports, rename halves included."""
    proc = subprocess.run(["git", "-C", root, "status", "--porcelain", "-z"],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise Fail(f"`git status` failed in {root}: "
                   f"{proc.stderr.strip() or f'exit {proc.returncode}'}. The implemented-scope "
                   f"check diffs the tree instead of reading a self-report, so it needs a git "
                   f"work tree here.")
    paths, recs, i = [], proc.stdout.split("\0"), 0
    while i < len(recs):
        rec = recs[i]
        i += 1
        if len(rec) < 4:
            continue
        paths.append(rec[3:])
        if "R" in rec[:2] or "C" in rec[:2]:
            # `-z` puts a rename/copy source in the record that follows.
            if i < len(recs):
                paths.append(recs[i])
                i += 1
    return paths


def load_tree_baseline(run_dir):
    path = os.path.join(run_dir, TREE_BASELINE)
    return [p for p in read(path).splitlines() if p.strip()] \
        if os.path.exists(path) else []


def write_tree_baseline(root, run_dir):
    """Snapshot what was already dirty, so no unit is charged with it later."""
    try:
        paths = git_dirty(root)
    except Fail:
        # Not a git work tree. `check --phase implemented` says so at the point it matters;
        # refusing `init` over it would be an environment gate, which this workflow does not do.
        paths = []
    write(os.path.join(run_dir, TREE_BASELINE), "\n".join(sorted(paths)) + "\n")
    return paths


def check_tree_scope(root, run_dir, plan, uid, findings, shared_paths):
    """Validate actual tree changes against the unit's authorized paths."""
    baseline = set(load_tree_baseline(run_dir))
    real_root = os.path.realpath(root)

    def resolve(rel):
        return os.path.realpath(os.path.join(real_root, rel.rstrip("/")))

    mine = [resolve(p) for p in findings.get("writable_paths") or []]
    # Other units' target directories: in a cluster campaign the earlier units' work is
    # legitimately sitting in the tree while this one is checked.
    others = [resolve(u["target"]) for u in plan["units"] if u["id"] != uid]
    # The skill's own directory, when it happens to live inside the tree it is migrating: the
    # engine's source is the tool, never the campaign's output, and a session that refines the
    # skill while a campaign runs against the same checkout would otherwise charge every
    # documentation edit to whichever unit is being checked.
    lead = [resolve(p) for p in shared_paths] + [resolve(WORK_DIR),
                                                 os.path.dirname(os.path.dirname(
                                                     os.path.abspath(__file__)))]

    def inside(real, bases):
        return any(real == b or real.startswith(b + os.sep) for b in bases)

    stray, touched = [], False
    for rel in git_dirty(root):
        if rel in baseline:
            continue
        real = resolve(rel)
        if inside(real, mine):
            touched = True
        elif not inside(real, others + lead):
            stray.append(rel)
    granted = ", ".join(findings.get("writable_paths") or ["<none>"])
    if stray:
        raise Fail("`git status` reports changes this unit's authorization does not cover:\n  "
                   + "\n  ".join(stray)
                   + f"\n  granted to this unit: {granted}"
                   + "\n  lead-owned shared paths from the ledger: "
                   + (", ".join(shared_paths) or "<none>")
                   + "\n  Either the change belongs to another unit, or it is a shared surface "
                     "the ledger never named -- and a grant is widened by re-presenting the "
                     "gate, never here.")
    if not touched:
        raise Fail(f"`git status` reports no change inside this unit's writable_paths "
                   f"({granted}). An implemented unit changed something: either the "
                   f"implementer wrote nowhere, or it wrote where the grant does not reach.")


# --- run directory -----------------------------------------------------------------------

def resolve_run_dir(arg, target_root):
    """Return a run directory fenced under `<target_root>/.agents-work/`."""
    work = os.path.join(os.path.realpath(target_root), WORK_DIR)
    if arg:
        resolved = os.path.realpath(os.path.abspath(arg))
        if resolved != work and not resolved.startswith(work + os.sep):
            raise Fail(
                f"--run-dir must be inside {work}, got {resolved}.\n"
                f"  Everything this tool generates belongs under `{WORK_DIR}/`: that is the one "
                f"path the ignore rule covers, so state written elsewhere escapes the audit "
                f"trail and can dirty a tree that never consented.\n"
                f"  Either omit --run-dir for the timestamped default, or name one inside the "
                f"fence, e.g. --run-dir {os.path.join(work, SKILL_NAME, 'my-campaign')}")
        # ...but not inside the cache subtree. `resolve_ref` owns every path under
        # `<work>/.cache/refs/` and rmtrees any entry there that is not a valid checkout at the
        # pin -- which a run directory is not -- so a campaign placed there is silently
        # destroyed by the next `refs`: exit 0, no warning, plan.json and events.jsonl gone.
        # That is total loss of the one artifact this design exists to protect. The wall is at
        # `.cache` rather than `.cache/refs` so nothing can tangle a campaign with the cache
        # even where deletion would not reach it.
        cache = os.path.dirname(os.path.realpath(cache_root(target_root)))
        if resolved == cache or resolved.startswith(cache + os.sep):
            raise Fail(
                f"--run-dir must not be inside {cache}, got {resolved}.\n"
                f"  That subtree belongs to `refs`, which deletes anything in it that is not a "
                f"checkout at the pin -- this campaign would be destroyed without warning.\n"
                f"  Name one beside it instead, e.g. --run-dir "
                f"{os.path.join(work, SKILL_NAME, 'my-campaign')}")
        return resolved
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return os.path.join(work, SKILL_NAME, stamp)


def ensure_ignored(target_root, run_dir):
    """Add one ignore line when the default run directory lands in a git work tree. The tool
    chose the path, so keeping it out of `git status` is the tool's job. Asks git rather than
    scanning .gitignore, so global excludes and an existing rule are honoured.

    The rule goes to `$GIT_DIR/info/exclude`, never to the tracked `.gitignore`: writing to a
    tracked file would dirty the user's tree before the CONFIRM gate, which is the one thing
    this workflow promises not to do. `git rev-parse --git-path` resolves the right file in a
    linked worktree too, where it lands in the shared common dir -- every worktree of the repo
    then ignores `.agents-work/`, which is what all of them want.
    """
    work = os.path.join(os.path.abspath(target_root), WORK_DIR)
    try:
        if os.path.commonpath([os.path.abspath(run_dir), work]) != work:
            return None
    except ValueError:
        return None

    def git(*argv):
        return subprocess.run(["git", "-C", target_root, *argv], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL).returncode
    try:
        if git("rev-parse", "--is-inside-work-tree") != 0:
            return None
        # Trailing slash matters: this runs before the directory exists, and check-ignore
        # answers 1 for a bare name that is not on disk yet but 0 for `name/`. Getting it
        # wrong appends a duplicate rule on every campaign.
        if git("check-ignore", "-q", f"{WORK_DIR}/") == 0:
            return None
        found = subprocess.run(["git", "-C", target_root, "rev-parse", "--git-path",
                                "info/exclude"], stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, text=True)
        if found.returncode != 0 or not found.stdout.strip():
            return None
    except OSError:
        return None
    path = found.stdout.strip()
    if not os.path.isabs(path):
        path = os.path.join(os.path.abspath(target_root), path)
    old = read(path) if os.path.exists(path) else ""
    lead = "" if not old or old.endswith("\n") else "\n"
    write(path, f"{old}{lead}\n# Agent skill working data: campaign state, artifacts, logs,\n"
                f"# and the pinned reference checkouts under .cache/. Never a deliverable.\n"
                f"{WORK_DIR}/\n")
    return path


def load_run(run_dir):
    run_dir = os.path.abspath(run_dir)
    plan_path = os.path.join(run_dir, "plan.json")
    if not os.path.exists(plan_path):
        raise Fail(f"no plan.json in {run_dir}; is that a run directory created by `init`?")
    plan = load_json(plan_path, "plan")
    return run_dir, plan


def unit_dir(run_dir, uid):
    return os.path.join(run_dir, "units", uid)

# Hardware access, declared beside the scope and separately from it. Zero, one or both sides
# is legal: a box with only A2 can still cross-compile 3510 and measure the A2 half of a
# comparison, and a box with neither can still compile. Access never blocks a campaign -- it
# decides which *claims* the campaign is allowed to make, and everything unreachable is
# reported rather than refused.
ACCESS_SIDES = ("a2", "a5")
ACCESS_KEYS = ("reachable", "arch", "soc", "host", "device", "notes", "transport")

# An ssh transport binds a reachable remote side to the declared connection.
TRANSPORT_KINDS = ("local", "ssh")
TRANSPORT_KEYS = ("kind", "host", "user", "port", "identity_file", "password",
                  "password_env", "workdir", "ssh_options")

# Added to every ssh/scp invocation before the declaration's own `ssh_options`, so any of them
# can be overridden. `BatchMode=yes` is added only for key auth: it disables every prompt,
# which is what turns a missing key into a fast failure instead of a hang -- and what would
# make password auth impossible.
SSH_DEFAULT_OPTS = ("ConnectTimeout=10", "StrictHostKeyChecking=accept-new")

# The architecture every unit in this workflow migrates *to*. It is the one arch value the
# engine may name, because it is what the skill is for -- every other target-specific fact is
# discovered into profile.json. `2201` is the source side and appears only in access.json.
TARGET_ARCH = "3510"

# A reachable declaration is a measured FRAME fact, not a device probe that `prove` may
# reinterpret. The architecture tags are deliberately strings because they are the exact values
# carried into the proof witness and report.
ACCESS_ARCH = {"a2": "2201", "a5": "3510"}
SOC_FAMILY_MARKERS = {
    "a2": ("910", "atlas a2", "atlasa2", "atlas a3", "atlasa3", "2201"),
    "a5": ("950", "ascend950", "a5", "3510"),
}


def access_identity_issue(sec, side):
    """Return the measured-identity defect for one declared side, if it has one.

    Strict operational loading raises this exact result. Lenient reporting uses the same result to
    preserve an old declaration as a reportable, non-authorizing fact rather than manufacturing the
    fields needed to make it valid.
    """
    where = f"access.{side}"
    if not isinstance(sec, dict):
        return f"{where} must be an object"
    reported = sec.get("_reporting_access_issue")
    if reported is not None:
        return reported
    if not isinstance(sec.get("reachable"), bool):
        return (f"{where}.reachable must be true or false -- an explicit answer, because "
                "'unknown' and 'no' lead to different reports")
    if not sec["reachable"]:
        return None
    arch = sec.get("arch")
    if not isinstance(arch, str) or not arch.strip():
        return f"{where}.arch must be a non-empty string"
    if arch != ACCESS_ARCH[side]:
        return f"{where}.arch must be exactly {ACCESS_ARCH[side]!r}, got {arch!r}"
    soc = sec.get("soc")
    if not isinstance(soc, str) or not soc.strip():
        return f"{where}.soc must be a non-empty string"
    if not any(marker in soc.lower() for marker in SOC_FAMILY_MARKERS[side]):
        family = "/".join(SOC_FAMILY_MARKERS[side])
        return f"{where}.soc must identify the {side} family ({family}), got {soc!r}"
    for key in ("host", "device"):
        value = sec.get(key)
        if not isinstance(value, str) or not value.strip():
            return f"{where}.{key} must be a non-empty string"
    return None


def load_access(run_dir, strict=True):
    """Load declared access strictly for operations or leniently for historic reporting.

    Only `status` and `report` may select lenient loading. They retain old declarations as
    non-authorizing evidence so the report can name the measured repair and re-proof still owed;
    every command that can act on the tree, a device, or an optimization gate remains strict.
    """
    path = os.path.join(run_dir, "access.json")
    if not os.path.exists(path):
        return {s: {"reachable": False,
                    "notes": "no access declared; assumed unreachable"} for s in ACCESS_SIDES}
    doc = load_json(path, "access")
    if not isinstance(doc, dict):
        if strict:
            raise Fail("access.json must be an object with an `a2` and/or `a5` section")
        return {s: {"_reporting_access_issue":
                    "access.json must be an object with an `a2` and/or `a5` section"}
                for s in ACCESS_SIDES}
    out = {}
    for side in ACCESS_SIDES:
        sec = doc.get(side)
        if sec is None:
            out[side] = {"reachable": False, "notes": "not declared"}
            continue
        if not isinstance(sec, dict):
            if strict:
                raise Fail(f"access.{side} must be an object")
            out[side] = {"_reporting_access_issue": f"access.{side} must be an object"}
            continue
        if strict:
            closed(sec, ACCESS_KEYS, f"access.{side}")
            issue = access_identity_issue(sec, side)
            if issue:
                raise Fail(issue)
            load_transport(sec, f"access.{side}")
        out[side] = sec
    return out


def hardware_identity(side):
    """The exact FRAME declaration that binds a target-device proof."""
    return {key: side[key] for key in ("arch", "soc", "host", "device")}


def access_for_arch(access, arch):
    """Which declared side serves a target architecture. `arch` is the profile's own value."""
    return access.get("a2" if str(arch) == "2201" else "a5")

def current_proof(row, access):
    """Return the current hardware-bound proof, or why the rank's old proof cannot authorize.

    Phase rank is historical and intentionally monotonic. Accuracy is current only when the last
    proof event has not been revoked and its identity is the exact reachable A5 declaration; an
    old `unit.proven` event therefore cannot gain authority merely by carrying a high rank.
    """
    side = access_for_arch(access, TARGET_ARCH)
    declaration_issue = access_identity_issue(side, "a5")
    if declaration_issue:
        return None, ("the declared Ascend950 identity is incomplete or invalid: "
                      + declaration_issue
                      + "; repair `access.json` only from measured hardware, then re-run "
                      "`prove`")
    if not side["reachable"]:
        return None, "no reachable declared Ascend950 device can bind the proof"
    failed = row.get("prove_failed")
    if failed:
        detail = f"a later prove attempt failed at {failed.get('at') or 'an unknown stage'}"
        if failed.get("cause") == "environment":
            detail += (" (environment: "
                       + ", ".join(failed.get("missing") or ["toolchain prerequisite"])
                       + " absent)")
        return None, detail
    proof = row.get("proof")
    if not isinstance(proof, dict) or proof.get("errors") != 0:
        return None, "there is no passing `unit.proven` event"
    hardware = proof.get("hardware")
    if not isinstance(hardware, dict):
        return None, ("the latest `unit.proven` has no declared hardware identity; its old "
                      "accuracy is not current, so re-run `prove` on the measured declared "
                      "Ascend950 device")
    if hardware != hardware_identity(side):
        return None, ("the latest `unit.proven` hardware identity does not match the declared "
                      "Ascend950 identity; repair `access.json` only if measurement shows the "
                      "declaration is wrong, then re-run `prove`")
    return proof, None


def load_transport(sec, where):
    """Validate one side's transport block; return it, or None when the device is local.

    `kind: "local"` and an absent block are the same answer, spelled two ways: the second is
    the default, the first is how someone says "that `host` string means this box" out loud.
    """
    tr = sec.get("transport")
    if tr is None:
        return None
    if not isinstance(tr, dict):
        raise Fail(f"{where}.transport must be an object")
    closed(tr, TRANSPORT_KEYS, f"{where}.transport")
    kind = need_str(tr, "kind", f"{where}.transport")
    if kind not in TRANSPORT_KINDS:
        raise Fail(f"{where}.transport.kind must be one of {', '.join(TRANSPORT_KINDS)}")
    if kind == "local":
        return None
    need_str(tr, "host", f"{where}.transport")
    workdir = need_str(tr, "workdir", f"{where}.transport")
    if not workdir.startswith("/"):
        raise Fail(f"{where}.transport.workdir must be an absolute path on that box. It is the "
                   f"one directory this campaign stages into and runs in there -- the remote "
                   f"counterpart of {WORK_DIR}/ here -- and a relative path lands wherever the "
                   f"login shell happens to start, which is how a tree gets littered.")
    port = tr.get("port")
    if port is not None and (type(port) is not int or not 0 < port < 65536):
        raise Fail(f"{where}.transport.port must be an integer between 1 and 65535")
    opts = tr.get("ssh_options")
    if opts is not None and not (isinstance(opts, list)
                                 and all(isinstance(o, str) for o in opts)):
        raise Fail(f"{where}.transport.ssh_options must be a list of `Key=Value` strings, each "
                   f"passed as one `-o` after the defaults so it overrides them")
    if tr.get("password") is not None and tr.get("password_env") is not None:
        raise Fail(f"{where}.transport declares both `password` and `password_env`. Pick one: "
                   f"two secrets cannot both be the one this connection uses.")
    for key in ("user", "identity_file", "password", "password_env"):
        if tr.get(key) is not None and not isinstance(tr[key], str):
            raise Fail(f"{where}.transport.{key} must be a string")
    return tr


def transport_of(side):
    """The ssh transport for a declared side, or None when its device is on this machine."""
    if not isinstance(side, dict):
        return None
    tr = side.get("transport")
    return tr if isinstance(tr, dict) and tr.get("kind") == "ssh" else None


# Spellings that mean "here" without asking anything. Deliberately syntactic: this decides
# whether a declaration naming a box *and* declaring no transport may be read as local, and a
# DNS lookup would answer a different question -- reachability, not identity.
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1", "0.0.0.0")


def host_is_this_machine(host):
    if not host:
        return True
    h = host.strip().lower()
    if h in LOCAL_HOSTS:
        return True
    full = socket.gethostname().lower()
    return h in {full, full.split(".")[0]}


def ssh_password(tr):
    """The password for this connection, or None.

    Returned apart from the argv and handed to `sshpass -e` through the environment, so it
    never reaches a command line, a log header, `ps`, an event or a printed summary.
    """
    if tr.get("password") is not None:
        return tr["password"]
    name = tr.get("password_env")
    if not name:
        return None
    pw = os.environ.get(name)
    if pw is None:
        raise Fail(f"the transport declares password_env={name!r} and that variable is not set "
                   f"in this environment. Export it for this run, or declare an "
                   f"`identity_file` and use key authentication.")
    return pw


def ssh_prefix(tr):
    """`(argv prefix, environment additions)` common to every ssh and scp invocation."""
    pw = ssh_password(tr)
    if pw is None:
        return [], {}
    if not shutil.which("sshpass"):
        raise Fail("this transport authenticates with a password, which needs `sshpass` on "
                   "PATH, and there is none here. Install it, or declare an `identity_file` "
                   "and use key authentication. Either way the compile outcome stands: this "
                   "refuses the device run, never the build.")
    return ["sshpass", "-e"], {"SSHPASS": pw}


def ssh_flags(tr, batch):
    flags = []
    for opt in SSH_DEFAULT_OPTS:
        flags += ["-o", opt]
    if batch:
        flags += ["-o", "BatchMode=yes"]
    for opt in tr.get("ssh_options") or []:
        flags += ["-o", opt]
    if tr.get("identity_file"):
        flags += ["-i", tr["identity_file"]]
    return flags


def ssh_target(tr):
    return f"{tr['user']}@{tr['host']}" if tr.get("user") else tr["host"]


def ssh_argv(tr, command):
    """Argv and env for running one command string on the declared box."""
    prefix, env = ssh_prefix(tr)
    argv = prefix + ["ssh"] + ssh_flags(tr, batch=not env)
    if tr.get("port"):
        argv += ["-p", str(tr["port"])]
    return argv + [ssh_target(tr), command], env


def scp_argv(tr, sources, dest):
    """Argv and env for one recursive copy. `sources`/`dest` carry their own `host:` prefix."""
    prefix, env = ssh_prefix(tr)
    argv = prefix + ["scp", "-r"] + ssh_flags(tr, batch=not env)
    if tr.get("port"):
        argv += ["-P", str(tr["port"])]
    return argv + list(sources) + [dest], env


def remote_command(tr, argv):
    """The one string the remote shell runs: enter the workdir, then the argv, quoted."""
    return (f"cd {shlex.quote(tr['workdir'])} && "
            + " ".join(shlex.quote(a) for a in argv))


def remote_label(tr):
    port = f":{tr['port']}" if tr.get("port") else ""
    return f"ssh://{ssh_target(tr)}{port}{tr['workdir']}"


def transport_summary(side):
    """How a side is reached, for printing. Names the auth method, never the secret."""
    tr = transport_of(side)
    if not tr:
        return ""
    if tr.get("password") is not None or tr.get("password_env"):
        auth = "password"
    elif tr.get("identity_file"):
        auth = "key"
    else:
        auth = "agent/ssh_config"
    return f" via {remote_label(tr)}, auth={auth}"


def show_access(access):
    """One line: which sides the campaign may make claims about.

    Printed by `init` (side by side, as it always was) and now by the migration packet too. A
    campaign with no reachable target device can reach COMPILED and no further, and that is a
    fact about what the authorization is worth -- so it belongs in front of the human granting
    it rather than only in the line that scrolled past at `init`.
    """
    parts = []
    for side in ACCESS_SIDES:
        sec = access.get(side) or {}
        where = sec.get("host") or sec.get("device") or sec.get("soc")
        parts.append(f"{side}={'reachable' if sec.get('reachable') else 'not reachable'}"
                     + (f" ({where}{transport_summary(sec)})"
                        if sec.get("reachable") and where else ""))
    line = "Hardware access: " + ", ".join(parts)
    if not (access.get("a5") or {}).get("reachable"):
        line += (f"\n  No reachable {TARGET_ARCH} device is declared, so this authorization "
                 f"covers work that can reach COMPILED and cannot reach PROVEN.")
    a5 = access.get("a5") or {}
    if a5.get("reachable") and not transport_of(a5) \
            and not host_is_this_machine(a5.get("host")):
        line += (f"\n  a5 is declared reachable on `{a5.get('host')}`, which is not this "
                 f"machine, and no transport says how to get there — so `prove` will refuse "
                 f"the device run rather than execute it here. Add a `transport` block, or "
                 f"`\"transport\": {{\"kind\": \"local\"}}` if that host names this box.")
    return line


def perf_cases_state(run_dir):
    """Describe the staged performance-case table without treating blank cells as zeros."""
    path = os.path.join(run_dir, PERF_CASES)
    if not os.path.exists(path):
        return f"not staged (`{PERF_CASES}` is absent)"
    # A body row of a pipe table whose first two cells are numbers: the shape rows. Header,
    # separator and prose are skipped, so the denominator is what a sweep could actually fill.
    total = filled = rows = 0
    for line in read(path).split("\n"):
        cells = [c.strip() for c in line.strip().strip("|").split("|")] \
            if line.strip().startswith("|") else []
        if len(cells) < 6 or not (cells[0].isdigit() and cells[1].isdigit()):
            continue
        rows += 1
        for c in cells[4:]:
            total += 1
            filled += bool(c)
    if not total:
        return f"staged at `{PERF_CASES}`; fill every cell the declared access can measure"
    return (f"staged at `{PERF_CASES}` — {filled}/{total} cell(s) filled across {rows} case "
            f"row(s)" + ("" if filled == total else "; the rest are blank, and a blank means "
                         "not measured rather than measured as zero"))


def skill_head():
    """The engine revision, or "unversioned"; append `-dirty` for local changes."""
    here = os.path.dirname(os.path.abspath(__file__))
    proc = subprocess.run(["git", "-C", here, "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True)
    if proc.returncode != 0 or not proc.stdout.strip():
        return "unversioned"
    rev = proc.stdout.strip()
    # Scoped to the skill directory, so unrelated work elsewhere in the tree is not reported
    # as an engine change.
    skill = os.path.dirname(here)
    dirt = subprocess.run(["git", "-C", here, "status", "--porcelain", "--", skill],
                          capture_output=True, text=True)
    return rev + ("-dirty" if dirt.returncode == 0 and dirt.stdout.strip() else "")


def cmd_init(args):
    plan = load_plan(args.plan)
    run_dir = resolve_run_dir(args.run_dir, plan["target_root"])
    ignored = ensure_ignored(plan["target_root"], run_dir)
    os.makedirs(run_dir, exist_ok=True)
    for uid in (u["id"] for u in plan["units"]):
        os.makedirs(os.path.join(unit_dir(run_dir, uid), "logs"), exist_ok=True)
    # The tree as it stood before this campaign wrote anything. `check --phase implemented`
    # diffs against it, so unrelated local work already in the operator's tree is reported as
    # pre-existing rather than charged to an implementer.
    dirty = write_tree_baseline(plan["target_root"], run_dir)
    # A supplied table takes precedence on first initialization. Without one, stage the
    # bundled template. Existing campaign data is never replaced.
    perf_path = os.path.join(run_dir, PERF_CASES)
    if not os.path.exists(perf_path):
        source = args.perf_cases or PERF_CASE_TEMPLATE
        try:
            shutil.copyfile(source, perf_path)
        except OSError as exc:
            raise Fail(f"cannot stage performance case table from {source}: {exc}")

    existing = os.path.join(run_dir, "plan.json")
    if os.path.exists(existing):
        old = load_json(existing, "existing plan")
        if old != plan:
            raise Fail(f"{existing} already exists and differs from {args.plan}. The unit "
                       "list is the campaign scope and is frozen at init; use a fresh "
                       "--run-dir for a different scope.")
    else:
        dump_json(existing, plan)
        append_event(run_dir, "plan.frozen", units=[u["id"] for u in plan["units"]],
                     request=plan["request"][:200], skill_head=skill_head())


    # Hardware access is settled here, with the scope, because which sides are reachable
    # decides what the campaign may later claim -- and discovering it at PROVE time means
    # discovering it after the work.
    acc_path = os.path.join(run_dir, "access.json")
    if args.access:
        shutil.copyfile(args.access, acc_path)
    access = load_access(run_dir)
    for side in ACCESS_SIDES:
        sec = access[side]
        where = f" ({sec.get('host') or sec.get('device') or sec.get('soc')}"\
                f"{transport_summary(sec)})" if sec.get("reachable") else ""
        print(f"access {side}: {'reachable' if sec.get('reachable') else 'not reachable'}"
              f"{where}")
    if not any(access[s].get("reachable") for s in ACCESS_SIDES):
        print("  no device on either side: this campaign can compile and cannot prove "
              "accuracy. That is a reported outcome, not an error.")
    # Access is declared at initialization and governs the claims this campaign can make.
    # Performance cases are staged before this summary.
    print(f"perf cases: {perf_cases_state(run_dir)}")
    print(f"units: {len(plan['units'])}  target_root: {plan['target_root']}  "
          f"optimize: {'on' if plan['optimize']['enabled'] else 'off'}")
    for name, rev in sorted(plan["refs"].items()):
        print(f"ref pin: {name} @ {rev}")
    if ignored:
        print(f"ignored: added `{WORK_DIR}/` to {ignored}")
    print(f"skill: {skill_head()}   tree baseline: {len(dirty)} path(s) already dirty"
          + (f" (recorded in {TREE_BASELINE}; not charged to any unit)" if dirty else ""))
    print(run_dir)
    return 0


# --- refs --------------------------------------------------------------------------------

def git_ok(*argv):
    return subprocess.run(argv, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0


def resolve_ref(name, rev, target_root):
    """local -> per-tree cache -> blobless clone -> fail loudly.

    The cache holds a real checkout at the pin so later reads and greps work offline. Objects
    come from the local tree when it already has the revision, which skips the network
    entirely in the common case of running inside a CATLASS checkout.
    """
    dest = os.path.join(cache_root(target_root), f"{name}@{rev}")
    if os.path.isdir(os.path.join(dest, ".git")) and git_ok("git", "-C", dest, "cat-file",
                                                            "-e", f"{rev}^{{commit}}"):
        return {"path": dest, "rev": rev, "how": "cache"}

    source, how = REF_REPOS[name], "clone"
    if git_ok("git", "-C", target_root, "cat-file", "-e", f"{rev}^{{commit}}"):
        source, how = os.path.abspath(target_root), "local"

    if os.path.exists(dest):
        shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    clone = ["git", "clone", "--no-checkout", "--filter=blob:none", "-q", source, dest]
    if subprocess.run(clone).returncode != 0 or not git_ok("git", "-C", dest, "checkout",
                                                           "-q", "--detach", rev):
        shutil.rmtree(dest, ignore_errors=True)
        raise Fail(
            f"could not obtain reference {name} at {rev}.\n"
            f"  tried: local tree {target_root}, cache {dest}, clone {REF_REPOS[name]}\n"
            f"  fix it by hand with:\n"
            f"    git clone --filter=blob:none --no-checkout {REF_REPOS[name]} {dest}\n"
            f"    git -C {dest} checkout --detach {rev}\n"
            f"  then re-run `refs`. Without this reference no upstream claim in this "
            f"campaign is verifiable, so the workflow stops here rather than guessing.")
    return {"path": dest, "rev": rev, "how": how}


def cmd_refs(args):
    run_dir, plan = load_run(args.run_dir)
    root = plan["target_root"]
    # Reference cache writes use the same ignore rule as campaign state.
    ignored = ensure_ignored(root, cache_root(root))
    if ignored:
        print(f"ignored: added `{WORK_DIR}/` to {ignored}")
    out = {}
    for name, rev in sorted(plan["refs"].items()):
        out[name] = resolve_ref(name, rev, root)
        print(f"{name} @ {rev}  [{out[name]['how']}]  {out[name]['path']}")
    append_event(run_dir, "refs.resolved", **out)
    return 0


# --- profile -----------------------------------------------------------------------------

SURFACE_KEYS = ("path", "symbol", "required", "why")


def registration_surfaces(run_dir, root, required_only=True):
    """Return the central registration surfaces discovered in PROBE."""
    path = os.path.join(run_dir, "profile.json")
    if not os.path.exists(path):
        return []
    rows = (load_json(path, "profile").get("registration") or {}).get("surfaces") or []
    return [s["path"] for s in rows
            if isinstance(s, dict) and isinstance(s.get("path"), str)
            and (s.get("required") or not required_only)]


def check_surfaces(doc, root):
    rows = doc["registration"].get("surfaces")
    if not isinstance(rows, list) or not rows:
        raise Fail("profile.registration.surfaces must be a non-empty list of the CENTRAL "
                   "surfaces an added directory must touch -- the arch directory list, the "
                   "target-arch test registry, anything else this tree requires:\n"
                   '  {"path": "<tree-relative>", "symbol": "<list/registry variable|null>", '
                   '"required": true, "why": "<what breaks without it>"}\n'
                   "  This is the one profile field the engine reads. It is what lets the "
                   "ledger land registration AFTER the directory exists, and what lets "
                   "`check --phase analyzed` refuse an analysis that forgot one. The unit's "
                   "own example CMakeLists.txt is not a surface -- it is inside the grant.")
    for i, s in enumerate(rows):
        where = f"profile.registration.surfaces[{i}]"
        if not isinstance(s, dict):
            raise Fail(f"{where} must be an object with {', '.join(SURFACE_KEYS)}")
        closed(s, SURFACE_KEYS, where)
        rel = need_str(s, "path", where)
        if not os.path.isfile(contained(root, rel, f"{where}.path")):
            raise Fail(f"{where}.path is {rel!r}, which is not a file under the target root. "
                       f"A surface is a file that exists and gains a line; a file the tree "
                       f"does not have yet is a gaps[] row, not a surface.")
        if not isinstance(s.get("required"), bool):
            raise Fail(f"{where}.required must be a JSON boolean -- true when every added "
                       f"directory must touch it, false when only some do.")
        need_str(s, "why", where)


def cmd_profile(args):
    """Validate the four required profile sections and typed registration surfaces."""
    run_dir, plan = load_run(args.run_dir)
    path = os.path.join(run_dir, "profile.json")
    if not os.path.exists(path):
        raise Fail(f"no profile at {path}. PROBE writes one document with a section per "
                   f"concern: {', '.join(PROBE_CONCERNS)}. "
                   f"See references/02-probe.md for what each must answer.")
    doc = load_json(path, "profile")
    if not isinstance(doc, dict):
        raise Fail("profile.json must be an object with one section per concern")
    empty = [c for c in PROBE_CONCERNS
             if not isinstance(doc.get(c), dict) or not doc.get(c)]
    if empty:
        raise Fail(f"profile.json section(s) missing or empty: {', '.join(empty)}. Every "
                   f"concern needs an answer before analysis, because the target-specific "
                   f"facts every later phase relies on are discovered here and nowhere else. "
                   f"A concern the tree genuinely does not have is recorded as an explicit "
                   f"finding with a `gaps` note, never omitted.")
    check_surfaces(doc, plan["target_root"])
    surfaces = registration_surfaces(run_dir, plan["target_root"])
    append_event(run_dir, "profile.recorded", concerns=list(PROBE_CONCERNS),
                 surfaces=surfaces)
    print(f"profile.json: {len(PROBE_CONCERNS)} concern(s) answered: "
          f"{', '.join(PROBE_CONCERNS)}")
    print(f"required registration surfaces: {', '.join(surfaces) or '(none)'}")
    return 0


# --- check -------------------------------------------------------------------------------

def select_units(units, plan, only):
    if only:
        if only not in units:
            raise Fail(f"no unit {only!r} in plan.json")
        return [units[only]]
    return [units[u["id"]] for u in plan["units"]]


def cmd_check(args):
    run_dir, plan = load_run(args.run_dir)
    units, camp = fold(run_dir, plan)
    root = plan["target_root"]
    surfaces = registration_surfaces(run_dir, root)
    access = (load_access(run_dir)
              if args.phase in ("screened", "applied", "optimized") else None)
    problems, promoted = [], []

    for row in select_units(units, plan, args.unit):
        uid = row["id"]
        if row["parked"]:
            problems.append(f"unit {uid}: PARKED ({row['parked']}); unpark it deliberately "
                            "before claiming a phase")
            continue
        if row["excluded"]:
            problems.append(f"unit {uid}: excluded at the migration gate; it stays at ANALYZED")
            continue
        if row["opt_excluded"] and args.phase in ("applied", "optimized"):
            problems.append(f"unit {uid}: set aside at the optimization gate. Its migration "
                            f"stands and is reported; it is simply not rewritten.")
            continue
        if row["opt_skipped"] and args.phase == "screened":
            problems.append(f"unit {uid}: optimization was intentionally skipped at the "
                            "migration gate; a skipped unit stops at PROVEN and is not screened.")
            continue
        udir = unit_dir(run_dir, uid)
        try:
            if args.phase == "analyzed":
                if not camp["profile"]:
                    raise Fail("no profile recorded yet; run `profile` first. Every analysis "
                               "reads the discovered profile, not assumptions.")
                # Recheck the refused Cube-operand dtype before rendering the gate.
                hits = scan_blocked_dtype(root, row["source"])
                if hits:
                    raise refuse_blocked_dtype([(uid, row["source"], hits)])
                f = check_findings(os.path.join(udir, "findings.json"), uid, root, surfaces)
                append_event(run_dir, "unit.analyzed", unit=uid, route=f["route"])
                promoted.append(f"{uid} -> ANALYZED (route: {f['route']})")
            elif args.phase == "implemented":
                if RANK[row["phase"]] < RANK["AUTHORIZED"]:
                    raise Fail(f"unit is at {row['phase']}; nothing may be implemented before "
                               "CONFIRM authorizes it")
                require_current_migration_authorization(run_dir, row)
                f = check_findings(os.path.join(udir, "findings.json"), uid, root, surfaces)
                if f["route"] == ROUTE_TERMINAL:
                    raise Fail(f"route is {ROUTE_TERMINAL}: the external contract changes, so "
                               "this is not a migration. Report it; do not implement it.")
                # The scope check is a diff of the tree now, not a reading of the
                # implementer's own list of what it says it wrote. Everything the lead owns --
                # the ledger's rows and every declared surface, required or not.
                led = compute_ledger(run_dir, plan, units)
                check_tree_scope(root, run_dir, plan, uid, f, sorted(
                    {r["path"] for r in led["landable"]}
                    | set(registration_surfaces(run_dir, root, required_only=False))))
                append_event(run_dir, "unit.implemented", unit=uid, route=f["route"])
                promoted.append(f"{uid} -> IMPLEMENTED")
            elif args.phase == "compiled":
                # `prove` appends `unit.compiled` itself, because it is the command that owns
                # the build. This exists for the case where the build was run outside `prove`
                # -- a shared-edit regression build that also covers this unit, say -- and the
                # fact needs recording. It asserts nothing `prove` would not have asserted.
                if RANK[row["phase"]] < RANK["IMPLEMENTED"]:
                    raise Fail(f"unit is at {row['phase']}; there is nothing to compile until "
                               "it is implemented")
                append_event(run_dir, "unit.compiled", unit=uid, argv=None, log=None)
                promoted.append(f"{uid} -> COMPILED")
            elif args.phase == "screened":
                # The five rows read the *migrated* unit, so the migration has to exist and has
                # to be accurate before they mean anything. That is the whole reason this is a
                # phase of its own rather than part of ANALYZE, where the target does not exist.
                if RANK[row["phase"]] < RANK["PROVEN"]:
                    raise Fail(f"unit is at {row['phase']}; only a PROVEN unit is screened for "
                               f"the L0C->UB rewrite. The rows are read from the migrated unit, "
                               f"and a unit whose accuracy is unestablished is not one to "
                               f"rewrite.")
                if not plan.get("optimize", {}).get("enabled"):
                    raise Fail("this plan set `optimize.enabled` to false, so the campaign "
                               "deliberately opted out of the optimization phases. Turning "
                               "them back on is a scope change, so it belongs in a plan the "
                               "user saw, in a fresh run directory.")
                proof, issue = current_proof(row, access)
                if proof is None:
                    raise Fail("SCREEN requires a current hardware-bound proof: " + issue
                               + ". Re-run `prove` on the declared matching Ascend950 hardware.")
                s = check_screen(os.path.join(udir, "screen.json"), uid, root, udir)
                append_event(run_dir, "unit.opt_screened", unit=uid,
                             applicable=s["applicable"], strategy=s.get("strategy"),
                             baseline_us=s["baseline"]["task_us"])
                verdict = (f"applicable, proposes {s['strategy']}" if s["applicable"]
                           else "not applicable")
                # A rank never regresses, so re-screening a unit already past OPT_SCREENED
                # records the corrected artifact without moving it. Saying "-> OPT_SCREENED"
                # there read as a promotion that had not happened, and the next command then
                # refused for a reason the operator had just been told did not apply.
                if RANK[row["phase"]] > RANK["OPT_SCREENED"]:
                    promoted.append(
                        f"{uid} re-screened, still {row['phase']} ({verdict}; baseline "
                        f"{us_text(s['baseline']['task_us'])} us). A rank does not regress: the "
                        f"corrected screen is recorded, and re-presenting it needs `gate "
                        f"--phase optimize` then `confirm --phase optimize`.")
                else:
                    promoted.append(f"{uid} -> OPT_SCREENED ({verdict}; baseline "
                                    f"{us_text(s['baseline']['task_us'])} us)")
            elif args.phase == "applied":
                # Record the declared rewrite before the required re-proof.
                s = check_apply_ready(row, root, udir, access)
                o = check_optimize(os.path.join(udir, "optimize.json"), uid, root, udir, s)
                append_event(run_dir, "unit.applied", unit=uid, mode=o["mode"],
                             strategy=s["strategy"], paths=o["paths"])
                promoted.append(f"{uid} applied ({o['mode']}, {s['strategy']}; "
                                f"{len(o['paths'])} declared path(s), all present on disk -- "
                                f"which is not proof of what is in them). Re-run `prove` now: "
                                f"the rewrite is the default build, and only a proof recorded "
                                f"after this event counts for OPTIMIZED.")
            else:
                s = check_apply_ready(row, root, udir, access)
                # Accuracy for the rewritten path is `prove`'s job, not a field in
                # `optimize.json`: the rewrite is what the example builds by default, so the
                # re-prove is its record. Counting `unit.proven` events established nothing --
                # `prove` is deliberately re-runnable, so two runs that both pre-dated the
                # rewrite satisfied the count, and a unit reached OPTIMIZED with no rewrite on
                # disk at all. Order is what is checked instead: `unit.applied` says a rewrite
                # was declared landed, and a `unit.proven` after it says the golden was re-run
                # on whatever that left in the tree. The log is append-only and ordered, so
                # *index* order is the evidence; `ts` is second-resolution and two events
                # written in the same second tie.
                applied_at, proven_at = None, None
                for i, ev in enumerate(read_events(run_dir)):
                    if ev.get("unit") != uid:
                        continue
                    if ev["event"] == "unit.applied":
                        # The latest landing is the one that owes a proof: re-running
                        # `check --phase applied` re-declares the rewrite, and the only reason
                        # to re-declare it is that the tree changed underneath the last one.
                        applied_at, proven_at = i, None
                    elif ev["event"] == "unit.proven" and applied_at is not None:
                        proven_at = i
                if applied_at is None:
                    raise Fail("nothing in the log records that the rewrite landed. Land it at "
                               "the authorized paths, measure it, write optimize.json, then "
                               "`check --phase applied --unit <id>` -- that event is what the "
                               "re-prove is ordered against, so it has to precede it.")
                if proven_at is None:
                    raise Fail("every `unit.proven` event for this unit precedes its "
                               "`unit.applied`, so no accuracy run has been recorded since the "
                               "rewrite was declared landed. Re-run `prove` unchanged: the "
                               "rewrite is the default build, so that run is what establishes "
                               "accuracy for the path the tree now takes.")
                o = check_optimize(os.path.join(udir, "optimize.json"), uid, root, udir, s)
                us = {c["config"]: c["task_us"] for c in o["profile"]}
                # The two indices are the ordering evidence, so they are recorded rather than
                # left to be re-derived by whoever later doubts the verdict.
                append_event(run_dir, "unit.optimized", unit=uid, mode=o["mode"],
                             strategy=s["strategy"], task_us=us,
                             applied_at=applied_at, proven_at=proven_at)
                against = us.get("baseline", s["baseline"]["task_us"])
                promoted.append(f"{uid} -> OPTIMIZED ({o['mode']}, {s['strategy']}; "
                                f"{us_text(us['l0c_to_ub'])} us against {us_text(against)} us; "
                                f"proof at log "
                                f"index {proven_at} follows the rewrite declared at "
                                f"{applied_at})")
        except Fail as exc:
            problems.append(f"unit {uid}: {exc}")

    for line in promoted:
        print(line)
    for line in problems:
        print(f"FAIL {line}", file=sys.stderr)
    # `applied` promotes nothing, so calling its result a promotion would misreport it.
    print(f"{args.phase}: {len(promoted)} "
          f"{'recorded' if args.phase == 'applied' else 'promoted'}, "
          f"{len(problems)} problem(s)")
    return 1 if problems else 0


# --- shared-component ledger ---------------------------------------------------------------

def derive_consumers(root, symbol, limit=200):
    """Return source consumers of `symbol`, or None when the tree cannot be searched."""
    proc = subprocess.run(["git", "-C", root, "grep", "-lIwF", "--untracked", "-e", symbol,
                           "--", ".", ":(exclude)*.md", ":(exclude)*.markdown"],
                          capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        return None
    hits = sorted(p for p in proc.stdout.split("\n") if p.strip())
    return {"count": len(hits), "files": hits[:limit]}


def compute_ledger(run_dir, plan, units):
    """Deduplicate shared edits and partition them before units and after unit directories."""
    root = plan["target_root"]
    surfaces = set(registration_surfaces(run_dir, root, required_only=False))
    ledger = {}
    for row in units.values():
        if RANK[row["phase"]] < RANK["ANALYZED"] or row["parked"] or row["excluded"]:
            continue
        path = os.path.join(unit_dir(run_dir, row["id"]), "findings.json")
        # Skipping silently understated the blast radius the gate shows, and the missing file
        # only surfaced later when `gate` tried to load it. Fail here, naming the unit.
        if not os.path.exists(path):
            raise Fail(f"unit {row['id']} is at {row['phase']} but has no findings at {path}. "
                       f"The ledger is computed from every unparked unit's shared_components, "
                       f"so a missing one silently understates the blast radius the gate shows. "
                       f"Re-analyze that unit, or park it deliberately.")
        for s in load_json(path, "findings").get("shared_components") or []:
            key = (s["path"], s["symbol"])
            entry = ledger.setdefault(key, {"path": s["path"], "symbol": s["symbol"],
                                            "kinds": {}, "declared_by": [],
                                            "consumers_of": None})
            entry["declared_by"].append(row["id"])
            entry["kinds"].setdefault(s["kind"], []).append(row["id"])
            entry["consumers_of"] = entry["consumers_of"] or s.get("consumers_of")

    rows = []
    for entry in ledger.values():
        # Strongest kind wins: if any unit must add it, it is an add.
        kind = "add" if "add" in entry["kinds"] else "generalize"
        sym = entry["consumers_of"]
        rows.append({"path": entry["path"], "symbol": entry["symbol"], "kind": kind,
                     "declared_by": sorted(set(entry["declared_by"])),
                     "consumers_of": sym,
                     "existing": derive_consumers(root, sym) if sym else None,
                     "surface": entry["path"] in surfaces})
    rows.sort(key=lambda r: (r["surface"], r["path"], r["symbol"]))
    order = sorted(r["id"] for r in units.values()
                   if RANK[r["phase"]] >= RANK["ANALYZED"] and not r["parked"]
                   and not r["excluded"])
    return {"landable": rows, "order": order,
            "declarations": [r for r in rows if not r["surface"]],
            "surfaces": [r for r in rows if r["surface"]]}


def render_existing(row):
    """The regression obligation for one landable row, as a measurement."""
    if not row["consumers_of"]:
        return "new symbol — no existing consumer, so no source-arch regression obligation"
    e = row["existing"]
    if e is None:
        return (f"existing consumers of {row['consumers_of']}: NOT MEASURED "
                f"(no git work tree here) — count them by hand before landing this")
    shown = ", ".join(e["files"][:6]) or "(none found — check the spelling)"
    more = f", +{e['count'] - 6} more" if e["count"] > 6 else ""
    return (f"{e['count']} existing consumer(s) of {row['consumers_of']}, from "
            f"`git grep -l`: {shown}{more}")


def render_ledger(led):
    """The execution plan, in an order that can be run literally."""
    out = ["1. shared declarations — the lead lands these first, then runs the source-arch "
           "regression build:"]
    for s in led["declarations"]:
        out.append(f"   {s['kind']:<10} {s['path']} :: {s['symbol']}   "
                   f"declared by unit(s) {','.join(s['declared_by'])}")
        out.append(f"   {'':<10} {render_existing(s)}")
    if not led["declarations"]:
        out.append("   (none)")
    out.append(f"2. units, one at a time: {', '.join(led['order']) or '(none)'}")
    out.append("3. registration surfaces — the lead lands each one AFTER that unit's "
               "implementer returns:")
    for s in led["surfaces"]:
        out.append(f"   {s['kind']:<10} {s['path']} :: {s['symbol']}   "
                   f"declared by unit(s) {','.join(s['declared_by'])}")
    if not led["surfaces"]:
        out.append("   (none)")
    else:
        out.append("   Registering a directory name before that directory exists makes the "
                   "add_subdirectory loop fail on every subsequent target-arch configure, "
                   "which is why this block is never step 1.")
    return "\n".join(out)


# --- gate / confirm ----------------------------------------------------------------------

# No accepted grammar, and none is wanted. A three-form reply table looks like a check and is
# not one -- nothing here ever parsed a reply. What actually protects the decision is that the
# agent's interpretation is written into the artifacts and the packet is re-rendered, so the
# last thing a human sees before authorization is the interpreted values rather than their own
# prose. The two asymmetries are rules because inference fails in one direction.
REPLY_GUIDANCE = (
    "\nReply in whatever form is natural: approval, exclusions, questions, or a decision\n"
    "other than the one proposed. There is no accepted grammar. The agent interprets the\n"
    "reply, writes that interpretation into the artifacts and re-renders this packet, so the\n"
    "last thing shown before authorization is the interpreted values. Two rules it does not\n"
    "bend: an ambiguous, partial or conditional reply is a question and never an approval;\n"
    "and a `replace` strategy needs an explicit yes, because it is the one irreversible step\n"
    "in this workflow -- nothing here commits, so a failed re-prove after a replace leaves\n"
    "nothing to restore.")


# The artifact each gate renders its per-unit block from. `confirm` re-digests exactly this file,
# so the binding covers what the packet showed and nothing else.
GATE_ARTIFACT = {"migrate": "findings.json", "optimize": "screen.json"}


def artifact_digest(path):
    """A short content digest of the file a packet block was rendered from.

    Truncated to 16 hex characters deliberately. What this detects is drift between two commands
    a human runs minutes apart in one run directory: an artifact rewritten after the packet was
    printed. It is not a security boundary -- anyone who can rewrite the artifact can rewrite
    `events.jsonl` beside it -- so a longer digest would buy nothing the log's own integrity does
    not already assume.
    """
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()[:16]
    except OSError as exc:
        raise Fail(f"cannot digest {path}: {exc}")

def migration_gate_pending(row):
    """Whether a unit needs a current migration packet decision."""
    return (not row["parked"] and
            (row["phase"] == "ANALYZED" or
             (row["authorization_invalidated"] and row["authorization_reanalyzed"])))


def migration_authorization_issue(run_dir, row):
    """Return why the migration grant cannot be used *now*, or ``None`` when it can."""
    if row["authorization_invalidated"]:
        return ("the migration authorization was invalidated when its confirmed claim was "
                "contradicted")
    authorization = row.get("authorization")
    if not isinstance(authorization, dict):
        return ("the recorded migration authorization has no plan/profile/findings snapshot "
                "(an older event cannot establish what the human approved)")
    missing = [key for key in ("plan", "profile", "findings")
               if not isinstance(authorization.get(key), str) or not authorization[key]]
    if missing:
        return ("the recorded migration authorization lacks its " + ", ".join(missing)
                + " digest(s) (an older event cannot establish what the human approved)")
    current = {
        "plan": artifact_digest(os.path.join(run_dir, "plan.json")),
        "profile": artifact_digest(os.path.join(run_dir, "profile.json")),
        "findings": artifact_digest(os.path.join(
            unit_dir(run_dir, row["id"]), "findings.json")),
    }
    changed = [key for key in ("plan", "profile", "findings")
               if authorization[key] != current[key]]
    if changed:
        return ("the current " + ", ".join(changed)
                + " artifact(s) no longer match the migration packet this unit was authorized "
                + "from")
    return None


def require_current_migration_authorization(run_dir, row):
    """Refuse a migration build or proof unless the human's exact grant remains current."""
    issue = migration_authorization_issue(run_dir, row)
    if issue is None:
        return
    if RANK[row["phase"]] >= RANK["IMPLEMENTED"]:
        raise Fail(f"{issue}. This contradiction was found after IMPLEMENTED, so this campaign "
                   "rank cannot be reused: start a fresh campaign and obtain a new GATE-1 "
                   "authorization there.")
    raise Fail(f"{issue}. No migration build may proceed. Park the AUTHORIZED unit with the "
               "evidenced contradiction, deliberately unpark it, re-run `check --phase "
               "analyzed`, then render GATE 1 and record a replacement confirmation.")


def status_authorization_recovery(run_dir, row):
    """A pasteable status recovery that never points a stale grant at a build."""
    if row["authorization_invalidated"]:
        if row["authorization_reanalyzed"]:
            return ("its invalidated authorization has been re-analyzed; render GATE 1 with "
                    f"`gate --run-dir {shlex.quote(run_dir)}`, then record replacement "
                    "`confirm` decision")
        return ("its authorization is invalidated; re-investigate it with "
                f"`check --run-dir {shlex.quote(run_dir)} --phase analyzed --unit {row['id']}`, "
                "then render GATE 1 and record replacement `confirm` decision")
    try:
        issue = migration_authorization_issue(run_dir, row)
    except Fail as exc:
        issue = f"the authorization cannot be checked ({exc})"
    if issue is None:
        return None
    if RANK[row["phase"]] >= RANK["IMPLEMENTED"]:
        return (f"{issue}; this was discovered after IMPLEMENTED, so start a fresh campaign "
                "rather than reuse this rank")
    return (f"{issue}; no build may proceed. Park the AUTHORIZED unit, deliberately unpark it, "
            "re-run the analyzed check, then render GATE 1 and record replacement confirmation")




def bind_gate(run_dir, gate_pay, pending, gate):
    """Require the confirmation to match the rendered packet and its artifacts."""
    presented = list(gate_pay.get("units") or [])
    recorded = gate_pay.get("digests") or {}
    pending = sorted(pending)
    unpresented = [u for u in pending if u not in presented]
    withdrawn = [u for u in sorted(presented) if u not in pending]
    # A presented unit with no recorded digest counts as changed rather than as verified: the
    # only way to reach that is a log written by hand or by an engine that predates this check,
    # and the price of the false accusation is one `gate` re-render.
    changed = [u for u in sorted(presented) if u in pending
               and recorded.get(u) != artifact_digest(
                   os.path.join(unit_dir(run_dir, u), GATE_ARTIFACT[gate]))]
    # The packet renders every unit's source and target out of plan.json, so the plan is part
    # of what was shown. Profile facts choose the build, registration and golden surfaces, so
    # a migration grant must bind it too. A missing legacy snapshot is drift, not a pass.
    plan_now = artifact_digest(os.path.join(run_dir, "plan.json"))
    plan_moved = gate_pay.get("plan") != plan_now
    profile_moved = (gate == "migrate" and
                     gate_pay.get("profile") != artifact_digest(
                         os.path.join(run_dir, "profile.json")))
    if not (unpresented or withdrawn or changed or plan_moved or profile_moved):
        return
    lines = [f"this confirmation does not describe the {gate} packet that was rendered here:"]
    if unpresented:
        lines.append(f"  not presented, but would be authorized: {', '.join(unpresented)}")
    if withdrawn:
        lines.append(f"  presented, but no longer awaiting this gate: {', '.join(withdrawn)}")
    if changed:
        lines.append(f"  {GATE_ARTIFACT[gate]} changed since the packet: {', '.join(changed)}")
    if plan_moved:
        lines.append("  plan.json changed since the packet — the sources and targets shown "
                     "are not the ones this would authorize")
    if profile_moved:
        lines.append("  profile.json changed since the migration packet — its discovered build, "
                     "golden, or registration facts are not the ones this would authorize")
    lines.append(f"Re-render the packet with `mig.py gate --run-dir {run_dir}"
                 f"{' --phase optimize' if gate == 'optimize' else ''}`, have the human read the "
                 f"current analysis, then confirm. Re-presentation is the designed recovery: "
                 f"nothing is lost and no artifact has to be reverted. (Excluding a unit at "
                 f"`confirm --exclude` does not help -- it is still a unit this gate presented "
                 f"or failed to present.)")
    raise Fail("\n".join(lines))


def na(val):
    """Render one contract or type-stack field for the packet.

    A field written {"not_applicable": "<reason>"} is an answer, not a gap, so it prints its
    reason. Printing the raw dict made an answered field look like a defect in the packet.
    """
    return f"not applicable — {val['not_applicable']}" if not_applicable(val) else val


# Per-column budgets for the tensor table, and one budget for the prose fields. Nothing here
# truncates an artifact -- `findings.json` keeps every character. What it bounds is the packet,
# which is the one place in this workflow where readability is load-bearing: a `dtype` cell
# that legitimately records "int4b_t packed on GM, int8_t at Cube via the prologue Cast"
# pushed an unclamped header past 1000 characters, and a human was asked to read it closely.
TENSOR_WIDTHS = (14, 22, 24, 20, 24, 10)
PROSE_WIDTH = 108


def show_contract(c, indent="     "):
    """The frozen contract, laid out for the comparison the gate exists for.

    Every field `check_contract` requires and none of the extras beside them: the schema is
    deliberately open there (`cli`, `frozen_from`, ...), and an unbounded dump of whatever an
    analyst added is not readable at a terminal. The reviewer who wants those opens the file.
    """
    inner = indent + "  "
    print(f"{indent}contract")
    tensors = c["tensors"]
    if not_applicable(tensors):
        print(f"{inner}tensors: {clip(na(tensors), PROSE_WIDTH)}")
    else:
        head = ("tensor", "role", "dtype", "layout", "storage", "aliases")
        rows = [tuple(clip(v, TENSOR_WIDTHS[i]) for i, v in enumerate(
                    (t["name"], t["role"], t["dtype"], t["layout"], t["storage"],
                     t["alias_of"] or "—")))
                for t in tensors]
        width = [max(len(r[i]) for r in (head, *rows)) for i in range(len(head))]
        for row in (head, *rows):
            print(inner + "  ".join(v.ljust(width[i]) for i, v in enumerate(row)).rstrip())
        if any(len(str(v)) > TENSOR_WIDTHS[i]
               for t in tensors for i, v in enumerate(
                   (t["name"], t["role"], t["dtype"], t["layout"], t["storage"],
                    t["alias_of"] or "—"))):
            print(f"{inner}(cells clipped for width; findings.json holds them whole)")
    print(f"{inner}output region:    {clip(na(c['output_region']), PROSE_WIDTH)}")
    print(f"{inner}supported domain: {clip(na(c['supported_domain']), PROSE_WIDTH)}")
    print(f"{inner}zero work:        {clip(na(c['zero_work']), PROSE_WIDTH)}")
    g = c["golden"]
    if not_applicable(g):
        print(f"{inner}golden:           {clip(na(g), PROSE_WIDTH)}")
    else:
        print(f"{inner}golden:           {clip(g['function'], PROSE_WIDTH)}")
        print(f"{inner}  comparator:     {clip(g['comparator'], PROSE_WIDTH)}")
        print(f"{inner}  compared:       tensor {g['compared_tensor']}  "
              f"dtype {g['compared_dtype']!r} (exact evidence label)")
        print(f"{inner}  compute_num:    {g['compute_num']}  "
              f"(read at {g['compute_num_read_from']})")


def show_type_stack(ts, indent="     "):
    """The layers in play, one line each.

    A summary, not the record: nine layers of expanded template types do not fit a packet a
    human will actually read, and `findings.json` holds all nine with the file each was read
    from. A layer the analyst declared inapplicable carries its reason there and nothing here,
    so only the live ones are printed and the count of the rest is stated.
    """
    live = [k for k in TYPE_STACK_LAYERS if not not_applicable(ts[k])]
    quiet = len(TYPE_STACK_LAYERS) - len(live)
    print(f"{indent}type stack — {len(live)} layer(s) in play"
          + (f", {quiet} not applicable" if quiet else "")
          + "; findings.json is the record")
    for k in live:
        print(f"{indent}  {k}: {clip(ts[k]['type'], PROSE_WIDTH)}  ({ts[k]['read_from']})")


def show_prove(f, indent="     "):
    """The build and run that become the accuracy claim, and the write grant.

    Both are things this authorization confers rather than describes, which is why they belong
    in front of the human granting it. A `redesign` unit owes no build plan and no writable
    paths -- there is nothing to build and nothing to write -- so it prints neither.
    """
    p = f.get("prove")
    if isinstance(p, dict):
        print(f"{indent}prove — the claim PROVEN will be made from")
        print(f"{indent}  build: {' '.join(p['build'])}")
        print(f"{indent}  run:   {' '.join(p['run'])}")
        print(f"{indent}  cwd:   {p['cwd']}   shape: "
              f"{'x'.join(str(v) for v in p['shape'])}   device: {p['device']}")
    paths = f.get("writable_paths") or []
    if paths:
        print(f"{indent}writable paths — the write grant this authorizes, and nothing else")
        for path in paths:
            print(f"{indent}  {path}")


def cmd_gate(args):
    """Render a gate packet and exit 2.

    Two gates, one renderer, because they differ only in what they show. Each sits immediately
    before the writes it authorizes: the migration gate before any migration write, the
    optimization gate before any byte of the L0C->UB rewrite. The promise is not "one gate" but
    "nothing reaches the tree that a gate has not already described".
    """
    run_dir, plan = load_run(args.run_dir)
    units, camp = fold(run_dir, plan)
    if args.phase == "optimize":
        return gate_optimize(run_dir, units, load_access(run_dir))
    # The migration packet derives its execution ledger from current findings.
    led = compute_ledger(run_dir, plan, units)
    # A freshly analyzed unit normally awaits its first authorization. An invalidated AUTHORIZED
    # unit keeps its rank, but may be shown again only after explicit unpark and re-analysis.
    pending = [r for r in units.values() if migration_gate_pending(r)]
    if not pending:
        raise Fail("no unit at ANALYZED or re-analyzed after an invalidated authorization to "
                   "confirm. An AUTHORIZED contradiction must be parked, deliberately unparked, "
                   "and re-analyzed before it can return to GATE 1. A contradiction after "
                   "IMPLEMENTED requires a fresh campaign.")

    print(f"\n# MIGRATION CONFIRMATION — {len(pending)} unit(s)\n")
    print(f"Request as recorded: {plan['request']}\n")
    print(show_access(load_access(run_dir)))
    print(f"Performance cases: {perf_cases_state(run_dir)}")
    print("Optimization: " + ("on — every unit that proves is then screened for the L0C→UB "
                              "rewrite and profiled once; the rewrite itself is authorized "
                              "separately at the optimization gate"
                              if plan["optimize"]["enabled"] else
                              "off — this plan set `optimize.enabled` false, so no unit is "
                              "screened and no unit is measured") + "\n")
    if plan["optimize"]["enabled"]:
        print("To intentionally stop a shown unit after PROVEN, record it now with "
              "`confirm --skip-optimize <ids>`. Each id must remain pending in this packet "
              "and cannot also be excluded.\n")
    digests = {}
    for row in sorted(pending, key=lambda r: r["id"]):
        fpath = os.path.join(unit_dir(run_dir, row["id"]), "findings.json")
        f = load_json(fpath, "findings")
        digests[row["id"]] = artifact_digest(fpath)
        print(f"## unit {row['id']}: {row['source']} -> {row['target']}")
        print(f"   route: {f['route']}"
              + ("   [TERMINAL: contract changes, not a migration]"
                 if f["route"] == ROUTE_TERMINAL else ""))
        # Render the counterpart finding for the human decision.
        cp = f.get("counterpart")
        if cp:
            print(f"   already migrated? {cp['verdict']} — suspect {cp['suspect']}")
            print(f"     {clip(cp['evidence'], PROSE_WIDTH)}")
        else:
            print("   already migrated? no — no target-arch directory plausibly implements "
                  "this operator")
        for r in f["routes"]:
            print(f"     {r['route']:<13} {r['verdict']:<16} "
                  f"{clip(r['evidence'], PROSE_WIDTH)}")
            # `needs-diagnostic` without the command that settles it is indistinguishable from
            # "not investigated", and the gate is where someone can ask for it to be run.
            if r.get("diagnostic"):
                print(f"     {'':<13} {'':<16} diagnostic: {r['diagnostic']}")
            # Inside a cluster, unit n>1 is a plain retarget that waits on the component unit 1
            # established. Saying so is what keeps the packet from claiming N unblocks.
            if r.get("conditional_on"):
                print(f"     {'':<13} {'':<16} conditional on: {r['conditional_on']}")
        show_contract(f["contract"])
        show_type_stack(f["type_stack"])
        show_prove(f)
        for s in f.get("shared_components") or []:
            print(f"     shared: {s['kind']:<10} {s['path']} :: {s['symbol']}")
            # The field `check_findings` requires because it is "read by exactly one audience:
            # the human at the gate deciding whether this shared edit is justified" -- which
            # only holds if the packet actually prints it.
            print(f"       why: {clip(s['why'], PROSE_WIDTH)}")
        print()
    print(render_ledger(led))
    print(REPLY_GUIDANCE)
    print("\nThen record the outcome:\n"
          f"  mig.py confirm --run-dir {run_dir} --intent '<final human decision>' "
          "[--exclude <ids>] [--skip-optimize <ids>]")
    # Records that a packet was rendered in this run directory, for which units, and from which
    # bytes. It does not prove a human read it or replied -- nothing here can -- but `confirm`
    # re-deriving all three makes "this decision is about this packet" mechanical instead of
    # assumed.
    append_event(run_dir, "gate.presented", gate="migrate",
                 units=sorted(r["id"] for r in pending), digests=digests,
                 plan=artifact_digest(os.path.join(run_dir, "plan.json")),
                 profile=artifact_digest(os.path.join(run_dir, "profile.json")))
    return 2


def gate_optimize(run_dir, units, access):
    """The second gate: which migrated units get the L0C->UB rewrite, and on which terms.

    Campaign-wide and after the migrations, because that is when the answer exists. The five
    rows were read from the built unit, the manifest was walked from the built unit's own
    includes, and the baseline was measured on it. None of that is available at the migration
    gate, which is why presenting it there would have meant presenting a prediction.
    """
    # Re-presentation includes OPT_AUTHORIZED units so an updated screen manifest can be
    # presented again. Completed rewrites are not eligible for another authorization.
    candidates = [r for r in units.values()
                  if r["phase"] in ("OPT_SCREENED", "OPT_AUTHORIZED") and not r["parked"]
                  and not r["opt_skipped"]]
    pending, withheld = [], []
    for row in candidates:
        proof, issue = current_proof(row, access)
        if proof is None:
            withheld.append((row["id"], issue))
        else:
            pending.append(row)
    if not pending:
        detail = (" Current-proof failures: " + "; ".join(
            f"{uid}: {issue}" for uid, issue in withheld)) if withheld else ""
        raise Fail("no unit at OPT_SCREENED or OPT_AUTHORIZED with a current hardware-bound "
                   "proof to confirm. Re-run `prove` on the declared matching Ascend950 "
                   "hardware before GATE 2." + detail)

    if withheld:
        print("Not eligible for this GATE 2 packet; re-prove before re-presenting:")
        for uid, issue in withheld:
            print(f"  unit {uid}: {issue}")
        print()

    print(f"\n# OPTIMIZATION CONFIRMATION — {len(pending)} unit(s)\n")
    print("The non-TLA L0C->UB data-path rewrite. Authorizing a unit authorizes writes to the\n"
          "files listed beneath it and to nothing else.\n")
    # The gate where performance is actually on the table, so the optional performance-case
    # comparison is offered here too -- every baseline below was measured to get this far, and
    # a table is the only place those numbers become a before-and-after a reader can check.
    print(f"Performance cases: {perf_cases_state(run_dir)}\n")
    digests = {}
    for row in sorted(pending, key=lambda r: r["id"]):
        spath = os.path.join(unit_dir(run_dir, row["id"]), "screen.json")
        s = load_json(spath, "screen")
        digests[row["id"]] = artifact_digest(spath)
        print(f"## unit {row['id']}: {row['target']}")
        print(f"   measured baseline: {us_text(s['baseline']['task_us'])} us  "
              f"({s['baseline']['source']})")
        for name in SCREEN_ROWS:
            print(f"     {name:<15} {clip(s['rows'][name], PROSE_WIDTH)}")
        if not s["applicable"]:
            print("   NOT APPLICABLE — no rewrite and no writes. The screen above and the "
                  "baseline are this unit's result.")
        else:
            print(f"   proposed strategy: {s['strategy']}"
                  + ("   [IRREVERSIBLE: removes the proven baseline]"
                     if s["strategy"] == "replace"
                     else "   [keeps the proven baseline compilable behind the opt-in switch]"))
            print("   files this authorizes:")
            for m in s["manifest"]:
                print(f"     tier {m['tier']}  {m['action']:<7} {m['path']}")
        print()
    print(REPLY_GUIDANCE)
    print("\nThen record the outcome:\n"
          f"  mig.py confirm --run-dir {run_dir} --phase optimize "
          "--intent '<final human decision>' [--exclude <ids>]")
    append_event(run_dir, "gate.presented", gate="optimize",
                 units=sorted(r["id"] for r in pending), digests=digests,
                 plan=artifact_digest(os.path.join(run_dir, "plan.json")))
    return 2


def cmd_confirm(args):
    run_dir, plan = load_run(args.run_dir)
    units, camp = fold(run_dir, plan)
    optimize = args.phase == "optimize"
    access = load_access(run_dir) if optimize else None
    skipped = [u.strip() for u in (args.skip_optimize or "").split(",") if u.strip()]
    if optimize and skipped:
        raise Fail("--skip-optimize is accepted only by the migration confirmation; the "
                   "optimization gate decides apply or skip for screened units.")
    if skipped and not plan.get("optimize", {}).get("enabled"):
        raise Fail("--skip-optimize is unavailable because this plan set "
                   "`optimize.enabled` to false.")

    gate_pay = camp["opt_gate" if optimize else "gate"]
    if gate_pay is None:
        raise Fail(f"no {'optimization' if optimize else 'migration'} gate packet has been "
                   f"rendered in this run directory; run "
                   f"`gate{' --phase optimize' if optimize else ''}` first.")
    excluded = [u.strip() for u in (args.exclude or "").split(",") if u.strip()]
    unknown = sorted(set(excluded + skipped) - set(units))
    if unknown:
        raise Fail(f"--exclude/--skip-optimize names unknown unit(s): {', '.join(unknown)}")
    overlap = sorted(set(excluded) & set(skipped))
    if overlap:
        raise Fail("a unit cannot be both excluded and intentionally skipped for optimization: "
                   + ", ".join(overlap))

    wants = ("OPT_SCREENED", "OPT_AUTHORIZED") if optimize else ("ANALYZED",)
    if optimize:
        stale = []
        for uid in gate_pay.get("units") or []:
            if uid in units:
                proof, issue = current_proof(units[uid], access)
                if proof is None:
                    stale.append(f"{uid}: {issue}")
        if stale:
            raise Fail("GATE 2 cannot be confirmed because its rendered unit(s) no longer have "
                       "a current hardware-bound proof. Re-run `prove`, then re-render GATE 2: "
                       + "; ".join(stale))
        pending = [r["id"] for r in units.values()
                   if r["phase"] in wants and not r["parked"]
                   and current_proof(r, access)[0] is not None]
    else:
        pending = [r["id"] for r in units.values() if migration_gate_pending(r)]
    bind_gate(run_dir, gate_pay, pending, args.phase)
    invalid_skips = sorted(set(skipped) - set(pending))
    if invalid_skips:
        raise Fail("--skip-optimize may name only units still pending in the rendered migration "
                   "packet: " + ", ".join(invalid_skips))

    actor = os.environ.get("USER") or os.environ.get("LOGNAME") or "unknown"
    # `intent` is recorded exactly as the human decision supplied it.
    append_event(run_dir, "campaign.confirmed", actor=f"human:{actor}", gate=args.phase,
                 excluded=excluded, skip_optimize=skipped, intent=args.intent)
    for uid in skipped:
        append_event(run_dir, "unit.opt_skipped", unit=uid, gate="migrate")

    authorized, terminal = [], []
    for row in sorted(units.values(), key=lambda r: r["id"]):
        pending_here = (row["phase"] in wants and current_proof(row, access)[0] is not None
                        if optimize else migration_gate_pending(row))
        if row["id"] in excluded or not pending_here:
            continue
        udir = unit_dir(run_dir, row["id"])
        if optimize:
            s = load_json(os.path.join(udir, "screen.json"), "screen")
            if not s["applicable"]:
                terminal.append(row["id"])
                continue
            append_event(run_dir, "unit.opt_authorized", unit=row["id"],
                         strategy=s["strategy"],
                         manifest=sorted(m["path"] for m in s.get("manifest") or []))
            authorized.append(f"{row['id']}({s['strategy']})")
            continue
        f = load_json(os.path.join(udir, "findings.json"), "findings")
        if f["route"] == ROUTE_TERMINAL:
            append_event(run_dir, "unit.parked", unit=row["id"],
                         reason=f"route {ROUTE_TERMINAL}: the external contract changes, so "
                                "this is new-contract work rather than a migration")
            continue
        append_event(run_dir, "unit.authorized", unit=row["id"], route=f["route"],
                     plan=gate_pay["plan"], profile=gate_pay["profile"],
                     findings=gate_pay["digests"][row["id"]])
        authorized.append(row["id"])

    print(f"confirmed by human:{actor}  ({args.phase} gate)")
    print(f"intent: {args.intent}")
    print(f"authorized: {', '.join(authorized) if authorized else '(none)'}")
    if terminal:
        print(f"not applicable (terminal at OPT_SCREENED): {', '.join(terminal)}")
    if excluded:
        print(f"set aside (stay {'OPT_SCREENED' if optimize else 'ANALYZED'}): "
              f"{', '.join(excluded)}")
    if skipped:
        print(f"optimization intentionally skipped after PROVEN: {', '.join(skipped)}")
    return 0


def cmd_park(args):
    run_dir, plan = load_run(args.run_dir)
    units, _ = fold(run_dir, plan)
    if args.unit not in units:
        raise Fail(f"no unit {args.unit!r} in plan.json")
    row = units[args.unit]
    if args.unpark:
        append_event(run_dir, "unit.unparked", unit=args.unit)
        print(f"unit {args.unit}: unparked (phase {row['phase']})")
        return 0
    if not args.reason:
        raise Fail("--reason is required: a parked unit with no reason is an unexplained gap "
                   "in the report")
    if RANK[row["phase"]] >= RANK["IMPLEMENTED"]:
        raise Fail(f"unit {args.unit} is already {row['phase']}. A contradiction after "
                   "IMPLEMENTED cannot be repaired by parking and reusing this rank; start a "
                   "fresh campaign.")
    if row["phase"] == "AUTHORIZED":
        append_event(run_dir, "unit.authorization_invalidated", unit=args.unit,
                     reason=args.reason)
    append_event(run_dir, "unit.parked", unit=args.unit, reason=args.reason,
                 at_phase=row["phase"])
    print(f"unit {args.unit}: PARKED at {row['phase']} — {args.reason}")
    if row["phase"] == "AUTHORIZED":
        print("Its migration authorization is invalidated. Surface this to the human, then "
              "unpark, re-analyze, render GATE 1, and record replacement confirmation.")
    else:
        print("Surface this to the human now; the rest of the campaign continues.")
    return 0


# --- prove -------------------------------------------------------------------------------

def run_stage(argv, cwd, log, timeout, env=None, mode="w"):
    with open(log, mode, encoding="utf-8") as fh:
        fh.write(f"$ {' '.join(argv)}\n(cwd: {cwd})\n\n")
        fh.flush()
        try:
            proc = subprocess.run(argv, cwd=cwd, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, timeout=timeout,
                                  env={**os.environ, **env} if env else None)
        except FileNotFoundError:
            fh.write(f"\n[not executable: {argv[0]}]\n")
            return 127, ""
        except subprocess.TimeoutExpired as exc:
            fh.write((exc.output or "") + f"\n[timed out after {timeout}s]\n")
            return 124, exc.output or ""
        fh.write(proc.stdout)
        return proc.returncode, proc.stdout


def effective_run_argv(run):
    """The words that actually reach the example's `main()`, through a shell wrapper if any.

    `references/02-probe.md` requires a genuinely shell-dependent run -- one that has to
    source the toolkit first, the common case in a CATLASS tree -- to be wrapped in an
    explicit shell argv such as `bash -lc "..."`, and then the program's own
    arguments live INSIDE the last element rather than beside it, so comparing the evidence
    line's `argv` echo against `run`'s tail compares it against a fragment of one shell string
    and fails every correct run. Split that element the way the shell will. A wrapper this
    cannot parse falls back to the raw argv, which is the pre-existing behaviour.
    """
    if len(run) >= 3 and os.path.basename(run[0]) in ("sh", "bash", "zsh", "dash") \
            and run[1].startswith("-") and "c" in run[1].lstrip("-"):
        try:
            return shlex.split(run[-1])
        except ValueError:
            return run
    return run


def campaign_start(run_dir):
    """Epoch seconds of `plan.frozen` -- the instant this campaign began."""
    for e in read_events(run_dir):
        if e.get("event") == "plan.frozen":
            return int(datetime.strptime(e["ts"], "%Y-%m-%dT%H:%M:%SZ")
                       .replace(tzinfo=timezone.utc).timestamp())
    raise Fail(f"{log_path(run_dir)} records no plan.frozen event, so this directory was not "
               f"minted by `init` and there is no campaign to date an artifact against.")


def run_artifacts(root, words, base):
    """Return in-tree run artifacts and their modification times."""
    root = os.path.realpath(root)
    out = []
    for word in words:
        w = str(word).strip("\"'")
        if not w or w.startswith("-") or os.sep not in w:
            continue
        full = os.path.realpath(w if os.path.isabs(w) else os.path.join(base, w))
        if full != root and not full.startswith(root + os.sep):
            continue
        is_file = os.path.isfile(full)
        out.append({"path": os.path.relpath(full, root), "present": is_file,
                    "mtime": int(os.path.getmtime(full)) if is_file else None})
    return out


def toolchain_preflight(argv, cwd):
    """What the declared command needs from this machine: checked, recorded, never enforced.

    Derived from the argv the profile and the analyst discovered, never from a remembered
    install location -- a hardcoded toolkit path would be exactly the target-specific default
    this skill may not hold. Two things are checkable without running
    anything: whether `argv[0]` is executable, and whether every script the command `source`s
    exists.

    It never blocks. A container, a module system or a login profile can put
    a toolkit somewhere this cannot see, so a missing reading is a warning to record rather
    than a verdict -- which is why the build runs regardless. What the reading buys is the
    distinction that matters when the build then fails: "the migrated code does not compile"
    versus "this machine has no compiler". Without it both are `at: build`, and a campaign on a
    box with no CANN reports every unit as a compile failure of the migration.
    """
    obs, exe = [], (argv[0] if argv else "")
    if exe:
        found = (exe if os.access(exe, os.X_OK) else None) if os.sep in exe \
            else shutil.which(exe)
        obs.append({"what": "executable", "path": exe, "present": bool(found)})
    # Only `source`/`.` operands, never every token that looks like a path: this must not
    # invent a missing prerequisite out of a `-D` value or a target name.
    toks = [t for a in argv[1:] for t in str(a).split()]
    for i, tok in enumerate(toks[:-1]):
        if tok in ("source", "."):
            cand = toks[i + 1].strip("\"'")
            if cand.startswith("-"):
                continue
            full = cand if os.path.isabs(cand) else os.path.join(cwd, cand)
            obs.append({"what": "env script", "path": cand,
                        "present": os.path.exists(full)})
    return obs


def cmd_prove(args):
    """Execute the discovered build and run, then parse the accuracy evidence.

    The tool runs the commands so that the numbers in proof.json come from an execution it
    witnessed. What it does not do is invent a verification strategy: build entry, run argv
    and cwd all come from the profile and the analyst's findings.
    """
    run_dir, plan = load_run(args.run_dir)
    units, _ = fold(run_dir, plan)
    if args.unit not in units:
        raise Fail(f"no unit {args.unit!r} in plan.json")
    row = units[args.unit]
    if RANK[row["phase"]] < RANK["IMPLEMENTED"]:
        raise Fail(f"unit is at {row['phase']}; there is nothing built to prove")
    require_current_migration_authorization(run_dir, row)
    # A parked unit is one whose confirmation covered a claim that turned out false. `check`
    # already refuses one; `prove` did not, so a park could still build, run on the device and
    # write a proof -- and the fold then reported `PARKED (was PROVEN)`, a proof for work that
    # was surfaced as blocked.
    if row["parked"]:
        raise Fail(f"unit is PARKED ({row['parked']}). A park means the recorded confirmation no "
                   f"longer describes the work, so there is nothing to prove yet: settle it with "
                   f"the user, `park --unit {args.unit} --unpark`, and re-confirm before proving.")
    if row["excluded"]:
        raise Fail(f"unit was excluded at CONFIRM; it stays at ANALYZED and owes no proof. "
                   f"Re-present the packet and confirm without excluding it first.")

    udir = unit_dir(run_dir, args.unit)
    f = load_json(os.path.join(udir, "findings.json"), "findings")
    prove = f["prove"]
    cwd = contained(plan["target_root"], prove.get("cwd") or ".", "findings.prove.cwd")
    access = load_access(run_dir)
    side = access_for_arch(access, TARGET_ARCH)
    hardware = hardware_identity(side) if (side or {}).get("reachable") else None
    logs = os.path.join(udir, "logs")
    os.makedirs(logs, exist_ok=True)

    # Build and run are separate outcomes, not two halves of one. Cross-compiling for the
    # target arch works on any recent-CANN box whatever NPU it has -- which presumes a CANN box,
    # so the preflight below records whether this one is, instead of letting an absent toolkit
    # arrive as a compile error. Welding the build to the device run meant a device-less machine
    # recorded `verdict: FAIL, at: run` over a compile that succeeded, and the successful build
    # survived only inside a `stages` array under a FAIL verdict.
    records, art = [], []
    env_obs = toolchain_preflight(prove["build"], cwd)
    missing = [o["path"] for o in env_obs if not o["present"]]
    if env_obs:
        print("toolchain: " + ", ".join(
            f"{o['path']} {'ok' if o['present'] else 'MISSING'}" for o in env_obs))
    if missing:
        print("  the build is attempted anyway: this reading cannot see a toolkit supplied by "
              "a container, a module system or a login profile. But it is the first thing to "
              "rule out if the build fails, and the failure is then recorded as environmental "
              "rather than as a verdict on the migrated code.")

    def fail_proof(at, msg, ev=None, log=None):
        """One exit for every failure: write proof.json, append the failure, then raise.

        The append is what stops a stale pass from outliving the run that refuted it. It sets
        an overlay rather than a rank, so a unit that once proved keeps the rank it earned
        while `status` and `report` show that the latest attempt failed.

        A build failure on a machine whose preflight came back short is attributed to the
        environment, in the artifact and in the log, because the alternative is a report that
        accuses the migration of not compiling on a box that cannot compile anything.

        The cited log is copied aside first. `run_stage` truncates, so a retry overwrote the
        very log the failure event pointed at: all three `unit.prove_failed` events of one
        campaign ended up citing a file that by then held a *successful* build, which is the
        revocation record contradicting itself.
        """
        cause = "environment" if at == "build" and missing else None
        if cause:
            msg += (" The toolchain preflight found " + ", ".join(missing) + " absent, so this "
                    "is recorded as an environment failure and says nothing about the migrated "
                    "code. Install or source the toolchain and re-run `prove`.")
        kept = None
        if log and os.path.exists(log):
            kept = f"{os.path.splitext(log)[0]}.failed-{now().replace(':', '')}.log"
            shutil.copy2(log, kept)
            msg += f" This attempt's log is preserved at {kept}."
        doc = {"unit": args.unit, "verdict": "FAIL", "at": at, "stages": records,
               "environment": env_obs, "artifacts": art, "hardware": hardware, "evidence": ev,
               "ts": now()}
        if cause:
            doc["cause"] = cause
        if kept:
            doc["failed_log"] = kept
        dump_json(os.path.join(udir, "proof.json"), doc)
        append_event(run_dir, "unit.prove_failed", unit=args.unit, at=at, cause=cause,
                     missing=missing if cause else [], log=kept, detail=msg[:400])
        return Fail(msg)

    blog = os.path.join(logs, "prove-build.log")
    code, _ = run_stage(prove["build"], cwd, blog, args.timeout)
    records.append({"stage": "build", "argv": prove["build"], "exit": code, "log": blog})
    print(f"build: exit {code}  ({blog})")
    if code != 0:
        raise fail_proof("build", f"build failed with exit {code}; see {blog}. Nothing "
                                  "downstream of a failed build means anything.", log=blog)
    # The compile is a result in its own right and is recorded before the device is touched,
    # so it survives whatever the run does next.
    append_event(run_dir, "unit.compiled", unit=args.unit, argv=prove["build"], log=blog)
    print(f"recorded: unit {args.unit} COMPILED")

    if args.compile_only:
        print("--compile-only: stopping before the device run. The unit is COMPILED, not "
              "PROVEN; accuracy is not established and must not be reported as such.")
        return 0

    # Device evidence is valid only on the target architecture; compile remains recorded when
    # no matching device is declared.
    if not (side or {}).get("reachable"):
        raise Fail(
            f"no reachable {TARGET_ARCH} device is declared in access.json. Unit {args.unit} "
            f"is COMPILED; device accuracy is not established.")

    # A remote declaration needs transport unless its host is this machine.
    tr = transport_of(side)
    if tr is None and not host_is_this_machine(side.get("host")):
        raise Fail(
            f"access.a5 puts the {TARGET_ARCH} device on `{side.get('host')}`, which is not "
            f"this machine, and declares no transport. Unit {args.unit} is COMPILED; device "
            "accuracy is not established.")

    # What this run is about to execute, dated against the campaign. The build succeeding says
    # a target was built; it does not say it was *this* one, because the run names a leaf binary
    # and a target name need not equal its directory name. The campaign's own start is the
    # anchor: an artifact older than `plan.frozen` cannot have been produced here, whatever the
    # build did. Anything newer is recorded and not adjudicated -- whether an unchanged rebuild
    # rewrites the file is the tree's business, not this engine's.
    art.extend(run_artifacts(plan["target_root"],
                             effective_run_argv(prove["run"]) if tr is None
                             else (prove.get("stage") or []),
                             cwd if tr is None else plan["target_root"]))
    started = campaign_start(run_dir)
    named = "the run argv names" if tr is None else "findings.prove.stage carries"
    for a in art:
        a["age_vs_campaign_s"] = None if a["mtime"] is None else a["mtime"] - started
        print(f"artifact: {a['path']} "
              + ("MISSING" if not a["present"] else f"{a['age_vs_campaign_s']:+d}s vs campaign"))
        if not a["present"]:
            raise fail_proof(
                "run",
                f"{named} {a['path']}, which does not exist after a build that succeeded. "
                f"Either the build entry produced a different leaf target than this run names, "
                f"or the path is wrong; nothing may be executed until that is settled.")
        if a["age_vs_campaign_s"] < 0:
            raise fail_proof(
                "run",
                f"{named} {a['path']}, which is {-a['age_vs_campaign_s']}s OLDER than this "
                f"campaign — so the build that just succeeded did not produce it and it is a "
                f"leftover from an earlier round. Executing it would compare a previous round's "
                f"binary against this contract's golden and record `errors=0` for work this "
                f"campaign never built. Delete it, confirm the build entry's target really is "
                f"the leaf this run names, and re-run `prove`.")
    if tr is None and not art:
        raise fail_proof(
            "run",
            "the local findings.prove.run resolves no in-tree executable or input artifact. "
            "Nothing may execute until its argv names at least one existing, campaign-fresh "
            "artifact under the target root; otherwise PROVEN could describe an undated binary.")
    rlog = os.path.join(logs, "prove-run.log")
    if tr is None:
        ran_at, exec_argv = "local", prove["run"]
        code, stdout = run_stage(prove["run"], cwd, rlog, args.timeout)
    else:
        # The build ran here, so nothing of this unit exists over there yet. What travels is
        # the analyst's own list -- the artifact the profile discovered, plus every input the
        # run argv reads -- and it lands in the declared workdir and nowhere else.
        ran_at = remote_label(tr)
        rel_stage = prove.get("stage") or []
        if not rel_stage:
            raise fail_proof(
                "run",
                f"the {TARGET_ARCH} device is at {ran_at} and this unit's findings.prove "
                f"declares no `stage`, so the binary this build just produced does not exist "
                f"there. List the artifact and every input the run argv reads, as "
                f"tree-relative paths, in findings.prove.stage, then re-run `prove`.")
        srcs = []
        for i, rel in enumerate(rel_stage):
            p = contained(plan["target_root"], rel, f"findings.prove.stage[{i}]")
            if not os.path.exists(p):
                raise fail_proof(
                    "run",
                    f"findings.prove.stage[{i}] is {rel!r}, which does not exist after a build "
                    f"that succeeded. Either the build does not produce it or the path is "
                    f"wrong; nothing can reach {ran_at} until that is settled.")
            srcs.append(p)
        slog = os.path.join(logs, "prove-stage.log")
        argv, env = ssh_argv(tr, f"mkdir -p {shlex.quote(tr['workdir'])}")
        code, _ = run_stage(argv, cwd, slog, args.timeout, env)
        if code == 0:
            argv, env = scp_argv(tr, srcs, f"{ssh_target(tr)}:{tr['workdir']}/")
            code, _ = run_stage(argv, cwd, slog, args.timeout, env, mode="a")
        records.append({"stage": "stage", "argv": argv, "where": ran_at, "exit": code,
                        "log": slog})
        print(f"stage: exit {code}  ({slog}; {len(srcs)} path(s) -> {ran_at})")
        if code != 0:
            raise fail_proof("run", f"staging to {ran_at} failed with exit {code}; see {slog}. "
                                    f"Nothing ran on the device.", log=slog)
        exec_argv, env = ssh_argv(tr, remote_command(tr, prove["run"]))
        code, stdout = run_stage(exec_argv, cwd, rlog, args.timeout, env)
    records.append({"stage": "run", "argv": prove["run"], "exec": exec_argv, "where": ran_at,
                    "exit": code, "log": rlog})
    print(f"run: exit {code}  ({rlog}; {ran_at})")
    if code != 0:
        raise fail_proof("run", f"run failed with exit {code}; see {rlog}. The unit stays "
                                "COMPILED: it builds, and its accuracy is not established.",
                         log=rlog)

    # Every evidence-stage refusal leaves the same record. `proof.json` is the only witness a
    # reader gets for a run that reached the device.
    def refuse(msg, ev=None):
        return fail_proof("evidence", msg, ev, log=rlog)

    found = EVIDENCE_RE.findall(stdout)
    if not found:
        raise refuse(
            "the run produced no CATLASS_EVIDENCE line, so there is no number to check.\n"
            "  The migrated example must print exactly one line of the form:\n"
            '    CATLASS_EVIDENCE {"shape":[...],"dtype":"fp16","computeNum":4096,'
            '"errors":0,"golden_sumsq":1234.5}\n'
            "  A `Compare success.` string is not evidence: it carries no numbers, so it "
            "cannot distinguish a real pass from a comparison that never ran.")
    # A proof is one marked evidence record for the frozen contract shape.
    if len(found) > 1:
        raise refuse(
            f"the run printed {len(found)} CATLASS_EVIDENCE lines; `prove` requires exactly one. "
            f"The verdict is a claim about a single execution at the frozen contract shape, so "
            f"there is no rule for reconciling several -- and taking the first would let a "
            f"passing line hide a failing one printed after it. A sweep across shapes or dtypes "
            f"is breadth, not the contract: it is not what PROVEN claims, so keep the run "
            f"stage printing the contract shape once.")
    try:
        ev = json.loads(found[0])
    except json.JSONDecodeError as exc:
        raise refuse(f"the CATLASS_EVIDENCE payload is not valid JSON: {exc}")
    if not isinstance(ev, dict) or "errors" not in ev:
        raise refuse("the CATLASS_EVIDENCE object must carry at least an 'errors' count", ev)
    errors = ev["errors"]
    if type(errors) is not int:
        raise refuse(f"CATLASS_EVIDENCE.errors must be an integer, got {errors!r}", ev)
    # `computeNum` is the comparator's tolerance selector, not an element count, so a migrated
    # example that hands it a different local variable than the source did is comparing on an
    # argument nobody reviewed and can still print `errors: 0`. The frozen contract names the
    # value; this is where the two are made to agree.
    frozen = (f.get("contract") or {}).get("golden") or {}
    expected = frozen.get("compute_num") if isinstance(frozen, dict) else None
    if type(expected) is int:
        if "computeNum" not in ev:
            raise refuse("the CATLASS_EVIDENCE object carries no 'computeNum'. The frozen "
                         f"contract records compute_num = {expected}, and it selects the "
                         "tolerance band, so the run has to show which value it actually passed "
                         "to the comparator.", ev)
        if ev["computeNum"] != expected:
            raise refuse(
                f"CATLASS_EVIDENCE.computeNum is {ev['computeNum']!r} but the frozen contract's "
                f"golden.compute_num is {expected}. The value selects the tolerance band, so "
                "the migrated example must pass the same value as the frozen source contract.", ev)
    expected_tensor = frozen.get("compared_tensor") if isinstance(frozen, dict) else None
    expected_dtype = frozen.get("compared_dtype") if isinstance(frozen, dict) else None
    if not isinstance(expected_tensor, str) or not expected_tensor \
            or not isinstance(expected_dtype, str) or not expected_dtype:
        raise refuse("findings.contract.golden must freeze compared_tensor and compared_dtype "
                     "before a run can establish accuracy; re-run `check --phase analyzed` and "
                     "re-present the corrected Gate-1 packet.", ev)

    # --- what was actually executed --------------------------------------------------------
    # PROVEN establishes the frozen contract shape by exact positional list equality.
    frozen_shape = prove.get("shape")
    if not isinstance(frozen_shape, list) or not frozen_shape \
            or not all(type(v) is int for v in frozen_shape):
        raise refuse("findings.prove.shape must be a non-empty list of integers before anything "
                     "can be proven at it; re-run `check --phase analyzed` on this unit.", ev)
    if "shape" not in ev:
        raise refuse(f"the CATLASS_EVIDENCE object carries no 'shape'. PROVEN claims accuracy at "
                     f"one shape and one only -- the frozen {frozen_shape} -- so the run has to "
                     f"name the shape it executed, or the claim has no subject.", ev)
    if not isinstance(ev["shape"], list) or not all(type(v) is int for v in ev["shape"]):
        raise refuse(f"CATLASS_EVIDENCE.shape must be a list of JSON integers, got "
                     f"{ev['shape']!r}. A quoted number or a `true` here would compare equal to "
                     f"something it is not.", ev)
    if ev["shape"] != frozen_shape:
        raise refuse(
            f"CATLASS_EVIDENCE.shape is {ev['shape']} but findings.prove.shape froze "
            f"{frozen_shape}. The run therefore proves accuracy at a shape the contract did not "
            f"freeze, while report.md would go on claiming the frozen one -- a smaller debug "
            f"case passing says nothing about the contract shape's tails, tiling or multi-core "
            f"split. Run the frozen argv, or, if the frozen shape was recorded wrong, fix "
            f"findings.prove.shape and re-`check --phase analyzed` so the change is on the "
            f"record rather than absorbed by the proof.", ev)

    # `shape` is the example's own summary of what it ran, so it is only as honest as its printf.
    # `argv` is the stronger form and needs no knowledge of the operator's argv grammar at all:
    # the example echoes the arguments it was handed and `prove` compares them against the argv
    # it executed. The *tail* of that argv, because `findings.prove.run` may be a wrapper whose
    # leading words never reach main(). It stays optional: an evidence line with `shape` alone is
    # still checked above, and this tells the implementer what to add rather than failing them.
    echoed = ev.get("argv")
    if echoed is None:
        print('note: no "argv" in the evidence line. Echoing the arguments main() received, as '
              'strings -- "argv":["256","512","1024","0"] -- lets `prove` check the run against '
              "the argv it executed, which is stronger than the shape echo it can check today.")
    else:
        if not isinstance(echoed, list) or not echoed \
                or not all(isinstance(a, str) for a in echoed):
            raise refuse(f"CATLASS_EVIDENCE.argv must be a non-empty list of strings -- the "
                         f"arguments main() received, verbatim -- got {echoed!r}. An example "
                         f"that takes no arguments cannot have a frozen shape either, since "
                         f"findings.prove.shape is that argv minus the device id.", ev)
        words = effective_run_argv(prove["run"])
        tail = words[len(words) - len(echoed):] if len(echoed) <= len(words) else None
        if tail != echoed:
            raise refuse(
                f"CATLASS_EVIDENCE.argv is {echoed} but `prove` executed "
                f"{words}, whose trailing {len(echoed)} argument(s) are {tail}. The "
                f"example did not receive what this run handed it, so it either parses its "
                f"command line differently than findings.prove.run assumes or ignores it and "
                f"runs a hardcoded case. Either way the numbers below describe a different "
                f"execution than the one recorded in proof.json.", ev)

    # One evidence dtype must identify one frozen tensor. `compared_dtype` is deliberately the
    # canonical evidence string, not a translation of the tensor's source dtype prose: a golden
    # may use upgraded precision (for example ElementGolden = float) while storage and Cube
    # operand descriptions remain different contractual facts.
    if not isinstance(ev.get("dtype"), str) or not ev["dtype"].strip():
        raise refuse("CATLASS_EVIDENCE.dtype must be a non-empty string naming the frozen "
                     f"golden.compared_tensor {expected_tensor!r}.", ev)
    if ev["dtype"] != expected_dtype:
        raise refuse(
            f"CATLASS_EVIDENCE.dtype is {ev['dtype']!r} but the frozen "
            f"golden.compared_dtype for tensor {expected_tensor!r} is {expected_dtype!r}. "
            "The label is a canonical witness field and must match exactly; do not normalize "
            "or relabel it from storage, Cube-operand, or upgraded-golden precision.", ev)

    print(f"evidence: {json.dumps(ev, ensure_ascii=False)}")
    if errors != 0:
        # Through `fail_proof`, not a bare raise. This is the commonest failure of all -- the
        # comparison ran and disagreed -- and it was the one path that recorded nothing in the
        # log, so a unit that passed last week and fails today kept reporting `errors=0`.
        raise fail_proof(
            "evidence",
            f"{errors} element(s) outside tolerance against the CPU golden. Triage per "
            "references/06-prove.md: verify the golden itself first — if the golden is "
            "wrong every later comparison is meaningless.", ev, log=rlog)
    dump_json(os.path.join(udir, "proof.json"),
              {"unit": args.unit, "verdict": "PASS", "at": None, "stages": records,
               "environment": env_obs, "artifacts": art, "hardware": hardware, "evidence": ev,
               "ts": now()})
    append_event(run_dir, "unit.proven", unit=args.unit, errors=0,
                 shape=ev.get("shape"), dtype=ev.get("dtype"), where=ran_at,
                 hardware=hardware)
    print(f"unit {args.unit}: PROVEN")
    return 0


# --- status / report ---------------------------------------------------------------------

# Every entry is formatted with `r` (the quoted run directory) and `u` (the unit id), because a
# printed command that omits `--run-dir` cannot be run as printed -- and `--run-dir` is required
# on every subcommand except `init`.
NEXT = {
    "ANALYZED": "`gate --run-dir {r}`, then "
                "`confirm --run-dir {r} --intent '<final human decision>'`",
    "AUTHORIZED": "land the ledger's shared declarations and run the source-arch regression, "
                  "then dispatch one Implementer for this unit and land its registration "
                  "text; then `check --run-dir {r} --phase implemented --unit {u}`",
    "IMPLEMENTED": "`prove --run-dir {r} --unit {u}` "
                   "(add `--compile-only` when no device for this unit's target architecture "
                   "is reachable)",
    "COMPILED": "it builds; accuracy is not established. With a device: "
                "`prove --run-dir {r} --unit {u}`. Without one: nothing further is reachable "
                "for this unit -- `report --run-dir {r}` will say so.",
    "PROVEN": "`report --run-dir {r}`",
    "OPT_SCREENED": "the optimization gate, once for the whole campaign: "
                    "`gate --run-dir {r} --phase optimize`, then "
                    "`confirm --run-dir {r} --phase optimize "
                    "--intent '<final human decision>'`",
    # Four steps in one entry because `unit.applied` does not promote: the unit sits at
    # OPT_AUTHORIZED from the moment the gate authorizes it until the re-prove is checked, so
    # this string is the only place the operator is told the order the log will be read in.
    "OPT_AUTHORIZED": "land the rewrite at the authorized paths, measure it and write "
                      "units/{u}/optimize.json; then record the landing with "
                      "`check --run-dir {r} --phase applied --unit {u}`, re-run "
                      "`prove --run-dir {r} --unit {u}` (the rewrite is the default build, so "
                      "that run is its accuracy record), and finally "
                      "`check --run-dir {r} --phase optimized --unit {u}`",
    "OPTIMIZED": "nothing; `report --run-dir {r}`",
}


def next_command(row, camp, plan, run_dir, access):
    """The one command to run next for this unit, ready to paste."""
    fmt = {"r": shlex.quote(run_dir), "u": row["id"]}
    if row["parked"]:
        # A `redesign` unit is parked by `confirm` itself and re-parked on every confirmation, so
        # telling the operator to unpark it sends them round a loop the engine will undo.
        if row["route"] == ROUTE_TERMINAL:
            return (f"report only — route {ROUTE_TERMINAL} changes the external contract, so this "
                    f"is not a migration and `confirm` re-parks it every time: "
                    f"`report --run-dir {fmt['r']}`")
        return (f"settle the park with the user, then `park --run-dir {fmt['r']} --unit {row['id']} "
                f"--unpark`, re-run `check --run-dir {fmt['r']} --phase analyzed --unit "
                f"{row['id']}`, then render GATE 1 and record replacement confirmation")
    recovery = (status_authorization_recovery(run_dir, row)
                if RANK[row["phase"]] >= RANK["AUTHORIZED"] else None)
    if recovery:
        return recovery
    if row["excluded"]:
        # Exclusion is defined by the LATEST confirmation, so re-admission is a re-presentation
        # and a confirmation that omits the id -- not a re-analysis.
        return (f"excluded at the migration gate; to re-admit it, re-present the packet "
                f"(`gate --run-dir {fmt['r']}`) and `confirm --run-dir {fmt['r']} --intent '...'` "
                f"without --exclude {row['id']}")
    if row["opt_excluded"]:
        return (f"set aside at the optimization gate — its migration stands and is reported. "
                f"To re-admit it, re-present the packet "
                f"(`gate --run-dir {fmt['r']} --phase optimize`) and confirm without "
                f"--exclude {row['id']}")
    if row["phase"] == "INTAKE":
        # The campaign-level prerequisites, in the order the engine enforces them: ANALYZE
        # reads profile.json, and `check --phase analyzed` refuses until it is recorded.
        if not camp["refs"]:
            return f"`refs --run-dir {fmt['r']}`"
        if not camp["profile"]:
            return (f"probe this tree's build / golden / registration / arch-gating "
                    f"conventions into {fmt['r']}/profile.json; then "
                    f"`profile --run-dir {fmt['r']}`")
        return (f"analyze it; then "
                f"`check --run-dir {fmt['r']} --phase analyzed --unit {row['id']}`")
    if RANK[row["phase"]] >= RANK["PROVEN"]:
        proof, issue = current_proof(row, access)
        if proof is None:
            return ("accuracy is NOT ESTABLISHED (" + issue + "); re-run "
                    f"`prove --run-dir {fmt['r']} --unit {row['id']}` on the declared "
                    "matching Ascend950 hardware before resuming this phase")
    if row["phase"] == "PROVEN":
        if row["opt_skipped"]:
            return (f"nothing — optimization was intentionally skipped at the migration gate: "
                    f"`report --run-dir {fmt['r']}`")
        if not plan.get("optimize", {}).get("enabled"):
            return (f"nothing — this plan opted out of the optimization phases: "
                    f"`report --run-dir {fmt['r']}`")
        return (f"screen it against the five applicability rows and measure its baseline: "
                f"`check --run-dir {fmt['r']} --phase screened --unit {row['id']}`")
    if row["phase"] == "OPT_SCREENED":
        # A unit the rows exclude has nothing further to do, and saying "go to the gate" would
        # send it to a packet that authorizes nothing. OPT_SCREENED is its terminal state.
        spath = os.path.join(unit_dir(run_dir, row["id"]), "screen.json")
        if os.path.exists(spath) and not load_json(spath, "screen").get("applicable"):
            return (f"nothing — the five rows say the L0C->UB rewrite does not apply here, so "
                    f"this is where the unit stops. Its screen and its measured baseline are "
                    f"in the report: `report --run-dir {fmt['r']}`")
    return NEXT[row["phase"]].format(**fmt)


def cmd_status(args):
    run_dir, plan = load_run(args.run_dir)
    units, camp = fold(run_dir, plan)
    access = load_access(run_dir, strict=False)
    print(f"run: {run_dir}")
    print(f"refs: {'resolved' if camp['refs'] else 'NOT RESOLVED — run `refs`'}   "
          f"profile: {'answered' if camp['profile'] else 'NOT ANSWERED — run `profile`'}")
    print(f"migration gate: {'confirmed' if camp['confirmed'] else 'NOT CONFIRMED — owed'}   "
          f"optimization gate: "
          f"{'confirmed' if camp['opt_confirmed'] else 'not yet — owed once units are screened'}")
    # Compare the engine revision recorded at initialization with the current engine.
    frozen, current = camp["skill_head"], skill_head()
    if frozen and frozen != current:
        print(f"engine: pinned at {frozen} by `init`, now {current} — the skill changed "
              f"underneath this campaign; artifacts before that point were produced by a "
              f"different engine")
    else:
        print(f"engine: {current}")
    print()
    print(f"{'unit':<8} {'phase':<14} {'route':<13} state")
    for row in sorted(units.values(), key=lambda r: r["id"]):
        state = "parked: " + row["parked"] if row["parked"] else (
            "migration authorization INVALIDATED — re-analysis and replacement GATE-1 "
            "authorization required" if row["authorization_invalidated"] else
            "excluded at the migration gate" if row["excluded"] else
            "set aside at the optimization gate" if row["opt_excluded"] else
            "optimization intentionally skipped at the migration gate"
            if row["opt_skipped"] else "")
        proof, issue = current_proof(row, access)
        if row["prove_failed"] or (RANK[row["phase"]] >= RANK["PROVEN"]
                                   and proof is None):
            base_state = state
            state = "accuracy is NOT ESTABLISHED — " + issue
            authorization_recovery = (status_authorization_recovery(run_dir, row)
                                      if RANK[row["phase"]] >= RANK["AUTHORIZED"] else None)
            if authorization_recovery:
                state += "; migration grant recovery: " + authorization_recovery
            elif base_state:
                state += f"; {base_state}"
        print(f"{row['id']:<8} {row['phase']:<14} {(row['route'] or '-'):<13} {state}")
    print("\nnext:")
    for row in sorted(units.values(), key=lambda r: r["id"]):
        print(f"  {row['id']}: " + next_command(row, camp, plan, run_dir, access))
    return 0


def cmd_report(args):
    run_dir, plan = load_run(args.run_dir)
    units, camp = fold(run_dir, plan)
    access = load_access(run_dir, strict=False)
    out = [f"# Migration report", "",
           f"- run directory: `{run_dir}`",
           f"- target root: `{plan['target_root']}`",
           f"- request as recorded: {plan['request']}",
           f"- performance cases: {perf_cases_state(run_dir)}"]
    if camp["skill_head"]:
        out.append(f"- engine at `init`: `{camp['skill_head']}` (now `{skill_head()}`)")
    if camp["refs"]:
        for name, info in sorted(camp["refs"].items()):
            out.append(f"- reference `{name}` @ `{info['rev']}` ({info['how']})")
    if camp["confirmed"]:
        c = camp["confirmed"]
        out += [f"- confirmed by `{c.get('actor')}`: {c.get('intent')}"]
        if c.get("excluded"):
            out.append(f"- excluded at the gate: {', '.join(c['excluded'])}")
        if c.get("skip_optimize"):
            out.append("- optimization intentionally skipped after PROVEN for: "
                       + ", ".join(c["skip_optimize"]))
    else:
        out.append("- **not confirmed**: no unit was authorized to write to the target tree")
    if camp["opt_confirmed"]:
        # The second gate's intent was recorded and never rendered, so the reason a rewrite was
        # authorized -- and, when a manifest had to be corrected, the correction and its
        # argument -- lived only in `events.jsonl`. The optimization half of the report could
        # not be read against what was actually authorized.
        o = camp["opt_confirmed"]
        out.append(f"- optimization gate, confirmed by `{o.get('actor')}`: {o.get('intent')}")
        if o.get("excluded"):
            out.append(f"- excluded at the optimization gate: {', '.join(o['excluded'])}")

    out += ["", "## Units", "",
            "| unit | source | target | route | phase | accuracy |", "|---|---|---|---|---|---|"]
    not_established = []
    for row in sorted(units.values(), key=lambda r: r["id"]):
        proof, issue = current_proof(row, access)
        if RANK[row["phase"]] < RANK["PROVEN"]:
            acc = "not proven"
        elif proof is None:
            acc = f"**NOT ESTABLISHED** — {issue}"
            not_established.append((row, issue))
        else:
            acc = "errors=0"
            identity = ", ".join(f"{key}={proof['hardware'][key]}"
                                 for key in ("arch", "soc", "host", "device"))
            acc += "<br/>declared " + identity.replace("|", "\\|")
        phase = (f"PARKED (was {row['phase']})" if row["parked"] else
                 (f"AUTHORIZATION INVALIDATED (was {row['phase']})"
                  if row["authorization_invalidated"] else row["phase"]))
        out.append(f"| {row['id']} | `{row['source']}` | `{row['target']}` | "
                   f"{row['route'] or '-'} | {phase} | {acc} |")

    parked = [r for r in units.values() if r["parked"]]
    if parked:
        out += ["", "## Parked", ""]
        for r in sorted(parked, key=lambda r: r["id"]):
            out.append(f"- unit {r['id']} at {r['phase']}: {r['parked']}")

    # The ledger is recomputed rather than read: it has no artifact of its own since the
    # sequencing phase was removed, and recomputing it from the same findings the gate read is
    # cheaper than persisting a file whose only reader was the packet.
    try:
        led = compute_ledger(run_dir, plan, units)
    except Fail:
        led = None
    if led is not None:
        out += ["", "## Shared components", "",
                "| path | symbol | kind | declared by | existing consumers |",
                "|---|---|---|---|---|"]
        for r in led["landable"]:
            e = r["existing"]
            existing = ("—" if not r["consumers_of"] else
                        "not measured" if e is None else
                        f"{e['count']} of `{r['consumers_of']}`")
            out.append(f"| `{r['path']}` | `{r['symbol']}` | {r['kind']} | "
                       f"{', '.join(r['declared_by'])} | {existing} |")
        if not led["landable"]:
            out.append("| _none_ | | | | |")
        out += ["", "Existing-consumer counts are `git grep -l` over the target tree at report "
                    "time — the regression obligation, measured rather than asserted. "
                    "Registration surfaces are landed after each unit's directory exists; the "
                    "rest are landed before any unit."]

    # Screened units have an artifact; intentionally skipped units stop at PROVEN.
    screened, unscreened, intentionally_skipped = [], [], []
    for row in sorted(units.values(), key=lambda r: r["id"]):
        if RANK[row["phase"]] < RANK["PROVEN"]:
            continue
        if row["opt_skipped"]:
            intentionally_skipped.append(row["id"])
            continue
        udir = unit_dir(run_dir, row["id"])
        spath = os.path.join(udir, "screen.json")
        if not os.path.exists(spath):
            unscreened.append(row["id"])
            continue
        opath = os.path.join(udir, "optimize.json")
        screened.append((row["id"], load_json(spath, f"screen for unit {row['id']}"),
                         load_json(opath, f"optimize for unit {row['id']}")
                         if os.path.exists(opath) else None))
    if screened:
        out += ["", "## L0C→UB rewrite", "",
                "| unit | applies | strategy | mode | baseline (us, epoch) | rewritten (us) "
                "| resolves? |",
                "|---|---|---|---|---|---|---|"]
        unresolved = []
        for uid, s, o in screened:
            us = {c["config"]: c["task_us"] for c in ((o or {}).get("profile") or [])}
            base, epoch = us.get("baseline"), "APPLY"
            if base is None:
                base, epoch = (s.get("baseline") or {}).get("task_us", "-"), "SCREEN"
            # Overlapping sample ranges do not establish a performance ordering.
            verdict = "—"
            if "l0c_to_ub" in us and base != "-":
                nb, nr = us_stat(base), us_stat(us["l0c_to_ub"])
                if nb["n"] < 2 or nr["n"] < 2:
                    verdict = "**unmeasurable** (n=1)"
                elif nr["max"] < nb["min"]:
                    verdict = "yes"
                elif nr["min"] > nb["max"]:
                    verdict = "**slower**"
                else:
                    verdict = "**no — ranges overlap**"
                    unresolved.append(uid)
            out.append(f"| {uid} | {'yes' if s.get('applicable') else 'no'} | "
                       f"{s.get('strategy') or '-'} | {(o or {}).get('mode', '-')} | "
                       f"{us_text(base) if base != '-' else '-'} ({epoch}) | "
                       f"{us_text(us['l0c_to_ub']) if 'l0c_to_ub' in us else '-'} | "
                       f"{verdict} |")
        out += ["",
                "Durations are transcribed from profiler output retained under each unit's "
                "`logs/`. A multi-sample value is displayed as a median with its sample count "
                "and range; a single sample has no error bar. The baseline column identifies "
                "whether it was recorded at SCREEN or APPLY.", "",
                "**`resolves?` compares sample ranges, not medians.** `yes` means every "
                "rewritten sample beat every baseline sample. Overlap does not establish an "
                "ordering.", "",
                "The human selected the data path at the optimization gate; measurements "
                "describe that outcome."]
        if unresolved:
            out.append(f"- **not established**: on unit(s) {', '.join(unresolved)} the "
                       f"improvement does not clear the measured spread of these very runs.")
        for uid, s, o in screened:
            if o and s.get("strategy") == "replace":
                out.append(f"- unit {uid}: `replace` — the pre-rewrite path was removed, and "
                           f"this run directory holds nothing to restore it from.")
            if s.get("applicable") and not o:
                out.append(f"- unit {uid}: screened as applicable, but no rewrite is recorded "
                           f"(`units/{uid}/optimize.json` is absent).")

    out += ["", "## Not established", "",
            "- Accuracy is the frozen contract's own shape only. No sweep across other shapes, "
            "dtypes, tails or zero-work was run, and nothing here claims one.",
            "- No sanity control runs, so an all-zero output compared against an all-zero "
            "golden would pass undetected.",
            "- Evidence content in `findings.json` is schema-checked for structure but not "
            "verified against the source tree."]
    for row, issue in not_established:
        out.append(f"- unit {row['id']}: accuracy is **NOT ESTABLISHED** because {issue}. "
                   "Its recorded rank is historical; re-run `prove` on the declared matching "
                   "Ascend950 hardware before screening, entering GATE 2, applying, or "
                   "optimizing.")
    # Report artifacts that changed after the migration packet was rendered.
    for uid, was in sorted(((camp["gate"] or {}).get("digests") or {}).items()):
        fpath = os.path.join(unit_dir(run_dir, uid), "findings.json")
        if uid in units and os.path.exists(fpath) and artifact_digest(fpath) != was:
            out.append(f"- unit {uid}: `findings.json` differs from the copy the migration gate "
                       f"rendered (`{was}` then, `{artifact_digest(fpath)}` now). Anything added "
                       f"to it since — a shared component discovered while implementing, a "
                       f"corrected path — was **not** in the packet the human approved. The "
                       f"artifact says what changed and why; this line says the approval "
                       f"predates it.")
    if not os.path.exists(os.path.join(run_dir, PERF_CASES)):
        out.append(f"- no performance case table is present (`{PERF_CASES}` is absent).")
    for row in sorted(units.values(), key=lambda r: r["id"]):
        pf = row["prove_failed"] or {}
        if pf.get("cause") != "environment":
            continue
        # The one failure in this report that is not about the code. Left out, a reader sees a
        # column of unproven units and concludes the migration does not build.
        out.append(f"- unit {row['id']}: the last build attempt failed with "
                   f"{', '.join(pf.get('missing') or ['a toolchain prerequisite'])} absent from "
                   f"this machine, so **whether the migrated code builds is not on the record** "
                   f"either way. This is a statement about the machine, not the migration: "
                   f"install or source the toolchain and re-run `prove`.")
    if not plan.get("optimize", {}).get("enabled"):
        out.append("- this plan set `optimize.enabled` to false, so no unit was screened for "
                   "the L0C→UB rewrite and no unit was measured.")
    if intentionally_skipped:
        out += ["", "## Optimization intentionally skipped", ""]
        for uid in intentionally_skipped:
            out.append(f"- unit {uid}: intentionally stopped at PROVEN by the migration "
                       "confirmation; no screen, optimization gate, rewrite, or measurement "
                       "was requested.")
    for uid in unscreened:
        out.append(f"- unit {uid} reached PROVEN but was **never screened** "
                   f"(`units/{uid}/screen.json` is absent), so neither its performance nor "
                   f"whether the rewrite applies to it is known.")

    out += ["", "## Scope", "",
            "- Generic tuning, including block/core counts and `SPLIT_N`, is outside this "
            "campaign. Its only optimization work is the non-TLA L0C→UB data-path rewrite."]
    path = os.path.join(run_dir, "report.md")
    write(path, "\n".join(out))
    print(path)
    return 0


def cmd_remote(args):
    """Reach a declared remote side directly: push files, run a command, fetch results.

    A utility, not a phase: it appends no event and promotes nothing. It exists because the
    profiler runs are the lead's and their output has to end up under `units/<id>/logs/` to be
    accepted as a sample (`check_sample`), which a measurement taken on another box cannot do
    by itself. Everything it writes over there lands under the declared workdir.
    """
    run_dir, plan = load_run(args.run_dir)
    tr = transport_of(load_access(run_dir).get(args.side))
    if tr is None:
        raise Fail(f"access.{args.side} declares no ssh transport, so there is no remote box "
                   f"to reach. A local side needs no `remote`: run the command directly.")
    chosen = [name for name in ("push", "exec", "fetch") if getattr(args, name)]
    if len(chosen) != 1:
        raise Fail("choose exactly one of --push, --exec, --fetch")
    log = os.path.join(run_dir, "remote.log")

    if args.push:
        srcs = []
        for rel in args.push:
            p = contained(plan["target_root"], rel, "--push")
            if not os.path.exists(p):
                raise Fail(f"--push {rel}: no such path in the target tree")
            srcs.append(p)
        argv, env = ssh_argv(tr, f"mkdir -p {shlex.quote(tr['workdir'])}")
        code, _ = run_stage(argv, plan["target_root"], log, args.timeout, env)
        if code == 0:
            argv, env = scp_argv(tr, srcs, f"{ssh_target(tr)}:{tr['workdir']}/")
            code, _ = run_stage(argv, plan["target_root"], log, args.timeout, env, mode="a")
        print(f"push: exit {code}  ({len(srcs)} path(s) -> {remote_label(tr)}; {log})")
        return code

    if args.exec:
        argv, env = ssh_argv(tr, f"cd {shlex.quote(tr['workdir'])} && {args.exec}")
        code, out = run_stage(argv, plan["target_root"], log, args.timeout, env)
        print(out, end="" if out.endswith("\n") else "\n")
        print(f"exec: exit {code}  ({remote_label(tr)}; {log})")
        return code

    # fetch: the only path back. The destination is inside the run directory, so a result
    # brought home is covered by the same ignore rule as everything else this tool writes.
    if not args.into:
        raise Fail("--fetch needs --into: where under the run directory the result lands, e.g. "
                   "--into units/07/logs/prof-baseline-1")
    rel = args.fetch.strip()
    if rel.startswith("/") or ".." in rel.split("/"):
        raise Fail("--fetch takes a path under the declared workdir, not an absolute path or "
                   "one that climbs out of it")
    dest = contained(run_dir, args.into, "--into")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    argv, env = scp_argv(tr, [f"{ssh_target(tr)}:{tr['workdir'].rstrip('/')}/{rel}"], dest)
    code, _ = run_stage(argv, run_dir, log, args.timeout, env)
    print(f"fetch: exit {code}  ({rel} -> {dest}; {log})")
    return code


# --- cli ---------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(prog="mig.py", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="validate plan.json and create the run directory")
    p.add_argument("--plan", required=True)
    p.add_argument("--run-dir")
    p.add_argument("--access", help="path to a hardware-access declaration; staged into the "
                                    "run directory. Zero, one or both of `a2`/`a5` may be "
                                    "reachable, and none is a legal, reported answer.")
    p.add_argument("--perf-cases", help="path to a performance case table staged into the run "
                                        "directory; without it, init stages the bundled "
                                        "`assets/perf_case_template.md`.")

    for name, help_text in (("refs", "resolve the reference checkouts at their pins"),
                            ("profile", "validate the discovered target profile"),
                            ("status", "fold the log; print the next command per unit"),
                            ("report", "generate report.md")):
        sub.add_parser(name, help=help_text).add_argument("--run-dir", required=True)

    p = sub.add_parser("gate", help="render a gate packet (exits 2)")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--phase", default="migrate", choices=GATES,
                   help="which gate: `migrate` authorizes the migration writes, `optimize` "
                        "the L0C->UB rewrite's writes")

    p = sub.add_parser("check", help="validate a phase's artifacts and record the promotion")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--phase", required=True, choices=CHECKABLE,
                   help="`applied` records that the L0C->UB rewrite landed and promotes "
                        "nothing; it must precede the re-prove that `optimized` requires")
    p.add_argument("--unit")

    p = sub.add_parser("confirm", help="record the human's gate decision")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--phase", default="migrate", choices=GATES,
                   help="which gate is being confirmed; must match the packet that was shown")
    p.add_argument("--intent", required=True,
                   help="final human decision, recorded verbatim")
    p.add_argument("--exclude", help="comma-separated unit ids to set aside at this gate")
    p.add_argument("--skip-optimize", metavar="IDS",
                   help="migration gate only: comma-separated shown unit ids that intentionally "
                        "stop at PROVEN without SCREEN or the optimization gate")

    p = sub.add_parser("park", help="record a unit as blocked, or release it")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--unit", required=True)
    p.add_argument("--reason")
    p.add_argument("--unpark", action="store_true")

    p = sub.add_parser("prove", help="build, record COMPILED, then run and parse accuracy")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--unit", required=True)
    p.add_argument("--compile-only", action="store_true",
                   help="stop after the build. Use when no device for this unit's target "
                        "architecture is reachable: the unit reaches COMPILED and its "
                        "accuracy stays unestablished, which is reported, not an error.")
    p.add_argument("--timeout", type=float, default=1800)

    p = sub.add_parser("remote", help="reach the declared remote side: push, exec, fetch")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--side", default="a5", choices=ACCESS_SIDES,
                   help="which declared side to reach; defaults to the target side")
    p.add_argument("--push", action="append", metavar="REL",
                   help="tree-relative path to copy into the remote workdir; repeatable")
    p.add_argument("--exec", metavar="CMD",
                   help="command string to run in the remote workdir")
    p.add_argument("--fetch", metavar="REMOTE_REL",
                   help="path under the remote workdir to copy back")
    p.add_argument("--into", metavar="RUN_DIR_REL",
                   help="destination for --fetch, relative to the run directory")
    p.add_argument("--timeout", type=float, default=1800)

    args = ap.parse_args(argv)
    table = {"init": cmd_init, "refs": cmd_refs, "profile": cmd_profile, "check": cmd_check,
             "gate": cmd_gate, "confirm": cmd_confirm, "remote": cmd_remote,
             "park": cmd_park, "prove": cmd_prove, "status": cmd_status,
             "report": cmd_report}
    try:
        return table[args.cmd](args)
    except Fail as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    # Restore default SIGPIPE so `mig.py status | head` is a normal pipeline rather than a
    # BrokenPipeError traceback that looks like a tool failure.
    try:
        import signal
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (AttributeError, ValueError, ImportError):
        pass
    sys.exit(main())
