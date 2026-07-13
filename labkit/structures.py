"""Resolve a protein NAME to a PDB entry — deterministically, not from an LLM's memory.

WHY THIS EXISTS
---------------
The translator benchmark localised every remaining error to exactly one field. On the
covered intents (temperature, salt, force field, duration, box, water model, timestep,
protocol shape) all three local models score 100% after the deterministic contract
overwrites them — model choice is irrelevant there, which is the whole point of the
design. Every point they lose is lost recalling a PDB ID for a protein outside the
18-entry lookup table:

    request                 qwen3:8b        qwen3:14b            gpt-oss:20b
    GFP                     1gfp   (X)      "GFP"      (X)       1GFP    (X)
    HIV-1 protease          PDB:1HVP (X)    "HIV-1 protease" (X) 1HVR    (ok)
    tendamistat             "tendamistat"(X) 1A4M      (X)       1TND    (X)

Two failure modes, neither fixable by a better prompt: hallucinating a plausible-looking
ID (1GFP, 1TND, 1A4M do not contain those proteins), and not emitting an ID at all.

From the earlier analysis: d P(correct) / d q = P(not covered). The marginal value of a
smarter model is bounded by the intent mass the contract does not cover. So the correct
move is NOT to buy a bigger model — it is to shrink the uncovered set. A PDB ID is a
LOOKUP, and the PDB has a search API. Asking a 20B parameter model to recall it from
memory was always the wrong tool.

WHICH STRUCTURE, THOUGH
-----------------------
Full-text search returns what ranks highest, not what is good to simulate: "hen egg white
lysozyme" gives 1LYZ, while the curated MD choice is 1AKI. So:

    1. the curated table wins where it has an entry (these are chosen FOR MD)
    2. otherwise: full-text search, filtered to X-ray structures at <= 2.8 A with a
       single polymer entity, ranked by resolution. Deterministic given the PDB.
    3. the winner is then VERIFIED: its title must actually mention the thing asked for.
       A search hit that does not is discarded rather than trusted.

Offline, this returns None and the caller falls back to the model — with the plan clearly
marked as carrying a model-sourced structure.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

from .config import DATA_DIR

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
CACHE = DATA_DIR / "_pdb_resolve"

MAX_RESOLUTION_A = 2.8

# Words that are about the SIMULATION, not about the molecule. They pollute a full-text
# query ("lysozyme at 300 K for 100 ps" must not search for "300" or "ps").
_NOISE = re.compile(
    r"\b(simulate|simulation|run|running|md|molecular|dynamics|model|system|box|"
    r"solvate[d]?|solvent|water|ions?|salt|physiological|saline|nacl|"
    r"equilibrate|equilibration|production|minimi[sz]e|minimisation|"
    r"temperature|body|room|ambient|constant|volume|pressure|nvt|npt|"
    r"force ?field|forcefield|amber\S*|charmm\S*|opls\S*|gromos\S*|martini\S*|"
    r"tip3p|tip4p\S*|tip5p|spce?|spc/e|"
    r"padding|pad|buffer|cubic|dodecahedron|timestep|time ?step|"
    r"for|at|in|with|of|the|a|an|and|then|use|using|please)\b",
    re.I)

# Standalone numbers only. The alternation above used \b\d+\b, which put a boundary
# between the hyphen and the 1 of "HIV-1" and ate it: the query became "hiv- protease",
# degraded to the bare word "protease", and matched TRICORN protease -- a different enzyme.
_NUM = re.compile(r"(?<![\w-])\d+(?:\.\d+)?\s*(?:ns|ps|fs|nm|k|m|c|bar|molar)?(?![\w-])",
                  re.I)


def _clean(text: str) -> str:
    t = _NUM.sub(" ", text.lower())
    t = _NOISE.sub(" ", t)
    t = re.sub(r"[^a-z0-9\- ]+", " ", t)
    t = re.sub(r"(?<![\w])-|-(?![\w])", " ", t)      # strip dangling hyphens
    return " ".join(t.split())


def _get(url, data=None, timeout=25):
    req = urllib.request.Request(
        url, data=json.dumps(data).encode() if data else None,
        headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _title(pdb_id: str) -> str | None:
    """The entry's own title. None = could not check (never cache that)."""
    try:
        with urllib.request.urlopen(
                f"https://files.rcsb.org/header/{pdb_id.upper()}.pdb", timeout=20) as r:
            head = r.read().decode("utf8", "replace")
    except urllib.error.HTTPError as e:
        return "" if e.code == 404 else None
    except Exception:  # noqa: BLE001
        return None
    return " ".join(l[10:].strip() for l in head.splitlines()
                    if l.startswith(("TITLE", "COMPND"))).lower()


def search(query: str, rows: int = 25) -> list:
    """X-ray, <= MAX_RESOLUTION_A, single polymer entity — ranked by RELEVANCE.

    NOT by resolution. Sorting by resolution ascending overrides relevance completely:
    you get the sharpest structures that merely pass the FILTERS, not ones matching the
    query. "carbonic anhydrase II" came back as cellulase and cytochrome c. The quality
    constraints belong in the filter; the ranking belongs to the search engine.
    """
    q = {
        "query": {"type": "group", "logical_operator": "and", "nodes": [
            {"type": "terminal", "service": "full_text",
             "parameters": {"value": query}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "exptl.method", "operator": "exact_match",
                "value": "X-RAY DIFFRACTION"}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_entry_info.resolution_combined",
                "operator": "less_or_equal", "value": MAX_RESOLUTION_A}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                "operator": "equals", "value": 1}},
        ]},
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": rows},
            "results_content_type": ["experimental"],
        },
    }
    try:
        d = _get(SEARCH_URL, q)
    except Exception:  # noqa: BLE001
        return []
    return [h["identifier"] for h in d.get("result_set", [])]


def resolve(text: str, known: dict = None) -> dict | None:
    """name/phrase -> {'pdb_id', 'title', 'source'}  or None if it cannot be resolved.

    source = 'curated'  : from the hand-picked MD table (best structure for simulation)
             'rcsb'     : searched, quality-filtered, and title-verified against the query
             None       : offline or nothing matched -> the CALLER must fall back and say so
    """
    q = _clean(text)
    if not q:
        return None

    if known:
        # A QUALIFIER changes the protein. "T4 lysozyme" is bacteriophage T4 lysozyme
        # (~2LZM), NOT hen egg white lysozyme (1AKI) -- a different protein entirely. A
        # bare substring match on the curated table happily returned 1AKI for it. So the
        # curated entry only wins when nothing else qualifies the name.
        for name in sorted(known, key=len, reverse=True):
            m = re.search(r"(?:^|\s)(\S+\s+)?" + re.escape(name) + r"(?:\s|$)", q)
            if m:
                qualifier = (m.group(1) or "").strip()
                if not qualifier:
                    return {"pdb_id": known[name], "title": name, "source": "curated"}
                break        # qualified -> the curated pick is not safe; go and search

    CACHE.mkdir(parents=True, exist_ok=True)
    key = CACHE / (re.sub(r"[^a-z0-9]+", "_", q)[:80] + ".json")
    if key.exists():
        try:
            c = json.loads(key.read_text())
            return c or None
        except Exception:  # noqa: BLE001
            pass

    # Candidate queries, most specific first. "gfp green fluorescent protein" will never
    # appear verbatim in a title, but "green fluorescent protein" will.
    words = q.split()
    cands = []
    for c in (q, " ".join(words[-4:]), " ".join(words[-3:]), " ".join(words[-2:]),
              words[-1] if words else ""):
        if c and c not in cands:
            cands.append(c)

    # Fetch hits + titles for every candidate ONCE.
    pool = []                                    # [(cand, pid, title)]
    for cand_q in cands:
        for pid in search(cand_q):
            t = _title(pid)
            if t is None:
                return None                      # offline: do NOT cache, do NOT guess
            pool.append((cand_q, pid, t))

    def _phrase_re(cq):
        return re.compile(r"\b" + r"[\s\-]+".join(re.escape(w) for w in cq.split())
                          + r"\w{0,2}\b")

    def _words_ok(cq, title):
        ws = [w for w in cq.split() if len(w) > 3]
        return bool(ws) and all(
            re.search(r"\b" + re.escape(w) + r"\w{0,2}\b", title) for w in ws)

    # TIER 1, across EVERY candidate query: the title names the protein as a CONTIGUOUS
    # PHRASE. Tiering must be outside the candidate loop -- doing tier-1-then-tier-2 per
    # candidate let the WEAK match on a long query beat the STRONG match on a shorter one:
    # "gfp green fluorescent protein" word-matched 5DTL (mEos2, a photoconvertible protein)
    # before "green fluorescent protein" was ever tried as a phrase.
    for cand_q in cands:
        rx = _phrase_re(cand_q)
        for c, pid, t in pool:
            # _words_ok(q, ...) -- the ORIGINAL query, not the degraded candidate. A short
            # candidate is allowed to FIND a hit, never to LOWER the bar it must clear.
            if c == cand_q and rx.search(t) and _words_ok(q, t):
                out = {"pdb_id": pid, "title": t[:120], "source": "rcsb", "match": "phrase"}
                key.write_text(json.dumps(out))
                return out

    # TIER 2: every distinctive word appears AS A WORD (not as a substring -- "protein" is
    # inside "chromoprotein", which is how 3VIC, a NON-fluorescent chromoprotein, was
    # returned for GFP).
    for cand_q in cands:
        for c, pid, t in pool:
            if c == cand_q and _words_ok(q, t):
                out = {"pdb_id": pid, "title": t[:120], "source": "rcsb", "match": "words"}
                key.write_text(json.dumps(out))
                return out

    key.write_text("null")
    return None


if __name__ == "__main__":
    from .agent.intent import KNOWN_PDB
    for req in ["Simulate GFP, the green fluorescent protein, at 300 K for 100 ps",
                "Run HIV-1 protease at 310 K in physiological salt for 50 ps",
                "Simulate the alpha-amylase inhibitor tendamistat at 300 K for 50 ps",
                "Lysozyme at body temperature for 50 ps",
                "simulate carbonic anhydrase II at 300 K",
                "Water box at 300 K"]:
        r = resolve(req, KNOWN_PDB)
        print(f"  {_clean(req)!r:<38} -> "
              f"{(r['pdb_id'] + '  [' + r['source'] + ']') if r else '— (unresolved)'}"
              f"   {r['title'][:52] if r else ''}")
