"""Codebase archaeologist -- ask a repo questions in English, get cited answers.

Single file for now; splits into modules once it earns the complexity.

    python arch.py chunks  <repo>
    python arch.py bm25    <repo> <query...>
    python arch.py vec     <repo> <query...>
    python arch.py hybrid  <repo> <query...>
    python arch.py route   <repo> <query...>
    python arch.py ask     <repo> <query...>   (needs GEMINI_API_KEY)\n    python arch.py serve   <repo>\n    python arch.py eval    <repo> <questions.json>
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import tree_sitter_python as tspython
from tree_sitter import Language, Parser

# ---------------------------------------------------------------- parsing

PY_LANGUAGE = Language(tspython.language())
_parser = Parser(PY_LANGUAGE)

TARGETS = {"function_definition", "class_definition"}
NEWLINE = b"\n"


def _body(node):
    return node.child_by_field_name("body")


def _iter_definitions(node):
    """Yield (definition, outer) depth first.

    `outer` is the decorated_definition when there is one, so its byte range
    covers the decorators. `@app.get("/prices/latest")` is the most searchable
    line in a route handler and the naive traversal drops it.
    """
    for child in node.children:
        if child.type == "decorated_definition":
            inner = child.child_by_field_name("definition")
            if inner is not None and inner.type in TARGETS:
                yield inner, child
                body = _body(inner)
                if body is not None:
                    yield from _iter_definitions(body)
                continue
        if child.type in TARGETS:
            yield child, child
            body = _body(child)
            if body is not None:
                yield from _iter_definitions(body)
            continue
        yield from _iter_definitions(child)


def _docstring(node, source: bytes):
    body = _body(node)
    if body is None or not body.children:
        return None
    first = body.children[0]
    if first.type != "expression_statement" or not first.children:
        return None
    literal = first.children[0]
    if literal.type != "string":
        return None
    return source[literal.start_byte:literal.end_byte].decode(errors="replace")


def _method_names(node) -> list[str]:
    body = _body(node)
    if body is None:
        return []
    names = []
    for child in body.children:
        target = child
        if child.type == "decorated_definition":
            target = child.child_by_field_name("definition")
        if target is not None and target.type == "function_definition":
            name_node = target.child_by_field_name("name")
            if name_node is not None:
                names.append(name_node.text.decode())
    return names


def _field_names(node) -> list[str]:
    """Names assigned or annotated directly in a class body.

    `symbol = Column(String)` and `symbol: str` both count; anything inside a
    method does not, because this walks only the class body's direct children.
    Dunders are skipped -- `__tablename__` is framework plumbing, and nobody
    asks a question whose answer is its presence.
    """
    body = _body(node)
    if body is None:
        return []
    names = []
    for child in body.children:
        stmt = child.children[0] if child.type == "expression_statement" else child
        if stmt is None or stmt.type not in ("assignment", "augmented_assignment"):
            continue
        target = stmt.child_by_field_name("left")
        if target is not None and target.type == "identifier":
            name = target.text.decode()
            if not (name.startswith("__") and name.endswith("__")):
                names.append(name)
    return names


def _class_stub(node, source: bytes) -> str:
    """Signature, docstring, and the *names* of what the class contains.

    Methods are chunked individually, so emitting the full class body as well
    would index the same source twice and let one big class dominate retrieval.
    Fields are listed the same way methods are, for a reason measured rather
    than assumed.

    The first attempt at fixing model classes emitted their bodies in full, on
    the theory that an ORM model is nothing but its fields and stubbing deleted
    its only content. Measured, that was wrong in an instructive way: overall
    MRR did not move (0.54 either way), R@1 rose and R@5 fell, and one question
    that had been answered -- "what gets recorded when something unusual
    happens" -> models.Anomaly -- stopped being answered at all. An embedding is
    a mean over tokens, so a two-line stub carrying a docstring is nearly pure
    signal, and ten lines of `Column(Float, nullable=False)` dilute it toward
    generic ORM boilerplate. More text made the chunk *less* like the question.

    Listing field names is the version that gives BM25 the tokens it needs
    without burying the docstring the embedding depends on.
    """
    body = _body(node)
    end = body.start_byte if body is not None else node.end_byte
    lines = [source[node.start_byte:end].decode(errors="replace").strip()]
    doc = _docstring(node, source)
    if doc:
        lines.append("    " + doc)
    fields = _field_names(node)
    if fields:
        lines.append("    # fields: " + ", ".join(fields))
    methods = _method_names(node)
    if methods:
        lines.append("    # methods: " + ", ".join(methods))
    return "\n".join(lines)


def _module_preamble(tree, source: bytes) -> tuple[int, int] | None:
    """Byte range of the top-level code before the first def/class.

    Constants and the comments explaining them live here -- MIN_PCT_MOVE, its
    three-line justification, FETCH_INTERVAL_SECONDS, the API URL. A chunker
    that only emits functions and classes cannot retrieve any of it, so the
    question "why don't stablecoins trigger alerts" is unanswerable even though
    the answer is written in the repo. Only emitted when there is at least one
    top-level assignment, so files that open with bare imports add no noise.
    """
    first_def = None
    has_assignment = False
    for child in tree.root_node.children:
        if child.type in TARGETS or child.type == "decorated_definition":
            first_def = child.start_byte
            break
        if child.type == "expression_statement" and child.children:
            if child.children[0].type in ("assignment", "augmented_assignment"):
                has_assignment = True
    end = first_def if first_def is not None else len(source)
    return (0, end) if has_assignment and end > 0 else None


def extract_chunks(path: Path, root: Path | None = None) -> list[dict]:
    source = path.read_bytes()
    tree = _parser.parse(source)
    rel = str(path.relative_to(root)) if root else str(path)

    chunks = []
    preamble = _module_preamble(tree, source)
    if preamble:
        start, end = preamble
        text = source[start:end].decode(errors="replace").strip()
        if text:
            last_line = source[:end].count(NEWLINE) + 1
            chunks.append({
                "chunk_id": rel + ":1-" + str(last_line) + ":<module>",
                "path": rel,
                "kind": "module",
                "name": "<module>",
                "start_line": 1,
                "end_line": last_line,
                "source": text,
            })
    for node, outer in _iter_definitions(tree.root_node):
        name_node = node.child_by_field_name("name")
        name = name_node.text.decode() if name_node else "<anonymous>"
        kind = node.type.removesuffix("_definition")
        text = (_class_stub(node, source) if kind == "class"
                else source[outer.start_byte:outer.end_byte].decode(errors="replace"))

        start_line = outer.start_point[0] + 1
        end_line = outer.end_point[0] + 1
        chunks.append({
            "chunk_id": f"{rel}:{start_line}-{end_line}:{name}",
            "path": rel,
            "kind": kind,
            "name": name,
            "start_line": start_line,
            "end_line": end_line,
            "source": text,
        })
    return chunks


# ------------------------------------------------------------- repo walk

SKIP_DIRS = {
    ".git", ".venv", "venv", "env", "__pycache__", "node_modules",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
    "build", "dist", ".eggs",
}


def chunk_repo(root: Path) -> list[dict]:
    """Chunk every .py file under root, skipping vendored and generated trees.

    The skip check is relative to root, not the absolute path -- otherwise a
    repo living under a folder called "build" or "env" has every file skipped.
    """
    root = root.resolve()
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    chunks, skipped = [], 0
    for path in sorted(root.rglob("*.py")):
        parts = path.relative_to(root).parts
        if any(p in SKIP_DIRS or p.endswith(".egg-info") for p in parts):
            continue
        try:
            chunks.extend(extract_chunks(path, root=root))
        except (UnicodeDecodeError, OSError):
            skipped += 1
    if skipped:
        print(f"skipped {skipped} unreadable file(s)")
    return chunks


# ------------------------------------------------------------------ bm25

K1, B = 1.5, 0.75

# Multiplier applied when a query token is exactly a chunk's own name, so a
# definition outranks the tests that merely call it. 1.0 disables it. Set via
# the environment so it can be swept against the eval harness rather than
# guessed: NAME_BOOST=3 python arch.py eval <repo> questions.json
NAME_BOOST = float(os.environ.get("NAME_BOOST", "3.0"))

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_PARTS = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+|\d+")


def tokenize(text: str) -> list[str]:
    """Emit each identifier whole AND split on snake_case / camelCase.

    `validate_token` has to match a query of "validate token". A stock English
    tokenizer does neither and loses most of the signal in code.
    """
    tokens = []
    for ident in _IDENT.findall(text):
        tokens.append(ident.lower())
        parts = [p.lower() for p in _PARTS.findall(ident) if p]
        if len(parts) > 1:
            tokens.extend(parts)
    return tokens


def chunk_text(chunk: dict) -> str:
    return f"{chunk['path']} {chunk['name']} {chunk['source']}"


class BM25:
    def __init__(self, chunks: list[dict], k1: float = K1, b: float = B):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.freqs = [Counter(tokenize(chunk_text(c))) for c in chunks]
        self.lengths = [sum(f.values()) for f in self.freqs]
        self.avgdl = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0

        df = Counter()
        for f in self.freqs:
            df.update(f.keys())
        n = len(self.freqs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def score(self, query_tokens: list[str], i: int) -> float:
        freq, length, total = self.freqs[i], self.lengths[i], 0.0
        for term in query_tokens:
            tf = freq.get(term, 0)
            if not tf:
                continue
            norm = tf + self.k1 * (1 - self.b + self.b * length / (self.avgdl or 1))
            total += self.idf.get(term, 0.0) * tf * (self.k1 + 1) / norm
        if total and self.chunks[i]["name"].lower() in query_tokens:
            total *= NAME_BOOST
        return total

    def search(self, query: str, k: int = 5) -> list[tuple[float, dict]]:
        tokens = tokenize(query)
        hits = [(self.score(tokens, i), c) for i, c in enumerate(self.chunks)]
        hits = [h for h in hits if h[0] > 0]
        hits.sort(key=lambda pair: -pair[0])
        return hits[:k]

    def explain(self, query: str, i: int) -> dict:
        """Itemise score() for one chunk: what each query term contributed.

        Deliberately a second implementation rather than a refactor of
        score(). score() runs once per chunk per query -- building a dict of
        per-term contributions in that loop costs more than the retrieval it
        is explaining, and the measured 2-3ms is a claim this project makes.
        The duplication is held honest by a test asserting the two agree, which
        is cheaper than the alternative and fails loudly if either drifts.
        """
        tokens = tokenize(query)
        freq, length = self.freqs[i], self.lengths[i]
        norm_len = 1 - self.b + self.b * length / (self.avgdl or 1)

        agg: dict[str, dict] = {}
        for term in tokens:
            tf = freq.get(term, 0)
            if not tf:
                continue
            idf = self.idf.get(term, 0.0)
            e = agg.setdefault(term, {"term": term, "tf": tf, "idf": idf,
                                      "contribution": 0.0})
            e["contribution"] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm_len)

        terms = sorted(agg.values(), key=lambda e: -e["contribution"])
        boosted = bool(terms) and self.chunks[i]["name"].lower() in tokens
        return {
            "terms": [{"term": e["term"], "tf": e["tf"],
                       "idf": round(e["idf"], 3),
                       "contribution": round(e["contribution"], 4)} for e in terms],
            "name_boost": NAME_BOOST if boosted else None,
        }


# ------------------------------------------------------------- embeddings

# Swappable so the choice of embedding model can be measured rather than argued
# about:  ARCH_MODEL=<name> python arch.py eval <repo> questions.json
# The cache key below includes this name, so switching models writes a separate
# cache rather than silently reusing the previous model's vectors.
MODEL_NAME = os.environ.get("ARCH_MODEL", "all-MiniLM-L6-v2")
CACHE_DIR = Path(".cache")

_model = None


def _load_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        kwargs = {}
        if os.environ.get("ARCH_MODEL_TRUST") == "1":
            # Some code-trained models (jina-embeddings-v2-base-code among them)
            # ship their own architecture and will not load without this. It
            # executes code downloaded from the Hub, so it is opt-in per run
            # rather than a default, and the default model does not need it.
            kwargs["trust_remote_code"] = True
        _model = SentenceTransformer(MODEL_NAME, **kwargs)
    return _model


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def _fingerprint(chunks: list[dict]) -> str:
    h = hashlib.sha256()
    h.update(MODEL_NAME.encode())
    for c in chunks:
        h.update(c["chunk_id"].encode())
        h.update(str(len(c["source"])).encode())
    return h.hexdigest()[:16]


class VectorIndex:
    """Brute-force cosine. No vector DB until a measurement says otherwise."""

    def __init__(self, chunks: list[dict], cache_dir: Path = CACHE_DIR):
        self.chunks = chunks
        cache_dir.mkdir(exist_ok=True)
        cache = cache_dir / f"emb-{_fingerprint(chunks)}.npy"

        if cache.exists():
            self.matrix, self.cached = np.load(cache), True
        else:
            vectors = _load_model().encode(
                [chunk_text(c) for c in chunks],
                batch_size=32, show_progress_bar=True, convert_to_numpy=True,
            )
            self.matrix, self.cached = _normalise(vectors), False
            np.save(cache, self.matrix)

    def search(self, query: str, k: int = 5) -> list[tuple[float, dict]]:
        q = _normalise(_load_model().encode([query], convert_to_numpy=True))[0]
        scores = self.matrix @ q
        return [(float(scores[i]), self.chunks[i]) for i in np.argsort(-scores)[:k]]


# ------------------------------------------------------------------ fusion

RRF_K = 60
FUSE_DEPTH = 20


def rrf(rankings: list[list[tuple[float, dict]]], k: int = RRF_K,
        top: int = 5) -> list[tuple[float, dict]]:
    """Reciprocal Rank Fusion.

    Each list contributes 1/(k + rank) per chunk; the scores are summed.
    Only ranks are used, never the raw scores -- BM25 returns unbounded
    positive numbers and cosine returns -1..1, so there is no sane way to add
    them directly. Normalising them would need a calibration set we don't have.
    Rank position is the one thing both retrievers agree on the meaning of.

    k=60 is the constant from Cormack et al. (2009). It damps the top ranks:
    without it, rank 1 would dominate so heavily that a second opinion from
    the other retriever could never promote anything.

    Fuse deeper than you return -- rankings are taken to FUSE_DEPTH so a chunk
    ranked 12th by both can beat one ranked 2nd by one and absent from the other.
    """
    scores: dict[str, float] = {}
    chunks: dict[str, dict] = {}
    for ranked in rankings:
        for rank, (_, chunk) in enumerate(ranked, 1):
            cid = chunk["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            chunks[cid] = chunk
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])
    return [(score, chunks[cid]) for cid, score in ordered[:top]]


# ------------------------------------------------------------------ router

def symbol_names(chunks: list[dict]) -> set[str]:
    return {c["name"].lower() for c in chunks}


_CODE_SHAPED = re.compile(r"^[A-Za-z]+(_[A-Za-z0-9]+)+$|^[a-z]+[A-Z]|^[A-Z][a-z]+[A-Z]")


def matched_symbol(query: str, names: set[str]) -> str | None:
    """The code-shaped word in the query that names a real symbol, if any."""
    for word in re.split(r"[\s(),:;\[\]{}'\"`]+", query):
        word = word.strip(".")
        if word and _CODE_SHAPED.search(word) and word.lower() in names:
            return word
    return None


def is_identifier_query(query: str, names: set[str]) -> bool:
    """True when the query *types out* a symbol that exists in the repo.

    Two conditions, and the second one was learned the hard way. The token has
    to name a real symbol, AND it has to be written like code -- snake_case or
    camelCase or CapWords.

    Matching on "names a real symbol" alone scored 40/40 on a repo whose
    functions are called `store_prices` and `check_asset`. Pointed at FastAPI,
    it collapsed: that framework has methods named `get`, `post`, `put`,
    `head`, `options` and `trace`, so "what happens when a request fails" was
    classified as an identifier query and sent to BM25. The heuristic had
    silently assumed symbol names are not ordinary English words.

    Requiring code shape costs nothing on the original corpus -- every
    identifier question there is written `fetch_prices` or `PriceReading` --
    and removes the whole class of collision.
    """
    for word in re.split(r"[\s(),:;\[\]{}'\"`]+", query):
        word = word.strip(".")
        if not word or not _CODE_SHAPED.search(word):
            continue
        if word.lower() in names:
            return True
    return False


def route(query: str, bm: "BM25", vec: "VectorIndex", names: set[str],
          k: int = 5) -> list[tuple[float, dict]]:
    """Pick one retriever per query instead of fusing both.

    Measured on 40 labelled questions: BM25 scores 10/10 on identifier queries
    and 0.11 MRR on paraphrase; the vector index is the reverse but milder.
    Unweighted RRF loses to the better of the two on BOTH kinds, because an
    equal vote lets the weaker retriever evict the stronger one's answers.
    """
    return (bm if is_identifier_query(query, names) else vec).search(query, k)


# ------------------------------------------------------------------ answer

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
              "{model}:generateContent")

PROMPT = """You are answering a question about a specific codebase. Below are \
the only code excerpts you may use. Each is headed by its file and line range.

Rules:
- Answer only from these excerpts. Do not use general knowledge about how such
  systems are usually written.
- Cite the file and line range for every claim, like (app/store.py:24-52).
- If the excerpts do not contain the answer, say exactly: "Not in the retrieved
  code." and then name which files you would look in next. Do not guess.
- Be brief. Two or three sentences unless the question needs more.

QUESTION: {question}

EXCERPTS:
{excerpts}
"""


def build_prompt(question: str, hits: list[tuple[float, dict]]) -> str:
    excerpts = []
    for _, c in hits:
        header = f"--- {c['path']}:{c['start_line']}-{c['end_line']} ---"
        excerpts.append(header + "\n" + c["source"])
    return PROMPT.format(question=question, excerpts="\n\n".join(excerpts))


def ask_gemini(prompt: str) -> str:
    """Call Gemini over plain REST.

    httpx is already a dependency and the REST contract is stable, so this
    avoids taking on an SDK whose import path has changed twice.
    """
    import httpx

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("set GEMINI_API_KEY first")

    resp = httpx.post(
        GEMINI_URL.format(model=GEMINI_MODEL),
        params={"key": key},
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=30.0,
    )
    if resp.status_code != 200:
        raise SystemExit(f"gemini {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        return f"unexpected response shape: {data}"


# -------------------------------------------------------------------- eval

def load_questions(path: Path) -> list[dict]:
    """Each entry: {"q": "...", "file": "app/x.py", "name": "func_name"}."""
    return json.loads(path.read_text())


def _rank_of(hits: list[tuple[float, dict]], want: dict) -> int | None:
    """1-based rank of the expected chunk, or None if absent.

    Matched on (path, name) rather than chunk_id -- line numbers shift every
    time the target repo is edited, and a question set that rots on every
    commit is a question set nobody re-runs.
    """
    targets = {(want["file"], want["name"])}
    targets |= {(a["file"], a["name"]) for a in want.get("alt", [])}
    for rank, (_, c) in enumerate(hits, 1):
        if (c["path"], c["name"]) in targets:
            return rank
    return None


def _metrics(ranks: list[int | None], at: int = 5) -> dict:
    n = len(ranks)
    if not n:
        return {"r@1": 0.0, f"r@{at}": 0.0, "mrr": 0.0}
    return {
        "r@1": sum(1 for r in ranks if r == 1) / n,
        f"r@{at}": sum(1 for r in ranks if r is not None and r <= at) / n,
        "mrr": sum((1.0 / r) for r in ranks if r is not None) / n,
    }


RETRIEVERS = ("BM25", "Vector", "Hybrid", "Routed")


def run_eval(chunks: list[dict], questions: list[dict], at: int = 5,
             bm: "BM25 | None" = None, vec: "VectorIndex | None" = None) -> dict:
    """Score every retriever on every labelled question and return the lot.

    One implementation behind both the CLI table and the /eval endpoint. The
    per-question ranks are kept, not just the aggregates, because the aggregate
    is the claim and the per-question rows are the evidence for it -- and a
    reader who cannot see which questions were missed has to take 0.54 on
    faith.
    """
    index = {(c["path"], c["name"]) for c in chunks}

    def known(q):
        pairs = [(q["file"], q["name"])]
        pairs += [(a["file"], a["name"]) for a in q.get("alt", [])]
        return all(pair in index for pair in pairs)

    missing = [q for q in questions if not known(q)]
    live = [q for q in questions if known(q)]
    if not live:
        raise SystemExit("no usable questions")

    bm = bm or BM25(chunks)
    vec = vec or VectorIndex(chunks)
    names = symbol_names(chunks)

    rows = []
    for q in live:
        keyword = bm.search(q["q"], k=FUSE_DEPTH)
        dense = vec.search(q["q"], k=FUSE_DEPTH)
        fused = rrf([keyword, dense], top=at)
        ranks = {
            "BM25": _rank_of(keyword[:at], q),
            "Vector": _rank_of(dense[:at], q),
            "Hybrid": _rank_of(fused, q),
            "Routed": _rank_of(route(q["q"], bm, vec, names, at), q),
        }
        rows.append({
            "q": q["q"],
            "kind": q.get("kind", "untagged"),
            "expected": f"{q['file']}:{q['name']}",
            "ranks": ranks,
            "answered": any(r is not None for r in ranks.values()),
        })

    def group(keep):
        picked = [r for r in rows if keep(r)]
        if not picked:
            return None
        return {
            "n": len(picked),
            "metrics": {name: _metrics([r["ranks"][name] for r in picked], at)
                        for name in RETRIEVERS},
        }

    kinds = sorted({r["kind"] for r in rows})
    if len(kinds) < 2:
        kinds = []
    return {
        "at": at,
        "name_boost": NAME_BOOST,
        "rows": rows,
        "groups": {"ALL": group(lambda r: True),
                   **{k.upper(): group(lambda r, k=k: r["kind"] == k) for k in kinds}},
        "missing": [{"q": q["q"], "expected": f"{q['file']}:{q['name']}"}
                    for q in missing],
    }


def evaluate(chunks: list[dict], questions: list[dict], at: int = 5) -> None:
    """Score BM25, vector and hybrid on the same labelled questions."""
    report = run_eval(chunks, questions, at)

    if report["missing"]:
        print(f"{len(report['missing'])} question(s) point at a chunk that does "
              f"not exist -- fix these before trusting the numbers:")
        for m in report["missing"]:
            print(f"  {m['expected']}   <- {m['q']!r}")
        print()

    print(f"{len(report['rows'])} questions, top-{at}, "
          f"NAME_BOOST={report['name_boost']}, model={MODEL_NAME}\n")
    for title, g in report["groups"].items():
        if not g:
            continue
        print(f"{title}  (n={g['n']})")
        print(f"{'':10}{'R@1':>8}{f'R@{at}':>8}{'MRR':>8}")
        for name in RETRIEVERS:
            m = g["metrics"][name]
            print(f"{name:10}{m['r@1']:8.2f}{m[f'r@{at}']:8.2f}{m['mrr']:8.2f}")
        print()

    unanswered = [r for r in report["rows"] if not r["answered"]]
    if unanswered:
        print(f"\n{len(unanswered)} question(s) no retriever answered in top {at}:")
        for r in unanswered:
            print(f"  {r['q']!r}\n      -> {r['expected']}")


# --------------------------------------------------------------------- api

def create_app(repo: Path, questions: list[dict] | None = None):
    """FastAPI wrapper. Index is built once at startup, not per request.

    Chunking and embedding a repo takes seconds; doing it per request would
    make the API useless. The index is immutable for the process lifetime --
    re-index by restarting, which is honest for a tool pointed at a checkout.

    `questions` is optional. When a labelled set is supplied, /eval exposes the
    full evaluation over the repo being served.
    """
    from fastapi import FastAPI, HTTPException, Query
    from pydantic import BaseModel, Field

    class Health(BaseModel):
        status: str = Field(examples=["ok"])
        repo: str
        repo_name: str = Field(
            examples=["fastapi"],
            description="Just the directory name -- what a UI should display, "
                        "so a screenshot does not publish someone's home path.")
        has_eval: bool = Field(
            description="Whether a labelled question set was supplied, and so "
                        "whether /eval will answer.")
        chunks: int = Field(examples=[45])
        sample_symbols: list[str] = Field(
            description="Real symbol names from this index, so a UI can offer "
                        "identifier-query examples that actually resolve.")

    class Hit(BaseModel):
        chunk_id: str = Field(
            examples=["app/anomaly.py:34-85"],
            description="Stable across retrievers, so a client can tell when two "
                        "of them returned the same chunk.")
        path: str = Field(examples=["app/anomaly.py"])
        start_line: int = Field(examples=[34])
        end_line: int = Field(examples=[85])
        name: str = Field(examples=["check_asset"])
        kind: str = Field(examples=["function"], description="function, class or module")
        score: float = Field(examples=[11.56])
        source: str
        terms: list[dict] | None = Field(
            default=None,
            description="For BM25 hits only: what each query term contributed "
                        "to the score, and whether NAME_BOOST fired. Cosine "
                        "similarity has no equivalent decomposition, so vector "
                        "hits carry null -- which is itself worth seeing.")
        name_boost: float | None = Field(
            default=None,
            description="The multiplier applied because a query token exactly "
                        "matched this chunk's name, or null if it did not fire.")

    class Ranking(BaseModel):
        retriever: str = Field(examples=["bm25"])
        took_ms: float = Field(examples=[2.3])
        results: list[Hit]

    class CompareResponse(BaseModel):
        query: str
        query_kind: str
        routed_to: str = Field(
            description="The retriever the router picked -- the one the project "
                        "argues you should use for this query.")
        rankings: list[Ranking] = Field(
            description="bm25, vector and their unweighted RRF fusion, over the "
                        "same query, so the three can be read against each other.")

    class ChunkRef(BaseModel):
        chunk_id: str
        path: str
        name: str
        start_line: int
        end_line: int
        kind: str

    class SearchResponse(BaseModel):
        query: str
        query_kind: str = Field(description="identifier or paraphrase")
        retriever: str = Field(description="which retriever the router chose")
        took_ms: float = Field(examples=[3.4], description="retrieval only")
        results: list[Hit]

    class Citation(BaseModel):
        path: str = Field(examples=["app/anomaly.py"])
        start_line: int = Field(examples=[34])
        end_line: int = Field(examples=[85])
        name: str = Field(examples=["check_asset"])

    class AskResponse(BaseModel):
        query: str
        query_kind: str
        answer: str = Field(
            description='Grounded in the cited chunks, or "Not in the retrieved '
                        'code." when they do not contain the answer.')
        citations: list[Citation]

    chunks = chunk_repo(repo)
    bm = BM25(chunks)
    vec = VectorIndex(chunks)
    names = symbol_names(chunks)

    app = FastAPI(
        title="Codebase Archaeologist",
        version="0.1.0",
        description=(
            "Ask a Python repository questions in English and get answers "
            "cited to file and line.\n\n"
            "Queries are **routed**, not fused: a query naming a real symbol "
            "goes to BM25, anything else to the embedding index. Measured on "
            "41 labelled questions, routing scores MRR 0.54 against 0.36 for "
            "Reciprocal Rank Fusion and 0.49 for the best single retriever."
        ),
    )

    index_of = {c["chunk_id"]: i for i, c in enumerate(chunks)}

    def _serialise(hits, query=None):
        """`query` is passed only for BM25 rankings, which can be itemised."""
        out = []
        for score, c in hits:
            row = {
                "chunk_id": c["chunk_id"],
                "path": c["path"],
                "start_line": c["start_line"],
                "end_line": c["end_line"],
                "name": c["name"],
                "kind": c["kind"],
                "score": round(score, 4),
                "source": c["source"],
                "terms": None,
            }
            if query is not None:
                e = bm.explain(query, index_of[c["chunk_id"]])
                row["terms"] = e["terms"]
                row["name_boost"] = e["name_boost"]
            out.append(row)
        return out

    @app.get("/", include_in_schema=False)
    def index():
        from fastapi.responses import FileResponse, PlainTextResponse

        page = Path(__file__).resolve().parent / "ui.html"
        if not page.exists():
            return PlainTextResponse("ui.html not found; API docs at /docs", 404)
        return FileResponse(page)

    class Classification(BaseModel):
        query: str
        query_kind: str = Field(description="identifier or paraphrase")
        retriever: str
        matched: str | None = Field(
            description="the code-shaped token that named a real symbol")

    @app.get("/classify", response_model=Classification,
             summary="Which retriever would handle this query (no retrieval)")
    def classify(q: str = Query("", max_length=500)):
        hit = matched_symbol(q, names)
        return {
            "query": q,
            "query_kind": "identifier" if hit else "paraphrase",
            "retriever": "bm25" if hit else "vector",
            "matched": hit,
        }

    @app.get("/health", response_model=Health,
             summary="Index status and chunk count")
    def health():
        interesting = sorted(
            (c for c in chunks
             if c["kind"] == "function" and not c["name"].startswith("_")
             and len(c["name"]) > 6),
            key=lambda c: -len(c["source"]),
        )
        seen, samples = set(), []
        for c in interesting:
            if c["name"] not in seen:
                seen.add(c["name"])
                samples.append(c["name"])
            if len(samples) == 3:
                break
        return {
            "status": "ok",
            "repo": str(repo),
            "repo_name": repo.resolve().name,
            "has_eval": questions is not None,
            "chunks": len(chunks),
            "sample_symbols": samples,
        }

    @app.get("/search", response_model=SearchResponse,
             summary="Retrieve chunks, routed by query type")
    def search(q: str = Query(..., min_length=1), k: int = Query(5, ge=1, le=20)):
        kind = "identifier" if is_identifier_query(q, names) else "paraphrase"
        t0 = time.perf_counter()
        hits = route(q, bm, vec, names, k)
        took = (time.perf_counter() - t0) * 1000
        return {
            "query": q,
            "query_kind": kind,
            "retriever": "bm25" if kind == "identifier" else "vector",
            "took_ms": round(took, 2),
            "results": _serialise(hits, q if kind == "identifier" else None),
        }

    @app.get("/compare", response_model=CompareResponse,
             summary="Run both retrievers and their fusion on one query")
    def compare(q: str = Query(..., min_length=1), k: int = Query(5, ge=1, le=20)):
        """The evaluation, one query at a time.

        Returns all three rankings rather than the routed one, so a client can
        show what the aggregate numbers in the README mean on a single query:
        where each retriever puts the right chunk, and what unweighted RRF does
        to that when it gives both an equal vote.
        """
        t0 = time.perf_counter()
        keyword = bm.search(q, k=FUSE_DEPTH)
        t_bm = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        dense = vec.search(q, k=FUSE_DEPTH)
        t_vec = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        fused = rrf([keyword, dense], top=k)
        t_rrf = (time.perf_counter() - t0) * 1000

        kind = "identifier" if is_identifier_query(q, names) else "paraphrase"
        return {
            "query": q,
            "query_kind": kind,
            "routed_to": "bm25" if kind == "identifier" else "vector",
            "rankings": [
                {"retriever": "bm25", "took_ms": round(t_bm, 2),
                 "results": _serialise(keyword[:k], q)},
                {"retriever": "vector", "took_ms": round(t_vec, 2),
                 "results": _serialise(dense[:k])},
                {"retriever": "rrf", "took_ms": round(t_rrf, 2),
                 "results": _serialise(fused)},
            ],
        }

    _eval_cache: dict = {}

    @app.get("/eval", summary="The full evaluation over the repo being served")
    def eval_endpoint():
        """Aggregates plus the per-question ranks behind them.

        Computed once and held, because the index is immutable for the process
        lifetime, so the answer cannot change between requests. Returns 404
        rather than an empty table when no labelled questions were supplied --
        a question set belongs to a specific repo, and silently evaluating one
        repo's questions against another's code would produce numbers that look
        real and mean nothing.
        """
        if questions is None:
            raise HTTPException(
                404, "No question set loaded. Start with: "
                     "python arch.py serve <repo> <questions.json>")
        if "report" not in _eval_cache:
            _eval_cache["report"] = run_eval(chunks, questions, bm=bm, vec=vec)
        return _eval_cache["report"]

    @app.get("/chunks", response_model=list[ChunkRef],
             summary="Every chunk in the index, without its source")
    def chunk_list():
        return [
            {"chunk_id": c["chunk_id"], "path": c["path"], "name": c["name"],
             "start_line": c["start_line"], "end_line": c["end_line"],
             "kind": c["kind"]}
            for c in chunks
        ]

    @app.get("/ask", response_model=AskResponse,
             summary="Answer a question with citations")
    def ask(q: str = Query(..., min_length=1), k: int = Query(5, ge=1, le=10)):
        if not os.environ.get("GEMINI_API_KEY"):
            raise HTTPException(503, "GEMINI_API_KEY is not set on the server")
        hits = route(q, bm, vec, names, k)
        kind = "identifier" if is_identifier_query(q, names) else "paraphrase"
        return {
            "query": q,
            "query_kind": kind,
            "answer": ask_gemini(build_prompt(q, hits)),
            "citations": [
                {"path": c["path"], "start_line": c["start_line"],
                 "end_line": c["end_line"], "name": c["name"]}
                for _, c in hits
            ],
        }

    return app


# -------------------------------------------------------------------- cli

def _show(hits, fmt="{:6.2f}"):
    if not hits:
        print("no matches")
    for rank, (score, c) in enumerate(hits, 1):
        print(f"{rank}. " + fmt.format(score) + f"  {c['chunk_id']}")
        print(f"            {c['source'].splitlines()[0].strip()}")


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        print(__doc__)
        raise SystemExit(1)

    command, root = argv[0], Path(argv[1]).expanduser()
    query = " ".join(argv[2:])
    chunks = chunk_repo(root)

    if command == "chunks":
        kinds = Counter(c["kind"] for c in chunks)
        files = len({c["path"] for c in chunks})
        lengths = sorted(len(c["source"]) for c in chunks)
        print(f"{len(chunks)} chunks from {files} files in {root}")
        print(f"  {kinds['function']} functions, {kinds['class']} classes")
        if lengths:
            print(f"  chunk chars: min {lengths[0]}, "
                  f"median {lengths[len(lengths)//2]}, max {lengths[-1]}")
        decorated = [c for c in chunks if c["source"].lstrip().startswith("@")]
        print(f"  {len(decorated)} chunks start at a decorator")

    elif command == "bm25":
        print(f"indexed {len(chunks)} chunks\nquery: {query!r}\n")
        _show(BM25(chunks).search(query))

    elif command == "vec":
        t0 = time.perf_counter()
        index = VectorIndex(chunks)
        print(f"indexed {len(chunks)} chunks in {time.perf_counter()-t0:.2f}s "
              f"({'cached' if index.cached else 'fresh'})\nquery: {query!r}\n")
        _show(index.search(query), "{:5.3f}")

    elif command == "hybrid":
        keyword = BM25(chunks).search(query, k=FUSE_DEPTH)
        dense = VectorIndex(chunks).search(query, k=FUSE_DEPTH)
        print(f"indexed {len(chunks)} chunks\nquery: {query!r}\n")

        print("BM25 top 5:")
        _show(keyword[:5])
        print("\nvector top 5:")
        _show(dense[:5], "{:5.3f}")
        print("\nhybrid (RRF) top 5:")
        _show(rrf([keyword, dense]), "{:6.4f}")

        fused = {c["chunk_id"] for _, c in rrf([keyword, dense])}
        seen_alone = ({c["chunk_id"] for _, c in keyword[:5]}
                      | {c["chunk_id"] for _, c in dense[:5]})
        print(f"\n{len(fused - seen_alone)} of the 5 fused hits were in "
              f"neither retriever's top 5 on its own")

    elif command == "route":
        bm, vec = BM25(chunks), VectorIndex(chunks)
        names = symbol_names(chunks)
        kind = "identifier" if is_identifier_query(query, names) else "paraphrase"
        print(f"indexed {len(chunks)} chunks\nquery: {query!r}\n"
              f"routed to {'BM25' if kind == 'identifier' else 'vector'} "
              f"({kind} query)\n")
        _show(route(query, bm, vec, names))

    elif command == "ask":
        bm, vec = BM25(chunks), VectorIndex(chunks)
        names = symbol_names(chunks)
        kind = "identifier" if is_identifier_query(query, names) else "paraphrase"
        hits = route(query, bm, vec, names, 5)
        print(f"query: {query!r}  ({kind} -> "
              f"{'BM25' if kind == 'identifier' else 'vector'})\n")
        for _, c in hits:
            print(f"  retrieved {c['path']}:{c['start_line']}-{c['end_line']}")
        print()
        print(ask_gemini(build_prompt(query, hits)))

    elif command == "serve":
        import uvicorn

        port = int(os.environ.get("PORT", "8000"))
        qs = None
        if len(argv) > 2:
            qs = load_questions(Path(argv[2]).expanduser())
        print(f"indexed {len(chunks)} chunks from {root}")
        if qs:
            print(f"loaded {len(qs)} labelled questions -- /eval is live")
        print(f"http://127.0.0.1:{port}")
        uvicorn.run(create_app(root, qs), host="127.0.0.1", port=port)

    elif command == "eval":
        if not argv[2:]:
            raise SystemExit("usage: python arch.py eval <repo> <questions.json>")
        evaluate(chunks, load_questions(Path(argv[2]).expanduser()))

    else:
        print(__doc__)
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
