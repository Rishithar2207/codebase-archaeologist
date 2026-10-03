"""Codebase archaeologist -- ask a repo questions in English, get cited answers.

Single file for now; splits into modules once it earns the complexity.

    python arch.py chunks  <repo>
    python arch.py bm25    <repo> <query...>
    python arch.py vec     <repo> <query...>
    python arch.py hybrid  <repo> <query...>
    python arch.py eval    <repo> <questions.json>
"""
from __future__ import annotations

import hashlib
import json
import math
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


def _class_stub(node, source: bytes) -> str:
    """Signature + docstring + method names.

    Methods are chunked individually, so emitting the full class body as well
    would index the same source twice and let one big class dominate retrieval.
    """
    body = _body(node)
    end = body.start_byte if body is not None else node.end_byte
    lines = [source[node.start_byte:end].decode(errors="replace").strip()]
    doc = _docstring(node, source)
    if doc:
        lines.append("    " + doc)
    methods = _method_names(node)
    if methods:
        lines.append("    # methods: " + ", ".join(methods))
    return "\n".join(lines)


def extract_chunks(path: Path, root: Path | None = None) -> list[dict]:
    source = path.read_bytes()
    tree = _parser.parse(source)
    rel = str(path.relative_to(root)) if root else str(path)

    chunks = []
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
        return total

    def search(self, query: str, k: int = 5) -> list[tuple[float, dict]]:
        tokens = tokenize(query)
        hits = [(self.score(tokens, i), c) for i, c in enumerate(self.chunks)]
        hits = [h for h in hits if h[0] > 0]
        hits.sort(key=lambda pair: -pair[0])
        return hits[:k]


# ------------------------------------------------------------- embeddings

MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_DIR = Path(".cache")

_model = None


def _load_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME)
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
    for rank, (_, c) in enumerate(hits, 1):
        if c["path"] == want["file"] and c["name"] == want["name"]:
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


def evaluate(chunks: list[dict], questions: list[dict], at: int = 5) -> None:
    """Score BM25, vector and hybrid on the same labelled questions."""
    index = {(c["path"], c["name"]) for c in chunks}
    missing = [q for q in questions if (q["file"], q["name"]) not in index]
    live = [q for q in questions if (q["file"], q["name"]) in index]

    if missing:
        print(f"{len(missing)} question(s) point at a chunk that does not exist "
              f"-- fix these before trusting the numbers:")
        for q in missing:
            print(f"  {q['file']} : {q['name']}   <- {q['q']!r}")
        print()
    if not live:
        raise SystemExit("no usable questions")

    bm = BM25(chunks)
    vec = VectorIndex(chunks)

    runs: dict[str, list[int | None]] = {"BM25": [], "Vector": [], "Hybrid": []}
    unanswered = []
    for q in live:
        keyword = bm.search(q["q"], k=FUSE_DEPTH)
        dense = vec.search(q["q"], k=FUSE_DEPTH)
        fused = rrf([keyword, dense], top=at)

        ranks = {
            "BM25": _rank_of(keyword[:at], q),
            "Vector": _rank_of(dense[:at], q),
            "Hybrid": _rank_of(fused, q),
        }
        for name, r in ranks.items():
            runs[name].append(r)
        if not any(ranks.values()):
            unanswered.append(q)

    def table(title, keep):
        idx = [i for i, q in enumerate(live) if keep(q)]
        if not idx:
            return
        print(f"{title}  (n={len(idx)})")
        print(f"{'':10}{'R@1':>8}{f'R@{at}':>8}{'MRR':>8}")
        for name, ranks in runs.items():
            m = _metrics([ranks[i] for i in idx], at)
            print(f"{name:10}{m['r@1']:8.2f}{m[f'r@{at}']:8.2f}{m['mrr']:8.2f}")
        print()

    print(f"{len(live)} questions, top-{at}\n")
    table("ALL", lambda q: True)
    kinds = sorted({q.get("kind", "untagged") for q in live})
    if len(kinds) > 1:
        for kind in kinds:
            table(kind.upper(), lambda q, k=kind: q.get("kind", "untagged") == k)

    if unanswered:
        print(f"\n{len(unanswered)} question(s) no retriever answered in top {at}:")
        for q in unanswered:
            print(f"  {q['q']!r}\n      -> {q['file']} : {q['name']}")


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

    elif command == "eval":
        if not argv[2:]:
            raise SystemExit("usage: python arch.py eval <repo> <questions.json>")
        evaluate(chunks, load_questions(Path(argv[2]).expanduser()))

    else:
        print(__doc__)
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])