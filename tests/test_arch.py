"""Tests for the retrieval logic.

Deliberately no network and no embedding model: everything here is the part
that can be wrong in a way a human would not notice -- tokenisation, chunk
boundaries, fusion arithmetic, metric definitions, routing. The model is a
dependency, not our code; what we own is how its output is combined.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import arch  # noqa: E402


# ----------------------------------------------------------------- fixtures

SAMPLE = '''\
import logging

from app.models import Asset

logger = logging.getLogger("demo")

# Minimum move that counts as real
MIN_PCT_MOVE = 2.0


class PriceRepository:
    """Reads price rows."""

    def __init__(self, session):
        self.session = session

    def latest(self, symbol):
        return symbol


@app.get("/prices/latest")
def get_latest_prices(repo):
    """Latest price for each asset."""
    return repo.all()


def validate_token(token):
    return bool(token)
'''

IMPORTS_ONLY = "import os\nfrom pathlib import Path\n\n\ndef helper():\n    return 1\n"


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "sample.py").write_text(SAMPLE)
    (tmp_path / "app" / "plain.py").write_text(IMPORTS_ONLY)
    return tmp_path


@pytest.fixture
def chunks(repo):
    return arch.chunk_repo(repo)


def by_name(chunks, name):
    return next(c for c in chunks if c["name"] == name)


# --------------------------------------------------------------- tokenizer

@pytest.mark.parametrize("text,expected", [
    ("validate_token", {"validate_token", "validate", "token"}),
    ("fetchPrices", {"fetchprices", "fetch", "prices"}),
    ("HTTPResponse", {"httpresponse", "http", "response"}),
    ("plain", {"plain"}),
])
def test_tokenize_splits_identifiers(text, expected):
    assert set(arch.tokenize(text)) == expected


def test_tokenize_keeps_the_whole_identifier_as_well_as_its_parts():
    # A query of "store_prices" must still match, not only "store" and "prices".
    assert "store_prices" in arch.tokenize("store_prices")


# ---------------------------------------------------------------- chunking

def test_decorated_function_chunk_starts_at_the_decorator(chunks):
    c = by_name(chunks, "get_latest_prices")
    assert c["source"].startswith('@app.get("/prices/latest")')
    assert "/prices/latest" in c["source"]


def test_class_chunk_is_a_stub_not_the_whole_body(chunks):
    c = by_name(chunks, "PriceRepository")
    assert c["kind"] == "class"
    assert "# methods: __init__, latest" in c["source"]
    assert "self.session = session" not in c["source"]


def test_methods_are_chunked_individually(chunks):
    assert by_name(chunks, "latest")["kind"] == "function"


def test_module_preamble_is_chunked_when_there_are_constants(chunks):
    c = next(c for c in chunks if c["kind"] == "module")
    assert "MIN_PCT_MOVE = 2.0" in c["source"]
    assert "Minimum move that counts as real" in c["source"]


def test_no_module_chunk_for_a_file_that_only_imports(chunks):
    modules = [c for c in chunks if c["kind"] == "module"]
    assert all(c["path"] != "app/plain.py" for c in modules)


def test_chunk_ids_are_unique(chunks):
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_missing_repo_path_raises_rather_than_returning_nothing(tmp_path):
    with pytest.raises(SystemExit):
        arch.chunk_repo(tmp_path / "does-not-exist")


def test_skip_dirs_are_matched_relative_to_the_repo_root(tmp_path):
    # A repo that happens to live under a folder called "build" must not be
    # skipped wholesale -- the check is relative, not absolute.
    root = tmp_path / "build" / "myrepo"
    root.mkdir(parents=True)
    (root / "code.py").write_text("def f():\n    return 1\n")
    assert len(arch.chunk_repo(root)) == 1


# -------------------------------------------------------------------- bm25

def test_bm25_finds_an_exact_identifier(chunks):
    hits = arch.BM25(chunks).search("validate_token", k=5)
    assert hits and hits[0][1]["name"] == "validate_token"


def test_name_boost_multiplies_only_when_the_query_names_the_chunk(chunks):
    bm = arch.BM25(chunks)
    i = next(i for i, c in enumerate(chunks) if c["name"] == "validate_token")
    toks = arch.tokenize("validate_token")

    original = arch.NAME_BOOST
    try:
        arch.NAME_BOOST = 1.0
        plain = bm.score(toks, i)
        arch.NAME_BOOST = 4.0
        boosted = bm.score(toks, i)
    finally:
        arch.NAME_BOOST = original

    assert boosted == pytest.approx(plain * 4.0)


def test_name_boost_does_not_fire_on_an_unrelated_query(chunks):
    bm = arch.BM25(chunks)
    i = next(i for i, c in enumerate(chunks) if c["name"] == "validate_token")
    toks = arch.tokenize("latest price for each asset")

    original = arch.NAME_BOOST
    try:
        arch.NAME_BOOST = 1.0
        plain = bm.score(toks, i)
        arch.NAME_BOOST = 9.0
        boosted = bm.score(toks, i)
    finally:
        arch.NAME_BOOST = original

    assert boosted == plain


# ------------------------------------------------------------------ fusion

def _fake(names):
    return [(1.0, {"chunk_id": n, "path": "f.py", "name": n}) for n in names]


def test_rrf_score_is_the_sum_of_reciprocal_ranks():
    a = _fake(["x", "y", "z"])          # x at rank 1
    b = _fake(["y", "x"])               # x at rank 2
    fused = arch.rrf([a, b], top=3)
    k = arch.RRF_K
    top_name, top_score = fused[0][1]["name"], fused[0][0]
    assert top_name == "x"
    assert top_score == pytest.approx(1 / (k + 1) + 1 / (k + 2))


def test_rrf_is_monotone_decreasing():
    fused = arch.rrf([_fake(["a", "b", "c"]), _fake(["c", "a"])], top=3)
    scores = [s for s, _ in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_keeps_a_chunk_that_only_one_retriever_found():
    fused = arch.rrf([_fake(["a"]), _fake(["b"])], top=5)
    assert {c["name"] for _, c in fused} == {"a", "b"}


def test_rrf_over_a_single_ranking_preserves_its_order():
    ranking = _fake(["a", "b", "c"])
    assert [c["name"] for _, c in arch.rrf([ranking], top=3)] == ["a", "b", "c"]


# ------------------------------------------------------------------ router

def test_identifier_query_is_detected(chunks):
    names = arch.symbol_names(chunks)
    assert arch.is_identifier_query("where is validate_token", names)


def test_paraphrase_query_is_not_mistaken_for_an_identifier(chunks):
    names = arch.symbol_names(chunks)
    assert not arch.is_identifier_query("how do we check permissions", names)


# ------------------------------------------------------------------ metrics

def test_rank_of_finds_the_expected_chunk():
    hits = _fake(["a", "b", "c"])
    assert arch._rank_of(hits, {"file": "f.py", "name": "b"}) == 2


def test_rank_of_returns_none_when_absent():
    hits = _fake(["a", "b"])
    assert arch._rank_of(hits, {"file": "f.py", "name": "zzz"}) is None


def test_rank_of_accepts_an_alternate_answer():
    hits = _fake(["a", "b", "c"])
    want = {"file": "f.py", "name": "zzz",
            "alt": [{"file": "f.py", "name": "c"}]}
    assert arch._rank_of(hits, want) == 3


def test_metrics_are_computed_over_all_questions_including_misses():
    # ranks 1, 3 and a miss -> R@1 = 1/3, R@5 = 2/3, MRR = (1 + 1/3 + 0)/3
    m = arch._metrics([1, 3, None], at=5)
    assert m["r@1"] == pytest.approx(1 / 3)
    assert m["r@5"] == pytest.approx(2 / 3)
    assert m["mrr"] == pytest.approx((1 + 1 / 3) / 3)


def test_metrics_of_an_empty_run_are_zero_not_an_error():
    assert arch._metrics([], at=5)["mrr"] == 0.0


# ------------------------------------------------------------------ prompt

def test_prompt_carries_every_excerpt_with_its_file_and_lines(chunks):
    hits = [(1.0, by_name(chunks, "validate_token"))]
    prompt = arch.build_prompt("what does it do", hits)
    assert "app/sample.py:" in prompt
    assert "def validate_token" in prompt
    assert "what does it do" in prompt


def test_prompt_instructs_the_model_to_refuse_rather_than_guess():
    prompt = arch.build_prompt("q", [])
    assert "Not in the retrieved" in prompt