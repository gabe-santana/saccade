"""Multilingual text normalisation and FTS5 query construction.

Indexing side: text is NFKC-normalised and case-folded (so ``Straße`` matches
``strasse``), and CJK ideographs/kana are spaced apart so each becomes its own token —
FTS5's ``unicode61`` tokenizer cannot segment unspaced scripts otherwise. The tokenizer
also folds Latin diacritics (``configuração`` ≈ ``configuracao``) and treats combining
marks as part of words, which keeps Devanagari, Thai vowel signs etc. intact.

Query side: natural-language questions are turned into an OR of their content words.
Each word also matches by a lightly stemmed prefix (``deployment`` → ``deploy*``), and
multi-word queries add the exact phrase, which BM25 then rewards. CJK runs are matched
as overlapping character bigrams.
"""

from __future__ import annotations

import itertools
import re
import unicodedata
from dataclasses import dataclass

TOKENIZER = "unicode61 remove_diacritics 2 categories 'L* N* Co M*'"
TOKENIZER_FALLBACK = "unicode61 remove_diacritics 2"

_MIN_STEM = 4
_MAX_PHRASE_TOKENS = 8

_RAW_STOPWORDS = (
    # English
    "a an and are as at be been but by can could did do does for from had has have how i if in "
    "into is it its me my no not of on or our please should so than that the their them then there "
    "these they this those to tell us was we were what when where which who whom why will with "
    "would you your about any some just also let lets did said say says talk talked discuss "
    "discussed explain summarize summarise "
    # Portuguese
    "o os as um uma uns umas de do da dos das em no na nos nas por para pelo pela com sem que "
    "qual quais quando onde como porque porquê e ou mas se é foi são ser estar está estava eles "
    "elas ele ela nós você vocês isso isto aquilo sobre ao aos à às me nos lhe seu sua "
    # Spanish
    "el la los las un una unos unas del al en por para con sin que cual cuales cuando donde "
    "como porqué y o pero si es fue son ser estar está ellos ellas él ella nosotros usted "
    "ustedes eso esto sobre lo le su sus "
    # French
    "le la les un une des du de au aux en par pour avec sans que qui quoi quel quelle quand où "
    "comment pourquoi et ou mais si est était sont être il elle ils elles nous vous ce cette ces "
    "sur dans leur leurs son sa ses "
    # German
    "der die das den dem des ein eine einen einem einer und oder aber wenn ist war sind sein "
    "mit ohne für von zu zum zur im in am an auf aus bei was wer wie wo warum wann welche "
    "welcher es er sie wir ihr nicht über dass"
)

_SUFFIX_LIST = " ".join(
    (
        # English
        "ations ation ments ment ingly ings ing edly ed ers er ies es ly ity ities ions ion ures ure",
        # Portuguese / Spanish (diacritics already folded)
        "amentos amento imentos imento acoes acao coes cao ciones cion mente idades idade idad",
        "ando endo indo ados adas idos idas ado ada ido ida ar ir os as",
        # French
        "ements ement euses euse ees ee",
        # German
        "ungen ung heiten heit keiten keit lich isch en",
    )
)
_SUFFIXES = tuple(sorted(set(_SUFFIX_LIST.split()), key=len, reverse=True))


def is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x3040 <= code <= 0x30FF  # Hiragana, Katakana
        or 0x31F0 <= code <= 0x31FF
        or 0x3400 <= code <= 0x4DBF  # CJK Extension A
        or 0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
        or 0xF900 <= code <= 0xFAFF
        or 0xFF66 <= code <= 0xFF9F  # Half-width Katakana
        or 0x20000 <= code <= 0x2FA1F
    )


def _is_token_char(ch: str) -> bool:
    category = unicodedata.category(ch)
    return category[0] in "LNM" or category == "Co"


def normalize_text(text: str) -> str:
    """Normalised form stored in the full-text index."""
    text = unicodedata.normalize("NFKC", text).casefold()
    if not any(is_cjk(ch) for ch in text):
        return text
    return "".join(f" {ch} " if is_cjk(ch) else ch for ch in text)


def fold(token: str) -> str:
    """Remove Latin diacritics, mirroring the tokenizer's ``remove_diacritics`` option."""
    decomposed = unicodedata.normalize("NFD", token)
    out: list[str] = []
    for ch in decomposed:
        if unicodedata.category(ch) == "Mn" and out and ord(out[-1]) < 0x0250:
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


_STOPWORDS = frozenset(fold(w) for w in _RAW_STOPWORDS.split())


@dataclass(frozen=True, slots=True)
class Token:
    text: str
    cjk: bool


_ASCII_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[Token]:
    """Split text into word tokens the same way the FTS index does."""
    if text.isascii():
        return [Token(word, False) for word in _ASCII_WORD.findall(text.lower())]
    tokens: list[Token] = []
    current: list[str] = []
    for ch in unicodedata.normalize("NFKC", text).casefold():
        if is_cjk(ch):
            if current:
                tokens.append(Token(fold("".join(current)), False))
                current = []
            tokens.append(Token(ch, True))
        elif _is_token_char(ch):
            current.append(ch)
        elif current:
            tokens.append(Token(fold("".join(current)), False))
            current = []
    if current:
        tokens.append(Token(fold("".join(current)), False))
    return tokens


def light_stem(term: str) -> str | None:
    """A conservative, language-agnostic prefix for ``term`` (or ``None``)."""
    if len(term) < _MIN_STEM + 1 or not term.isalpha():
        return None
    for suffix in _SUFFIXES:
        if term.endswith(suffix) and len(term) - len(suffix) >= _MIN_STEM:
            return term[: -len(suffix)]
    if term.endswith("s") and not term.endswith("ss"):
        return term[:-1]
    return None


@dataclass(frozen=True, slots=True)
class QueryTerm:
    text: str
    stem: str | None = None
    cjk: bool = False

    def matches(self, tokens: set[str], joined_cjk: str) -> bool:
        if self.cjk:
            return self.text in joined_cjk
        if self.text in tokens:
            return True
        return self.stem is not None and any(t.startswith(self.stem) for t in tokens)


@dataclass(frozen=True, slots=True)
class MatchQuery:
    expression: str
    terms: tuple[QueryTerm, ...]
    phrase: str | None = None


def _quote(text: str) -> str:
    return '"' + text.replace('"', '""') + '"'


def build_match(query: str) -> MatchQuery | None:
    """Translate a free-text query into an FTS5 ``MATCH`` expression."""
    tokens = tokenize(query)
    if not tokens:
        return None

    terms: list[QueryTerm] = []
    seen: set[str] = set()

    def add(term: QueryTerm) -> None:
        if term.text not in seen:
            seen.add(term.text)
            terms.append(term)

    words = [t.text for t in tokens if not t.cjk]
    content = [w for w in words if w not in _STOPWORDS] or words
    for word in content:
        add(QueryTerm(word, light_stem(word)))

    run: list[str] = []
    for token in [*tokens, Token("", False)]:
        if token.cjk:
            run.append(token.text)
            continue
        if run:
            if len(run) <= 2:
                add(QueryTerm("".join(run), cjk=True))
            else:
                for a, b in itertools.pairwise(run):
                    add(QueryTerm(a + b, cjk=True))
            run = []

    if not terms:
        return None

    parts: list[str] = []
    for term in terms:
        if term.cjk:
            parts.append(_quote(" ".join(term.text)))
        elif term.stem:
            parts.append(f"{_quote(term.text)} OR {_quote(term.stem)} *")
        else:
            parts.append(_quote(term.text))

    phrase = None
    if 2 <= len(tokens) <= _MAX_PHRASE_TOKENS and len(terms) >= 2:
        phrase = " ".join(t.text for t in tokens)
        parts.append(_quote(phrase))

    return MatchQuery(expression=" OR ".join(parts), terms=tuple(terms), phrase=phrase)


def matched_terms(match: MatchQuery, text: str | list[Token]) -> tuple[str, ...]:
    tokens = tokenize(text) if isinstance(text, str) else text
    words = {t.text for t in tokens if not t.cjk}
    joined_cjk = "".join(t.text for t in tokens if t.cjk)
    return tuple(term.text for term in match.terms if term.matches(words, joined_cjk))


def contains_phrase(match: MatchQuery, text: str | list[Token]) -> bool:
    if match.phrase is None:
        return False
    tokens = tokenize(text) if isinstance(text, str) else text
    return f" {match.phrase} " in " " + " ".join(t.text for t in tokens) + " "
