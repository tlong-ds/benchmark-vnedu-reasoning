"""Shared text classifiers for the reasoning-chain validity gate.

Used by both ``chains.py`` (base-chain generation gate) and
``corruption.py`` (corrupted/placebo step-3 text) — general-purpose text
classification, not specific to either stage.
"""

import re
from enum import StrEnum


class ReasoningLanguage(StrEnum):
    """Observed language of a chain's text, not a claim about origin."""

    VI = "vi"
    EN = "en"
    MIXED_VI_EN = "mixed_vi_en"
    OTHER = "other"


_VI_DIACRITIC_RE = re.compile(
    r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩị"
    r"òóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]",
    re.IGNORECASE,
)
_VI_WORD_RE = re.compile(
    r"\b(?:bước|đáp|án|vì|vậy|nên|theo|từ|với|không|của|là|cho|"
    r"suy|luận|kết|quả|điều)\b",
    re.IGNORECASE,
)
_EN_WORD_RE = re.compile(
    r"\b(?:the|and|therefore|because|thus|answer|step|is|are|with|"
    r"from|this|that|reasoning|conclusion|we|calculate|plus|minus|"
    r"choose|option|fact|value|given|result|implies|select|compute|"
    r"integral|integrals|via|substitution|derive|multiply|divide|equation|"
    r"since|hence)\b",
    re.IGNORECASE,
)
_META_PLANNING_RE = re.compile(
    r"\b(?:i will (?:solve|start|present|proceed|break down|analyze|look at)|"
    r"my plan is to|first,? (?:i|we) (?:need to|will)|as an ai|plan to solve|"
    r"tôi sẽ (?:giải|bắt đầu|trình bày|tiến hành|phân tích|xem xét|thực hiện|đi qua)|"
    r"chúng ta sẽ (?:giải|bắt đầu|tiến hành)|"
    r"kế hoạch của tôi(?: là|:)|kế hoạch giải(?: bài)?(?: là|:)|"
    r"trước hết,? tôi cần|tôi cần lập kế hoạch|bước đầu tiên tôi sẽ)\b",
    re.IGNORECASE,
)

# A model explicitly naming its own continuation as wrong/inconsistent --
# distinct from AnswerToken.FLAG (choosing the dedicated "flag" option).
# Heuristic only: a regex over the model's own words, not a semantic check.
_ERROR_LANGUAGE_RE = re.compile(
    r"sai|nhầm|lỗi|không đúng|không chính xác|chưa (?:đúng|chính xác)",
    re.IGNORECASE,
)

# A corrupted step must state the false premise as if it were part of the
# reasoning.  It must not reveal the experimental manipulation by calling
# *itself* wrong, mistaken, or planted.  This is deliberately narrower than
# ``_ERROR_LANGUAGE_RE``: domain content can legitimately mention an error
# rate, a mismatched variable, or a mistaken assignment without disclosing
# that the displayed step is the injected corruption.
_PLANTED_CORRUPTION_DISCLOSURE_PATTERNS = (
    re.compile(
        r"\b(?:ở\s+)?bước\s*(?:thứ\s*)?3(?:\s*này)?\b.{0,80}?\b"
        r"(?:lỗi|sai|nhầm|không\s*đúng|không\s*chính\s*xác)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:bước\s*(?:thứ\s*)?3|bước\s*này|lập\s*luận\s*này|"
        r"suy\s*luận\s*này|phép\s*tính\s*(?:này|trên|ở\s*đây)?|"
        r"giả\s*định\s*này|kết\s*luận\s*này|cách\s*"
        r"(?:tính|suy\s*luận)\s*này)\s*"
        r"(?:đã|đang|bị|là|vẫn|hoàn\s*toàn)?\s*"
        r"(?:một\s*)?(?:lỗi|sai|nhầm|không\s*đúng|không\s*chính\s*xác)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:đây|điều\s*này|nội\s*dung\s*này)\s+"
        r"(?:chính\s+)?là\s+(?:một\s+)?"
        r"(?:lỗi|sai\s*lầm|sai|nhầm|không\s*đúng|không\s*chính\s*xác)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:lỗi|sai\s*lầm|sai\s*lệch)\s+(?:được\s+)?"
        r"(?:cài|gieo|đặt|chèn|tạo)\b|\bcài\s+(?:nhầm|sai|lỗi)\b|"
        r"\b(?:cố\s*ý|chủ\s*đích)\s+(?:cài|đặt|chèn)\s+(?:lỗi|sai)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:gán|áp\s*dụng|tính|phân\s*loại|sắp\s*xếp|ghi|đặt)\s+"
        r"nhầm\b|\bbị\s+(?:\w+\s+){0,3}nhầm\b|\b(?:làm|khiến)\s+"
        r"sai\s*lệch\b",
        re.IGNORECASE,
    ),
)

# A concluding connective ("do đó", "vậy", ...) bound to an explicit option
# letter or an "đáp án <letter>" / "chọn <letter>" phrase. Flags a corrupted
# step 3 that already states its own conclusion outright, which makes a
# model's later match to that letter reading comprehension, not logic
# propagation. Heuristic and intentionally conservative -- a curator should
# confirm or override a positive, not treat it as a final verdict.
_CONCLUSION_CONNECTIVE = r"(?:do đó|vậy|suy ra|nên chọn|kết luận|như vậy)"
_LETTER_MENTION = r"(?:đáp án|phương án|chọn)\s*[:\s]*([A-D])\b"
_CONCLUSION_LEAK_PATTERNS = (
    re.compile(rf"{_CONCLUSION_CONNECTIVE}.{{0,40}}{_LETTER_MENTION}", re.IGNORECASE),
    re.compile(rf"{_LETTER_MENTION}.{{0,40}}{_CONCLUSION_CONNECTIVE}", re.IGNORECASE),
    re.compile(rf"^\s*{_LETTER_MENTION}", re.IGNORECASE),
)

# English counterpart of the pair above, for text that is not gated to
# Vietnamese -- an ExtractedChain step, unlike a CorruptedChain step 3, may
# still be in the source model's own thinking language pre-translation.
#
# Deliberately excludes bare "option" as a trigger word (unlike the VI list's
# "phương án"/"chọn"): elimination-style step-2 reasoning routinely walks
# through "Option A: ... Option B: ..." as per-candidate headers without
# concluding anything -- piloted against real extracted step-2 text and
# every "option"-triggered match found was a false positive of exactly this
# shape. Only an explicit "the answer/the correct answer/the correct
# option" phrase counts as a letter mention here; no anchored
# start-of-text pattern, for the same reason (a step frequently opens with
# "Option A: ..." as its first candidate, not a conclusion).
_EN_CONCLUSION_CONNECTIVE = r"(?:so|thus|therefore|hence|this means)"
_EN_LETTER_MENTION = (
    r"(?:the answer|the correct answer|the correct option)\s*(?:is|:)?\s*([A-D])\b"
)
_CONCLUSION_LEAK_PATTERNS_EN = (
    re.compile(
        rf"{_EN_CONCLUSION_CONNECTIVE}.{{0,40}}{_EN_LETTER_MENTION}", re.IGNORECASE
    ),
    re.compile(
        rf"{_EN_LETTER_MENTION}.{{0,40}}{_EN_CONCLUSION_CONNECTIVE}", re.IGNORECASE
    ),
    re.compile(rf"^\s*{_EN_LETTER_MENTION}", re.IGNORECASE),
)


def classify_reasoning_language(text: str) -> ReasoningLanguage:
    """Classify the language actually observed in a chain.

    Deliberately conservative: records a mixed chain instead of treating a
    few Vietnamese or English tokens as proof that the whole chain is in
    one language.  Do not treat a bare ``I`` as English: it is commonly a
    mathematical variable (investment or current) in Vietnamese reasoning.
    """

    has_vi = bool(_VI_DIACRITIC_RE.search(text) or _VI_WORD_RE.search(text))
    has_en = bool(_EN_WORD_RE.search(text))
    if has_vi and has_en:
        return ReasoningLanguage.MIXED_VI_EN
    if has_vi:
        return ReasoningLanguage.VI
    if has_en:
        return ReasoningLanguage.EN
    return ReasoningLanguage.OTHER


def contains_meta_planning(text: str) -> bool:
    """Return whether text looks like a plan for answering, not reasoning."""

    return bool(_META_PLANNING_RE.search(text))


def contains_error_language(text: str) -> bool:
    """Return whether a continuation explicitly names something as wrong.

    A self-report signal for the corrupted arm, independent of which
    option letter the model ultimately picks.
    """

    return bool(_ERROR_LANGUAGE_RE.search(text))


def discloses_planted_corruption(step_3_text: str) -> bool:
    """Return whether a step-3 text exposes its own injected error.

    This is a hard authoring gate for a ``CorruptedChain``.  It is separate
    from :func:`contains_error_language`, which remains a broad, descriptive
    classifier for a model's later free-form response.
    """

    return any(
        pattern.search(step_3_text)
        for pattern in _PLANTED_CORRUPTION_DISCLOSURE_PATTERNS
    )


def states_conclusion_directly(step_3_text: str) -> bool:
    """Return whether a corrupted step 3 already names its own conclusion.

    If step 3 states the target letter outright, a model matching it
    demonstrates reading comprehension, not logic propagation -- flag such
    corruptions for curator review rather than trusting the propagation
    rate as-is.
    """

    return any(pattern.search(step_3_text) for pattern in _CONCLUSION_LEAK_PATTERNS)


