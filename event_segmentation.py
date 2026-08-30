from __future__ import annotations

import re


_BOUNDARY_RE = re.compile(
    r"[，,。！？!?；;\n\r]+|"
    r"(?=还要|还得|另外|此外|并且|以及|"
    r"同时(?=还要|需要|给|去|打|回|交|预约|联系|准备|处理)|"
    r"还有(?=一|两|三|个|件|份|封|给|去|打|回|交|预约|联系|准备|处理|水费|邮件|报告)|"
    r"并(?=给|去|打|回|交|预约|联系))"
)
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.、])\s*")
_STANDALONE_SHIFTS = {"另外", "此外", "同时"}
_INDEPENDENT_PREFIX_RE = re.compile(
    r"^(?:另外|此外|还要|还得|以及|并且|"
    r"同时(?=还要|需要|给|去|打|回|交|预约|联系|准备|处理)|"
    r"还有(?=一|两|三|个|件|份|封|给|去|打|回|交|预约|联系|准备|处理|水费|邮件|报告)|"
    r"并(?=给|去|打|回|交|预约|联系))"
)
_ANAPHORA_RE = re.compile(r"^(?:这件事|这事|这让我|它|对此|为此|想到这(?:件事)?)")
_CAUSAL_RE = re.compile(r"因为|由于|所以|因此|因而|导致|以至|原因|才会|让我|使我|令我|造成")
_SUPPLEMENT_RE = re.compile(r"^(?:还差|还没|其中|主要|具体|也就是|也就是说|特别是|尤其是)")
_STEP_RE = re.compile(r"^(?:先|再|然后|接着|下一步)")
_GOAL_RE = re.compile(r"为了|目标|计划|打算|准备|需要|完成")
_DESIRE_ACTION_RE = re.compile(
    r"^(?:我)?(?:现在|今晚|此刻)?(?:一直|特别|非常|很|有点)?"
    r"(?:想|想要|想去|想做|想吃|想喝|希望|渴望|忍不住想|有(?:一种)?冲动)"
)
_INTERNAL_STATE_RE = re.compile(
    r"难耐|难受|不舒服|烦闷|烦躁|焦虑|担心|紧张|不安|害怕|心慌|"
    r"嘴馋|饥饿|(?:很|太|有点)?饿|口渴|(?:很|太|有点)?渴|"
    r"兴奋|冲动|欲望|寂寞|空虚|堵得慌|静不下来|停不下来"
)
_TASK_RE = re.compile(
    r"要|需要|得|想|打算|计划|准备|继续|完成|提交|回复|回邮件|回电话|"
    r"开会|考试|练习|报告|汇报|作业|论文|水费|预约|处理|联系|整理|写|交|做|买"
)
_STATE_TERMS = (
    "睡不着",
    "失眠",
    "焦虑",
    "担心",
    "放不下",
    "紧张",
    "烦躁",
    "不安",
    "害怕",
    "心慌",
    "来不及",
    "会出错",
    "停不下来",
    "静不下来",
)
_TOPIC_STOP_PHRASES = tuple(sorted({
    "这件事情", "这件事", "睡不着", "会出错", "来不及", "明天", "后天", "今天", "今晚",
    "下周", "现在", "一直", "有点", "因为", "由于", "所以", "因此", "因而", "导致",
    "原因", "让我", "使我", "令我", "另外", "此外", "还要", "还得", "还需", "并且",
    "以及", "打算", "计划", "准备", "继续", "完成", "需要", "想做", "要做", "我想",
    "我要", "我需要", "担心", "焦虑", "放不下", "紧张", "烦躁", "不安", "害怕", "心慌",
    "还没有", "没有", "还没", "已经", "一点", "一下", "这事", "对此", "为此", "事情",
    "的", "了", "我", "它", "是", "会", "再", "先", "然后", "接着",
}, key=len, reverse=True))


def split_candidate_clauses(text: str) -> list[str]:
    raw_parts = [_clean_clause(part) for part in _BOUNDARY_RE.split(text)]
    clauses: list[str] = []
    pending_shift = ""
    for part in raw_parts:
        if not part:
            continue
        if part in _STANDALONE_SHIFTS:
            pending_shift = part
            continue
        if pending_shift:
            part = f"{pending_shift}，{part}"
            pending_shift = ""
        clauses.append(part)
    return clauses


def segment_events(text: str) -> list[str]:
    clauses = split_candidate_clauses(text)
    return [compose_event([clauses[index] for index in group]) for group in group_related_texts(clauses)]


def split_explicit_events(text: str) -> list[str]:
    """Split a model-produced item only where the user explicitly starts another item."""
    clauses = split_candidate_clauses(text)
    groups: list[list[str]] = []
    current: list[str] = []
    for clause in clauses:
        if current and _is_explicitly_independent(clause):
            groups.append(current)
            current = []
        current.append(clause)
    if current:
        groups.append(current)
    return [compose_event(group) for group in groups]


def group_related_texts(texts: list[str]) -> list[list[int]]:
    if not texts:
        return []
    parents = list(range(len(texts)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for index in range(1, len(texts)):
        if _should_merge_adjacent(texts[index - 1], texts[index]):
            union(index - 1, index)

    # A later explanatory clause can refer back across a short state clause.
    for left in range(len(texts)):
        for right in range(left + 2, min(len(texts), left + 4)):
            if _is_explicitly_independent(texts[right]):
                continue
            shared_topics = _topic_keys(texts[left]) & _topic_keys(texts[right])
            if shared_topics and (_is_causal(texts[right]) or _is_anaphoric(texts[right])):
                union(left, right)

    grouped: dict[int, list[int]] = {}
    for index in range(len(texts)):
        grouped.setdefault(find(index), []).append(index)
    return sorted(grouped.values(), key=lambda group: group[0])


def compose_event(clauses: list[str]) -> str:
    kept: list[str] = []
    covered_topics: set[str] = set()
    covered_states: set[str] = set()
    for clause in clauses:
        cleaned = _strip_independent_prefix(clause) if not kept else clause
        topics = _topic_keys(cleaned)
        states = _state_keys(cleaned)
        is_redundant_explanation = (
            bool(kept)
            and bool(topics or states)
            and topics <= covered_topics
            and states <= covered_states
            and (_is_causal(cleaned) or _is_anaphoric(cleaned))
        )
        if is_redundant_explanation:
            continue
        if cleaned and cleaned not in kept:
            kept.append(cleaned)
            covered_topics.update(topics)
            covered_states.update(states)
    return "，".join(kept)


def _should_merge_adjacent(left: str, right: str) -> bool:
    if _is_explicitly_independent(right):
        return False
    if _is_anaphoric(right) or _is_causal(left) or _is_causal(right):
        return True
    if _SUPPLEMENT_RE.search(right):
        return True
    if _STEP_RE.search(right) and (_GOAL_RE.search(left) or _STEP_RE.search(left)):
        return True
    if _is_internal_state(left) and _is_desire_action(right):
        return True
    if _is_state_only(left) and _is_task(right):
        return True
    if _is_task(left) and _is_state_only(right):
        return True
    shared_topics = _topic_keys(left) & _topic_keys(right)
    return bool(shared_topics) and (_has_state(left) or _has_state(right))


def _topic_keys(text: str) -> set[str]:
    normalized = text.lower()
    for phrase in _TOPIC_STOP_PHRASES:
        normalized = normalized.replace(phrase, "")
    runs = re.findall(r"[\u4e00-\u9fff]+|[a-z0-9]+", normalized)
    keys: set[str] = set()
    for run in runs:
        if len(run) < 2:
            continue
        keys.add(run)
        if re.fullmatch(r"[\u4e00-\u9fff]+", run):
            keys.update(run[index:index + 2] for index in range(len(run) - 1))
    return keys


def _state_keys(text: str) -> set[str]:
    return {term for term in _STATE_TERMS if term in text}


def _has_state(text: str) -> bool:
    return bool(_state_keys(text))


def _is_state_only(text: str) -> bool:
    return _has_state(text) and not _is_task(text)


def _is_task(text: str) -> bool:
    return bool(_TASK_RE.search(text))


def _is_desire_action(text: str) -> bool:
    return bool(_DESIRE_ACTION_RE.search(text.strip()))


def _is_internal_state(text: str) -> bool:
    return bool(_INTERNAL_STATE_RE.search(text)) and not _is_task(text)


def _is_causal(text: str) -> bool:
    return bool(_CAUSAL_RE.search(text))


def _is_anaphoric(text: str) -> bool:
    return bool(_ANAPHORA_RE.search(text.strip()))


def _is_explicitly_independent(text: str) -> bool:
    return bool(_INDEPENDENT_PREFIX_RE.search(text.strip()))


def _strip_independent_prefix(text: str) -> str:
    return _INDEPENDENT_PREFIX_RE.sub("", text.strip()).lstrip("，, ")


def _clean_clause(text: str) -> str:
    return _BULLET_RE.sub("", text).strip(" ，,。；;！？!?\t")
