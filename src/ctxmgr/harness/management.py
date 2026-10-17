"""Shared context-edit protocol for self and peer managers (draft; freeze with the team).

Both conditions get the same action space and the same instructions: the live context is shown as
numbered blocks, and the editor answers with one JSON object of edits. Blocks 0 (system prompt) and
1 (task instruction) are protected. What differs between conditions is only *who* edits and *from
where* (see `agents.py`): self-management asks the actor inside its own conversation; a peer is a
separate session that reads the context as a document.

Validation rules (from the first managed pilot on 2B/4B):
- A delete is all-or-nothing: if any id it names is protected, out of range or not an integer, the
  whole edit is rejected. Dropping the bad ids and applying the rest turned 2B's "delete 0..N" into
  a wipe of every editable block.
- A delete takes either an id list or an inclusive range ("from"/"to"), not both, so long id lists
  don't run into the manager's output-token cap.
- An edit without "op" that has an integer "id" and a string "text" is read as a replace (4B wrote
  replaces this way). Any other edit without a known op is rejected.
- Edits are validated independently: one rejected edit doesn't block the others.
"""

import json
import re
from dataclasses import dataclass

PROTECTED = 2  # system prompt + task instruction

EDIT_INSTRUCTIONS = """\
The context is shown as numbered blocks. It is close to its size limit, so free space by editing it.
Allowed edits (blocks 0 and 1 cannot be edited):
- {"op": "delete", "ids": [<block ids>]} removes blocks.
- {"op": "delete", "from": <first id>, "to": <last id>} removes a run of blocks (both ends included).
  A delete that names block 0 or 1 is rejected entirely.
- {"op": "replace", "id": <block id>, "text": "<new text>"} rewrites one block, e.g. to keep only the
  lines later operations will need.
Keep everything that later operations will need, copied exactly. Do not answer the task here.
Reply with only one JSON object: {"edits": [ ... ]}"""

PEER_SYSTEM = (
    "You manage the working context of another assistant that is processing a long task. You never "
    "solve the task yourself; you only edit its context so it stays within its size limit."
)

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class EditResult:
    applied: list[dict]
    error: str | None


def render(messages: list[dict]) -> str:
    """The context as numbered blocks, as a peer manager sees it."""
    return "\n\n".join(f"[block {i} | {m['role']}]\n{m['content']}" for i, m in enumerate(messages))


def index(messages: list[dict], preview: int = 80) -> str:
    """Block ids with short previews, for a self-manager that already holds the full text."""
    lines = []
    for i, m in enumerate(messages):
        text = " ".join(m["content"].split())
        lines.append(f"[block {i} | {m['role']}] {text[:preview]}")
    return "\n".join(lines)


def parse_edits(reply: str) -> tuple[list[dict], str | None]:
    """Extract the edit list; a malformed reply means no edits plus an error string."""
    match = JSON_RE.search(reply or "")
    if not match:
        return [], "no JSON object in reply"
    try:
        edits = json.loads(match.group(0)).get("edits")
    except (json.JSONDecodeError, AttributeError) as error:
        return [], f"invalid JSON: {error}"
    if not isinstance(edits, list):
        return [], "missing 'edits' list"
    return edits, None


def _is_id(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _delete_ids(edit: dict, size: int) -> tuple[list[int] | None, str | None]:
    """The ids a delete names, or None plus the reason the whole edit is rejected."""
    has_list, has_range = "ids" in edit, "from" in edit or "to" in edit
    if has_list == has_range:
        return None, f"delete needs 'ids' or 'from'/'to' {str(edit)[:80]}"
    if has_range:
        first, last = edit.get("from"), edit.get("to")
        if not (_is_id(first) and _is_id(last)) or first > last:
            return None, f"bad delete range {str(first)[:20]}..{str(last)[:20]}"
        # Check the endpoints before expanding: a huge "to" must not build a huge list.
        if first < PROTECTED:
            return None, f"delete names protected blocks {str(edit)[:80]}"
        if last >= size:
            return None, f"delete names missing blocks {str(edit)[:80]}"
        ids = list(range(first, last + 1))
    else:
        ids = edit["ids"]
        if not isinstance(ids, list) or not ids or not all(_is_id(i) for i in ids):
            return None, f"bad delete ids {str(ids)[:80]}"
    if min(ids) < PROTECTED:
        return None, f"delete names protected blocks {str(edit)[:80]}"
    if max(ids) >= size:
        return None, f"delete names missing blocks {str(edit)[:80]}"
    return sorted(set(ids)), None


def apply_edits(messages: list[dict], edits: list[dict]) -> EditResult:
    """Apply valid edits; skip (and report) invalid ones. Ids refer to the pre-edit numbering."""
    deletes, replaces, applied, problems = set(), {}, [], []
    for edit in edits:
        op = edit.get("op") if isinstance(edit, dict) else None
        if op is None and isinstance(edit, dict) and _is_id(edit.get("id")) and "text" in edit:
            op = "replace"  # 4B omitted "op" on replaces; id + text is unambiguous
        if op == "delete":
            ids, problem = _delete_ids(edit, len(messages))
            if problem:
                problems.append(problem)
            else:
                deletes.update(ids)
                applied.append({"op": "delete", "ids": ids})
        elif op == "replace" and isinstance(edit.get("text"), str):
            i = edit.get("id")
            if _is_id(i) and PROTECTED <= i < len(messages):
                replaces[i] = edit["text"]
                applied.append({"op": "replace", "id": i, "chars": len(edit["text"])})
            else:
                problems.append(f"bad replace id {i}")
        else:
            problems.append(f"unknown edit {str(edit)[:80]}")
    kept = []
    for i, m in enumerate(messages):
        if i in deletes:
            continue
        kept.append({**m, "content": replaces[i]} if i in replaces else m)
    messages[:] = kept
    return EditResult(applied, "; ".join(problems) or None)
