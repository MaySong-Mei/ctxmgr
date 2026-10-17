"""Shared context-edit protocol for self and peer managers (draft; freeze with the team).

Both conditions get the same action space and the same instructions: the live context is shown as
numbered blocks, and the editor answers with one JSON object of edits. Blocks 0 (system prompt) and
1 (task instruction) are protected. What differs between conditions is only *who* edits and *from
where* (see `agents.py`): self-management asks the actor inside its own conversation; a peer is a
separate session that reads the context as a document.
"""

import json
import re
from dataclasses import dataclass

PROTECTED = 2  # system prompt + task instruction

EDIT_INSTRUCTIONS = """\
The context is shown as numbered blocks. It is close to its size limit, so free space by editing it.
Allowed edits (blocks 0 and 1 cannot be edited):
- {"op": "delete", "ids": [<block ids>]} removes blocks.
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


def apply_edits(messages: list[dict], edits: list[dict]) -> EditResult:
    """Apply valid edits; skip (and report) invalid ones. Ids refer to the pre-edit numbering."""
    deletes, replaces, applied, problems = set(), {}, [], []
    for edit in edits:
        op = edit.get("op") if isinstance(edit, dict) else None
        if op == "delete" and isinstance(edit.get("ids"), list):
            ids = [i for i in edit["ids"] if isinstance(i, int) and PROTECTED <= i < len(messages)]
            if len(ids) != len(edit["ids"]):
                problems.append(f"bad delete ids {edit['ids']}")
            deletes.update(ids)
            if ids:
                applied.append({"op": "delete", "ids": sorted(ids)})
        elif op == "replace" and isinstance(edit.get("text"), str):
            i = edit.get("id")
            if isinstance(i, int) and PROTECTED <= i < len(messages):
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
