"""Shared context-edit protocol for self and peer managers (draft; freeze with the team).

Both conditions get the same action space and the same instructions: the live context is shown as
numbered blocks, and the editor answers with one JSON object of edits. Blocks 0 (system prompt) and
1 (task instruction) are protected. What differs between conditions is only *who* edits and *from
where* (see `agents.py`): self-management asks the actor inside its own conversation; a peer is a
separate session that reads the context as a document.

Validation rules (from the managed pilots on 2B/4B and the 27B peer):
- A reply is all-or-nothing: if any edit in it is invalid, none is applied and every reason is
  reported. A reply is one plan; applying half of it destroyed data (27B paired "copy the needles
  into block 1" with "delete blocks 2-11"; the copy was rejected, the delete went through).
- A delete takes either an id list or an inclusive range ("from"/"to"), not both, so long id lists
  don't run into the manager's output-token cap. Protected (0/1), missing, negative or non-integer
  ids make it invalid.
- A missing "op" is inferred when the fields are unambiguous: "id" + "text" is a replace (4B wrote
  replaces this way), "ids" or "from"/"to" without "text" is a delete. Inferred edits carry
  `"inferred_op": true` in the log, so protocol slips stay countable. Anything else is invalid.
- `note` appends to the notes block: one block right after the task instruction that holds
  consolidated content, created on first use. It gives a manager a legal place to gather what must be
  kept (27B otherwise tried to rewrite the protected instruction). Appending (v4) instead of rewriting
  the whole block (v3) keeps a note's size proportional to the new content: in v3, rewriting 53+
  needles hit the 2,048-token output cap and the cut-off reply was rejected. To prune notes, replace
  the notes block by its id; it keeps its header and stays the notes block.
- Conflicts within one valid reply are resolved, not rejected: a block both replaced and deleted is
  kept with its replacement, a block replaced twice keeps the last text; both are reported. Several
  notes in one reply are appended in order.
"""

import json
import re
from dataclasses import dataclass

PROTECTED = 2  # system prompt + task instruction
NOTES_HEADER = "[NOTES kept by context management]"

# Bump when EDIT_INSTRUCTIONS or the edit semantics change; logged per episode.
# v1 = first pilots; v2 = says what to keep; v3 = all-or-nothing replies + notes block;
# v4 = notes are append-only.
EDIT_PROMPT_VERSION = "edit-v4"

EDIT_INSTRUCTIONS = """\
The context is shown as numbered blocks. It is close to its size limit, so free space by editing it.
Your edits are judged by whether the assistant can still complete its task afterwards, not by how
much space you free.
First read block 1, the task instruction: it says what the assistant must keep and what later
operations will ask about. Before removing a block, check it for such information:
- If a block holds nothing the task needs, delete it.
- If a block mixes needed lines with filler, replace it with only the needed lines, copied exactly,
  or copy those lines into the notes block and then delete the block.
- Never delete the only copy of something the task needs. Deleting every block is almost always wrong.
Allowed edits (blocks 0 and 1 cannot be edited):
- {"op": "delete", "ids": [<block ids>]} removes blocks.
- {"op": "delete", "from": <first id>, "to": <last id>} removes a run of blocks (both ends included).
- {"op": "replace", "id": <block id>, "text": "<new text>"} rewrites one block.
- {"op": "note", "text": "<text>"} appends text to the notes block, kept right after block 1 (it is
  created on first use). Only write what is new; earlier notes stay. To prune the notes, replace the
  notes block by its id.
Your reply is applied as a whole: if any edit is invalid (e.g. it names block 0 or 1), none of your
edits are applied. Do not answer the task here.
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
        if not all(_is_id(i) and i >= 0 for i in (first, last)) or first > last:
            return None, f"bad delete range {str(first)[:20]}..{str(last)[:20]}"
    else:
        ids = edit["ids"]
        if not isinstance(ids, list) or not ids or not all(_is_id(i) and i >= 0 for i in ids):
            return None, f"bad delete ids {str(ids)[:80]}"
        first, last = min(ids), max(ids)
    # Check the endpoints before expanding a range: a huge "to" must not build a huge list.
    if first < PROTECTED:
        return None, f"delete names protected blocks {str(edit)[:80]}"
    if last >= size:
        return None, f"delete names missing blocks {str(edit)[:80]}"
    return (list(range(first, last + 1)) if has_range else sorted(set(edit["ids"]))), None


def _infer_op(edit: dict) -> str | None:
    if _is_id(edit.get("id")) and "text" in edit:
        return "replace"
    if ("ids" in edit or "from" in edit or "to" in edit) and "text" not in edit:
        return "delete"
    return None


def _is_notes(message: dict) -> bool:
    return message["content"].startswith(NOTES_HEADER)


def apply_edits(messages: list[dict], edits: list[dict]) -> EditResult:
    """Apply a reply's edits as one unit. Ids refer to the pre-edit numbering."""
    deletes, replaces, applied, rejected, warnings = set(), {}, [], [], []
    notes = []
    for edit in edits:
        op = edit.get("op") if isinstance(edit, dict) else None
        inferred = op is None and isinstance(edit, dict) and _infer_op(edit) is not None
        if inferred:
            op = _infer_op(edit)
        flag = {"inferred_op": True} if inferred else {}
        if op == "delete":
            ids, problem = _delete_ids(edit, len(messages))
            if problem:
                rejected.append(problem)
            else:
                deletes.update(ids)
                applied.append({"op": "delete", "ids": ids, **flag})
        elif op == "replace" and isinstance(edit.get("text"), str):
            i = edit.get("id")
            if _is_id(i) and PROTECTED <= i < len(messages):
                if i in replaces:
                    warnings.append(f"block {i} replaced twice; kept the last text")
                    applied = [a for a in applied if not (a["op"] == "replace" and a["id"] == i)]
                replaces[i] = edit["text"]
                applied.append({"op": "replace", "id": i, "chars": len(edit["text"]), **flag})
            else:
                rejected.append(f"bad replace id {i}")
        elif op == "note" and isinstance(edit.get("text"), str):
            notes.append(edit["text"])
            applied.append({"op": "note", "chars": len(edit["text"])})
        else:
            rejected.append(f"unknown edit {str(edit)[:80]}")
    if rejected:
        return EditResult([], "rejected whole reply: " + "; ".join(rejected + warnings))
    # A replacement is a deliberate "keep this": a delete in the same reply skips that block.
    conflicts = deletes & replaces.keys()
    if conflicts:
        deletes -= conflicts
        warnings.append(f"kept replaced blocks {sorted(conflicts)} that a delete also named")
        for a in applied:
            if a["op"] == "delete":
                a["ids"] = [i for i in a["ids"] if i not in conflicts]
        applied = [a for a in applied if a["op"] != "delete" or a["ids"]]
    kept, notes_at = [], None
    for i, m in enumerate(messages):
        if i in deletes:
            continue
        is_notes = i >= PROTECTED and _is_notes(m)
        if i in replaces:
            text = replaces[i]
            # Replacing the notes block prunes it; it stays the notes block.
            if is_notes and not text.startswith(NOTES_HEADER):
                text = f"{NOTES_HEADER}\n{text}"
            m = {**m, "content": text}
        if is_notes and notes_at is None:
            notes_at = len(kept)
        kept.append(m)
    if notes:
        added = "\n".join(notes)
        if notes_at is None:
            kept.insert(PROTECTED, {"role": "user", "content": f"{NOTES_HEADER}\n{added}"})
        else:
            block = kept[notes_at]
            kept[notes_at] = {**block, "content": f"{block['content']}\n{added}"}
    messages[:] = kept
    return EditResult(applied, "; ".join(warnings) or None)
