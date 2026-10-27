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
  needles hit the 2,048-token output cap and the cut-off reply was rejected.
- v5: the notes block is kept by the protocol itself. 27B ignored "append only" in v4 and rebuilt the
  whole notes block every event ("delete 2..12" over the notes plus a note re-copying every needle),
  which hit the output cap at 384 needles (K8-C48). Now a delete never removes the notes block (a
  range over it skips it), a replace of the notes block is read as a note, and a note only adds lines
  that the notes block does not hold yet. The notes block is labelled as such in the context view.
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
# v4 = notes are append-only; v5 = notes block can't be deleted or rewritten, appends are deduplicated;
# v6 = v5 semantics, instructions say to note needed lines from every block before deleting it;
# v7 = v6 without "always kept" wording (27B read it as "the notes save needed lines for me");
# v8 = v7 + a bare edit object (no {"edits": [...]} wrapper) is read as a one-edit list.
EDIT_PROMPT_VERSION = "edit-v8"
# Archive conditions (`*_arc`): v8 semantics, but deleted and replaced blocks move to an archive the
# actor can query with RECALL; the instructions say so instead of "gone for good".
ARCHIVE_PROMPT_VERSION = "edit-v8-arc"

_EDIT_TEMPLATE = """\
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
- {"op": "note", "text": "<text>"} adds lines to the notes block right after block 1 (it is created
  on first use and cannot be deleted).
{keep_paragraph}
Your reply is applied as a whole: if any edit is invalid (e.g. it names block 0 or 1), none of your
edits are applied. Do not answer the task here.
Reply with only one JSON object: {"edits": [ ... ]}"""

_GONE_FOR_GOOD = """\
The notes block holds only what you write into it. A deleted block is gone for good: nothing in it
is saved unless you noted it. So whenever you delete blocks that hold lines the task needs, copy
those lines from them into a note in the same reply, every time, not only the first time. Lines the
notes block already holds are dropped automatically, so repeating them does no harm."""

_ARCHIVED = """\
The notes block holds only what you write into it. A deleted block is not lost: its text moves to
an archive outside the context and a one-line placeholder stays in its place. The assistant can bring
lines back from the archive by key (it writes `RECALL <key>`), so archiving bulky blocks is safe,
but anything it needs often is cheaper to keep in the context or in a note. Replacing a block also
archives its old text. Lines the notes block already holds are dropped automatically."""

EDIT_INSTRUCTIONS = _EDIT_TEMPLATE.replace("{keep_paragraph}", _GONE_FOR_GOOD)
ARCHIVE_EDIT_INSTRUCTIONS = _EDIT_TEMPLATE.replace("{keep_paragraph}", _ARCHIVED)
ARCHIVE_PLACEHOLDER = "[ARCHIVED block: "


def is_placeholder(message: dict) -> bool:
    return message["content"].startswith(ARCHIVE_PLACEHOLDER)


def placeholder(message: dict) -> str:
    """One line naming an archived block. Angle brackets are dropped so an archived reply that
    starts with `<<<ANSWER key=...>>>` cannot leave an unterminated sentinel in the context (the KV
    grader would read it as the start of an ANSWER block and swallow the next real one). 80 chars
    cannot hold a whole KV value or Needle line, so a placeholder never earns credit by itself."""
    first = " ".join(message["content"].split("\n", 1)[0].split())
    first = re.sub(r"[<>]", "", first)
    return f"{ARCHIVE_PLACEHOLDER}{first[:80]}]"


PEER_SYSTEM = (
    "You manage the working context of another assistant that is processing a long task. You never "
    "solve the task yourself; you only edit its context so it stays within its size limit."
)

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class EditResult:
    applied: list[dict]
    error: str | None


def _label(i: int, m: dict) -> str:
    kind = "notes" if i >= PROTECTED and _is_notes(m) else m["role"]
    return f"[block {i} | {kind}]"


def render(messages: list[dict]) -> str:
    """The context as numbered blocks, as a peer manager sees it."""
    return "\n\n".join(f"{_label(i, m)}\n{m['content']}" for i, m in enumerate(messages))


def index(messages: list[dict], preview: int = 80) -> str:
    """Block ids with short previews, for a self-manager that already holds the full text."""
    lines = []
    for i, m in enumerate(messages):
        text = " ".join(m["content"].split())
        lines.append(f"{_label(i, m)} {text[:preview]}")
    return "\n".join(lines)


def parse_edits(reply: str) -> tuple[list[dict], str | None]:
    """Extract the edit list; a malformed reply means no edits plus an error string."""
    match = JSON_RE.search(reply or "")
    if not match:
        return [], "no JSON object in reply"
    try:
        parsed = json.loads(match.group(0))
        edits = parsed.get("edits")
    except (json.JSONDecodeError, AttributeError) as error:
        return [], f"invalid JSON: {error}"
    if edits is None and ("op" in parsed or _infer_op(parsed)):
        # v8: a bare edit object is read as a one-edit list (27B in `self` wrote these in 26 of 31
        # v7 events; 0 of 104 as `peer_same`). Marked so the format slip stays countable.
        return [{**parsed, "_wrapped": True}], None
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


def apply_edits(
    messages: list[dict], edits: list[dict], archive: list[str] | None = None
) -> EditResult:
    """Apply a reply's edits as one unit. Ids refer to the pre-edit numbering.

    With an `archive` list, a deleted block leaves a one-line placeholder and its text is appended
    to the archive (a placeholder that is deleted again is simply removed); a replaced block's old
    text is archived too.
    """
    deletes, replaces, applied, rejected, warnings = set(), {}, [], [], []
    notes = []
    for edit in edits:
        op = edit.get("op") if isinstance(edit, dict) else None
        inferred = op is None and isinstance(edit, dict) and _infer_op(edit) is not None
        if inferred:
            op = _infer_op(edit)
        flag = {"inferred_op": True} if inferred else {}
        if isinstance(edit, dict) and edit.get("_wrapped"):
            flag["wrapped_bare_edit"] = True
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
    # v5: the notes block is kept by the protocol. Deletes skip it; a replace of it is read as a note.
    notes_idx = next((i for i, m in enumerate(messages) if i >= PROTECTED and _is_notes(m)), None)
    if notes_idx is not None and notes_idx in deletes:
        deletes.discard(notes_idx)
        warnings.append(f"the notes block (block {notes_idx}) is never deleted; skipped it")
        for a in applied:
            if a["op"] == "delete":
                a["ids"] = [i for i in a["ids"] if i != notes_idx]
        applied = [a for a in applied if a["op"] != "delete" or a["ids"]]
    if notes_idx is not None and notes_idx in replaces:
        text = replaces.pop(notes_idx)
        notes.insert(0, text.removeprefix(NOTES_HEADER).lstrip("\n"))
        warnings.append(f"replace of the notes block (block {notes_idx}) read as a note")
        for a in applied:
            if a["op"] == "replace" and a["id"] == notes_idx:
                a.update(op="note", from_replace=True)
                del a["id"]
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
            if archive is not None and not is_placeholder(m):
                archive.append(m["content"])
                kept.append({**m, "content": placeholder(m)})
            continue
        if i in replaces:
            if archive is not None and not is_placeholder(m):
                archive.append(m["content"])
            m = {**m, "content": replaces[i]}
        if i == notes_idx:
            notes_at = len(kept)
        kept.append(m)
    if notes:
        held = kept[notes_at]["content"].split("\n")[1:] if notes_at is not None else []
        seen = {line.strip() for line in held if line.strip()}
        added, duplicates = [], 0
        for line in "\n".join(notes).split("\n"):
            key = line.strip()
            if not key:
                continue
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            added.append(line)
        if duplicates:
            warnings.append(f"note: dropped {duplicates} lines already in the notes block")
        if added and notes_at is None:
            kept.insert(PROTECTED, {"role": "user", "content": "\n".join([NOTES_HEADER, *added])})
        elif added:
            block = kept[notes_at]
            kept[notes_at] = {**block, "content": "\n".join([block["content"], *added])}
    messages[:] = kept
    return EditResult(applied, "; ".join(warnings) or None)
