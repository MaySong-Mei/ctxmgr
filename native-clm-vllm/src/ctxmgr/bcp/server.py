"""Loopback search server over the BrowseComp-Plus corpus (envs/bcp).

    python -m ctxmgr.bcp.server --data DIR --port PORT

``GET /search?q=...`` returns the top 10 documents as ``docid``, ``score`` and a snippet of
the first 512 o200k tokens; ``GET /doc?docid=...`` returns a document capped at 8,192
tokens. These follow the CLM paper's BrowseComp-Plus setting (top-10 snippets of 512
tokens, document reader capped at 8,192 tokens). Only the corpus is served.
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

TOP_K = 10
SNIPPET_TOKENS = 512
DOCUMENT_TOKENS = 8192


class Corpus:
    def __init__(self, data: Path):
        import bm25s
        import tiktoken
        from datasets import load_dataset

        from .data import CORPUS

        self.retriever = bm25s.BM25.load(str(data / "bm25"))
        self.docids = json.loads((data / "docids.json").read_text())
        corpus = load_dataset(CORPUS, split="train")
        self.text = dict(zip(corpus["docid"], corpus["text"]))
        self.encoder = tiktoken.get_encoding("o200k_base")
        self.bm25s = bm25s
        self.lock = threading.Lock()

    def truncate(self, text: str, limit: int) -> str:
        tokens = self.encoder.encode(text, disallowed_special=())
        return text if len(tokens) <= limit else self.encoder.decode(tokens[:limit])

    def search(self, query: str, k: int = TOP_K) -> list[dict]:
        query_tokens = self.bm25s.tokenize([query], stopwords="en", show_progress=False)
        with self.lock:
            ids, scores = self.retriever.retrieve(query_tokens, k=k, show_progress=False)
        out = []
        for index, score in zip(ids[0], scores[0]):
            docid = self.docids[int(index)]
            out.append(
                {
                    "docid": docid,
                    "score": round(float(score), 3),
                    "snippet": self.truncate(self.text[docid], SNIPPET_TOKENS),
                }
            )
        return out

    def document(self, docid: str) -> dict | None:
        text = self.text.get(docid)
        if text is None:
            return None
        return {"docid": docid, "text": self.truncate(text, DOCUMENT_TOKENS)}


def handler_for(corpus: Corpus):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlsplit(self.path)
            params = parse_qs(url.query)
            if url.path == "/health":
                return self._send(200, {"ok": True, "documents": len(corpus.docids)})
            if url.path == "/search":
                query = (params.get("q") or [""])[0].strip()
                if not query:
                    return self._send(400, {"error": "missing q"})
                return self._send(200, corpus.search(query))
            if url.path == "/doc":
                found = corpus.document((params.get("docid") or [""])[0].strip())
                return self._send(200 if found else 404, found or {"error": "no such docid"})
            return self._send(404, {"error": "unknown path"})

        def log_message(self, *args):  # keep job logs free of query text
            return

    return Handler


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args(argv)
    corpus = Corpus(args.data)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(corpus))
    print(
        json.dumps({"ready": True, "port": args.port, "documents": len(corpus.docids)}), flush=True
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
