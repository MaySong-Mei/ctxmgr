"""Smoke test for a running vLLM server: short chat, long-context recall, throughput."""

import argparse
import random
import time

from openai import OpenAI


def chat(client, model, prompt, max_tokens):
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
        # Qwen3.5 thinks by default; the pilot only checks serving, so turn it off.
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return resp, time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-pairs", type=int, default=1500, help="KV pairs in the long prompt")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")

    resp, dt = chat(client, args.model, "Reply with exactly: OK", 8)
    print(f"[short] {dt:.2f}s -> {resp.choices[0].message.content!r}")

    rng = random.Random(0)
    pairs = {f"key_{i:05d}": f"{rng.randrange(10**8):08d}" for i in range(args.n_pairs)}
    target = rng.choice(list(pairs))
    lines = "\n".join(f"{k}: {v}" for k, v in pairs.items())
    prompt = f"{lines}\n\nWhat is the value of {target}? Answer with the value only."
    resp, dt = chat(client, args.model, prompt, 16)
    answer = resp.choices[0].message.content.strip()
    usage = resp.usage
    print(
        f"[long] prompt_tokens={usage.prompt_tokens} {dt:.2f}s "
        f"expected={pairs[target]} got={answer!r} correct={pairs[target] in answer}"
    )

    resp, dt = chat(client, args.model, "Write a 300-word story about a lighthouse.", 512)
    out = resp.usage.completion_tokens
    print(f"[decode] {out} tokens in {dt:.2f}s -> {out / dt:.1f} tok/s")


if __name__ == "__main__":
    main()
