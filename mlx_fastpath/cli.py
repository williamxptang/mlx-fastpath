"""mlx-fastpath CLI.

  mlx-fastpath generate --model PATH --prompt TEXT [--max-tokens N] [--memory-limit-gb G]
  mlx-fastpath serve [optiq serve args…]   # optiq's own OpenAI-compatible server, with the speedups applied
  mlx-fastpath n2 SRC                      # build the N2-gdnhead copy SRC-N2-gdnhead (load() uses it by default)
  mlx-fastpath versions
"""
import argparse
import json
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["serve"]:
        from .apply import apply
        print("[mlx-fastpath] applied:", apply(), flush=True)
        from optiq.cli import cli
        return cli.main(args=argv, prog_name="mlx-fastpath")

    ap = argparse.ArgumentParser(prog="mlx-fastpath")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--model", required=True)
    g.add_argument("--prompt", required=True)
    g.add_argument("--max-tokens", type=int, default=256)
    g.add_argument("--memory-limit-gb", type=float, default=None)
    g.add_argument("--no-n2", action="store_true", help="load the model as given, not its N2-gdnhead copy")
    n = sub.add_parser("n2")
    n.add_argument("src")
    sub.add_parser("versions")
    a = ap.parse_args(argv)

    from . import __version__
    from .apply import TESTED, installed_versions
    if a.cmd == "versions":
        print(json.dumps({"mlx-fastpath": __version__, "installed": installed_versions(), "tested": TESTED}, indent=1))
        return 0

    if a.cmd == "n2":
        from . import n2
        from .guard import MemoryGuard
        import mlx.core as mx
        with MemoryGuard(min_free_gb=5, max_footprint_gb=4):      # streaming: ~0.5 GB
            mx.set_memory_limit(3 << 30)
            mx.set_cache_limit(256 << 20)
            dst = n2.build(a.src)
            problems = n2.check(a.src, dst)
        print(json.dumps({"n2": str(dst), "problems": problems[:20]}, indent=1))
        return 1 if problems else 0

    from .load import load
    engine = load(a.model, memory_limit_gb=a.memory_limit_gb, n2=False if a.no_n2 else None)
    from .apply import apply, generate
    print("[mlx-fastpath] applied:", apply(), file=sys.stderr, flush=True)
    tok = engine.tokenizer
    ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": a.prompt}], tokenize=False,
                                             add_generation_prompt=True))
    st = generate(engine, ids, max_tokens=a.max_tokens)
    print(st.text)
    print(f"[mlx-fastpath] {st.generated_tokens} tokens, {st.decode_tok_s:.1f} tok/s, "
          f"accept {st.acceptance_rate:.2f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
