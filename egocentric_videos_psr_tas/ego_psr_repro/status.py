#!/usr/bin/env python
"""Show which pipeline stages are already built (outputs present) vs still to run."""
import argparse

import orchestrate as o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="all")
    args = ap.parse_args()
    archs = o.ARCHES if args.arch == "all" else [a.strip() for a in args.arch.split(",")]
    order = o.resolve(archs)
    ndone = sum(1 for s in order if o.done(s))
    print(f"=== Pipeline status  archs: {', '.join(archs)}  ({ndone}/{len(order)} stages done) ===\n")
    print(f"  {'stage':<18s} {'status':<6s} {'gpu':<3s} produces")
    print("  " + "-" * 78)
    for sid in order:
        s = o.STAGES[sid]
        st = "DONE" if o.done(sid) else "TODO"
        prod = s["done"] or "(harness output)"
        print(f"  {sid:<18s} {st:<6s} {'GPU' if s['gpu'] else 'cpu':<3s} {prod}")


if __name__ == "__main__":
    main()
