import argparse
from .core import run_analysis

def main():
    parser = argparse.ArgumentParser(description="VISIA-style face analysis panel (approx).")
    parser.add_argument("input", help="Input image path (jpg/png).")
    parser.add_argument("-o", "--output", default="analysis_panel.png", help="Output image path.")
    parser.add_argument("--max-width", type=int, default=900, help="Resize input to this max width.")
    args = parser.parse_args()

    out = run_analysis(args.input, args.output, args.max_width)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
