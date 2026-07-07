#!/usr/bin/env python3
"""Create a sampled version of a large CSV targeting a specific file size.

Usage:
  python scripts/sample_csv.py --input creditcard.csv --output creditcard_sampled.csv --target-mb 99

This reads the input CSV in chunks and randomly samples each chunk by the
calculated fraction so the output is approximately `target_mb` in size.
"""
import os
import argparse
import math
import sys
import pandas as pd


def sample_csv_to_target(input_path, output_path, target_mb=99, chunksize=200_000, random_state=42):
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    cur_bytes = os.path.getsize(input_path)
    cur_mb = cur_bytes / (1024 * 1024)
    print(f"Current file size: {cur_mb:.2f} MB")

    if cur_mb <= target_mb:
        print("File already under target size — copying file.")
        # copy without loading entire file into memory
        with open(input_path, 'rb') as src, open(output_path, 'wb') as dst:
            while True:
                buf = src.read(1 << 20)
                if not buf:
                    break
                dst.write(buf)
        return

    frac = target_mb / cur_mb
    frac = max(min(frac, 1.0), 0.000001)
    print(f"Sampling fraction: {frac:.6f} (target {target_mb} MB)")

    written = 0
    header_written = False
    reader = pd.read_csv(input_path, chunksize=chunksize)
    for i, chunk in enumerate(reader):
        # sample rows from this chunk
        try:
            sampled = chunk.sample(frac=frac, random_state=(random_state + i))
        except ValueError:
            # if frac * len(chunk) < 1, sample at least one row deterministically
            if len(chunk) > 0:
                sampled = chunk.iloc[:1]
            else:
                sampled = chunk

        mode = 'w' if not header_written else 'a'
        header = not header_written
        sampled.to_csv(output_path, mode=mode, header=header, index=False)
        header_written = True
        written += len(sampled)
        if (i + 1) % 5 == 0:
            # report progress
            out_bytes = os.path.getsize(output_path)
            print(f"Chunks processed: {i+1}, rows written: {written}, output size: {out_bytes/(1024*1024):.2f} MB")

    out_bytes = os.path.getsize(output_path)
    out_mb = out_bytes / (1024 * 1024)
    print(f"Finished. Wrote {written} rows -> {output_path} ({out_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='Input CSV path')
    parser.add_argument('--output', '-o', required=True, help='Output CSV path')
    parser.add_argument('--target-mb', type=float, default=99.0, help='Target maximum size in MB')
    parser.add_argument('--chunksize', type=int, default=200000, help='Pandas read_csv chunksize')
    parser.add_argument('--random-state', type=int, default=42)
    args = parser.parse_args()

    sample_csv_to_target(args.input, args.output, target_mb=args.target_mb, chunksize=args.chunksize, random_state=args.random_state)


if __name__ == '__main__':
    main()
