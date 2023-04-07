#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2022 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

import argparse
import os
import textgrid
from tqdm import tqdm

from seq2seq_vc.utils import find_files

def get_basename(path):
    return os.path.splitext(os.path.split(path)[-1])[0]

def get_parser():
    parser = argparse.ArgumentParser(description="objective evaluation script.")
    parser.add_argument("--textgrid_dir", required=True, type=str, help="directory for textgrid files")
    parser.add_argument("--output", required=True, type=str, help="path to output segments file")
    return parser

def main():
    args = get_parser().parse_args()

    with open(args.output, "w") as outf:
        textgrid_files = sorted(find_files(args.textgrid_dir, query="*.TextGrid"))

        for f in tqdm(textgrid_files):
            filename = get_basename(f)

            tg = textgrid.TextGrid.fromFile(f)
            l = len(tg[0])
            
            # find start time
            for i in range(l):
                if len(tg[0][i].mark) > 0:
                    start_time = float(tg[0][i].minTime)
                    break

            # find end time
            for i in reversed(range(l)):
                if len(tg[0][i].mark) > 0:
                    end_time = float(tg[0][i].maxTime)
                    break

            outf.write(f"{filename} {filename} {start_time} {end_time}\n")

if __name__ == "__main__":
    main()