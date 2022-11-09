#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import contextlib
import sys
import numpy as np
from itertools import repeat
from collections import Counter
from setproctitle import setproctitle
from multiprocessing import Pool, Manager

from fairseq.data.encoders.sentencepiece_bpe import SentencepieceBPE, SentencepieceConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spm_model",
        help="path to spm.model",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["-"],
        help="input files to filter/encode",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        default=["-"],
        help="path to save encoded outputs",
    )
    parser.add_argument("--output_format", choices=["piece", "id"], default="piece")
    parser.add_argument(
        "--keep_empty",
        action="store_true",
        help="keep empty lines",
    )
    parser.add_argument(
        "--proctitle",
        type=str,
        default="preprocessing"
    )
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    setproctitle(args.proctitle)

    assert len(args.inputs) == len(
        args.outputs
    ), "number of input and output paths should match"

    with contextlib.ExitStack() as stack:
        inputs = [
            stack.enter_context(open(input, "r", encoding="utf-8"))
            if input != "-"
            else sys.stdin
            for input in args.inputs
        ]
        outputs = [
            stack.enter_context(open(output, "w", encoding="utf-8"))
            if output != "-"
            else sys.stdout
            for output in args.outputs
        ]

        encoder = MultiprocessingEncoder(args)
        pool = Pool(args.workers, initializer=encoder.initializer)
        encoded_lines = pool.imap(encoder.encode_lines, zip(*inputs), 100)

        stats = Counter()
        for i, (filt, enc_lines) in enumerate(encoded_lines, start=1):
            if filt == "PASS":
                for enc_line, output_h in zip(enc_lines, outputs):
                    print(enc_line, file=output_h)
            else:
                stats["num_filtered_" + filt] += 1
            if i % 10000 == 0:
                print("processed {} lines".format(i), file=sys.stderr)
            if i % 500000 == 0:
                print("processed example: {}".format(enc_line), file=sys.stderr)

        for k, v in stats.most_common():
            print("[{}] filtered {} lines".format(k, v), file=sys.stderr)


class MultiprocessingEncoder(object):

    def __init__(self, args):
        self.args = args

    def initializer(self):
        global bpe
        cfg = SentencepieceConfig(sentencepiece_model=self.args.spm_model)
        bpe = SentencepieceBPE(cfg)

    def encode(self, line):
        global bpe
        if self.args.output_format == "piece":
            tokens = bpe.sp.EncodeAsPieces(line)
            return tokens
        elif self.args.output_format == "id":
            ids = bpe.sp.EncodeAsIds(line)
            return list(map(str, ids))
        else:
            raise NotImplementedError("Only support output formats piece or id. Check your argument.")

    def decode(self, tokens):
        global bpe
        return bpe.decode(tokens)

    def encode_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        enc_lines = []
        for line in lines:
            line = line.strip()
            if len(line) == 0 and not self.args.keep_empty:
                return ["EMPTY", None]
            tokens = self.encode(line)
            enc_lines.append(" ".join(tokens))
        return ["PASS", enc_lines]

    def decode_lines(self, lines):
        dec_lines = []
        for line in lines:
            tokens = map(int, line.strip().split())
            dec_lines.append(self.decode(tokens))
        return ["PASS", dec_lines]


if __name__ == "__main__":
    main()
