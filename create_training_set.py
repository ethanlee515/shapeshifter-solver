#!/usr/bin/env python

import torch
import torch.nn as nn
from torch import Tensor
import random
from config import *

def sample_piece() -> (Tensor, int, int):
    width = random.randrange(piece_width_min, piece_width_max + 1)
    height = random.randrange(piece_height_min, piece_height_max + 1)
    piece = torch.zeros(piece_height_max, piece_width_max, dtype=bool)
    piece[0:height, 0:width] = torch.rand(height, width) < 0.5
    x = random.randrange(0, board_width - width + 1)
    y = random.randrange(0, board_height - height + 1)
    return piece, x, y

def sample_puzzle() -> (list[(Tensor, int, int)]):
    puzzle = list()
    for _ in range(pieces_per_puzzle):
        puzzle.append(sample_piece())
    return puzzle

if __name__ == "__main__":
    training_set_size = 15
    puzzles = list()
    for i in range(training_set_size):
        puzzles.append(sample_puzzle())
    torch.save(puzzles, training_set_path)
