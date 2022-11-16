#!/usr/bin/env python

import random
import torch
import torch.nn as nn
from config import *
from torch import Tensor

def sample_piece() -> (Tensor, int, int):
    width = random.randrange(piece_width_min, piece_width_max + 1)
    height = random.randrange(piece_height_min, piece_height_max + 1)
    piece = torch.zeros(piece_height_max, piece_width_max, dtype=bool, device=device)
    piece[0:height, 0:width] = torch.rand(height, width, device=device) < 0.5
    x = random.randrange(0, board_width - width + 1)
    y = random.randrange(0, board_height - height + 1)
    return piece, x, y

def sample_puzzle() -> (list[(Tensor, int, int)]):
    pieces = list()
    for _ in range(pieces_per_puzzle):
        pieces.append(sample_piece())
    return pieces

def add_piece(board: Tensor, piece: Tensor, x: int, y: int) -> Tensor:
    board[y:y+piece_height_max, x:x+piece_width_max] += piece[0:board_height - y, 0:board_width - x]
    board[y:y+piece_height_max, x:x+piece_width_max] = torch.remainder(board[y:y+piece_height_max, x:x+piece_width_max], cycle_length)
    return None

def make_board(pieces: list[(Tensor, int, int)]):
    board = torch.zeros(board_height, board_width, device=device, dtype=torch.uint8)
    for p, x, y in pieces:
        add_piece(board, p, x, y)
    return board

class ShapeshiftSolver(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_size = board_width * board_height + pieces_per_puzzle * piece_width_max * piece_height_max
        out_size = pieces_per_puzzle * 2
        net_width = self.in_size
        self.network = nn.Sequential(
                nn.Linear(self.in_size, net_width),
                nn.ReLU(),
                nn.Linear(net_width, net_width),
                nn.ReLU(),
                nn.Linear(net_width, net_width),
                nn.ReLU(),
                nn.Linear(net_width, out_size))

    def forward(self, board, pieces):
        v = [board.view(board_width * board_height).float()]
        for piece, _, _ in pieces:
            v.append(piece.view(piece_width_max * piece_height_max))
        net_in = torch.cat(v)
        return self.network(net_in)

if __name__ == "__main__":
    solver = ShapeshiftSolver()
    solver.to(device)
    optim = torch.optim.Adam(solver.parameters())
    for ctr in range(num_iterations):
        pieces = sample_puzzle()
        # preparing ingredients
        board = make_board(pieces)
        solution = list()
        for _, x, y in pieces:
            solution.extend([x, y])
        solution = torch.tensor(solution, device=device, dtype=torch.float)
        # Training loop
        optim.zero_grad()
        output = solver(board, pieces)
        loss = torch.nn.functional.mse_loss(output, solution)
        loss.backward()
        optim.step()
        if ctr % steps_per_update == 0:
            print(f"current step = {ctr}; loss = {float(loss)}")
