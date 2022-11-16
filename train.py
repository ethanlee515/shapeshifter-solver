#!/usr/bin/env python

import torch
import torch.nn as nn
from config import *
from torch import Tensor

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
    puzzles = torch.load('training_set.pt', map_location=device)
    solver = ShapeshiftSolver()
    solver.to(device)
    optim = torch.optim.Adam(solver.parameters())
    for ctr, pieces in enumerate(puzzles):
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
