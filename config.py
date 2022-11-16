# Quick and dirty solution; probably doesn't scale.
# TODO Command line arguments or something.
board_width = 6
board_height = 6
piece_width_min = 2
piece_width_max = 5
piece_height_min = 1
piece_height_max = 5
pieces_per_puzzle = 14
training_set_path = "./dev_set.pt"
cycle_length = 2
device = "cuda:0"
steps_per_update = 1000
