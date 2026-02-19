import sys
import tensorflow as tf
import numpy as np
import chess
import chess.pgn
import random

# Ensure compatibility (TensorFlow requires Python < 3.14)
assert sys.version_info < (3, 14), "Please use Python 3.13 or older."

# ============================================================
# 1. CONSTANTS & ELO BUCKETING
# ============================================================
TARGET_ELOS = [1200, 1400, 1600, 1800, 2000]
NUM_ELOS = len(TARGET_ELOS)

def get_elo_class(elo):
    diffs = [abs(elo - r) for r in TARGET_ELOS]
    return diffs.index(min(diffs))

# ============================================================
# 2. BOARD & MOVE ENCODING
# ============================================================
def build_move_dict():
    moves = []
    for from_sq in chess.SQUARES:
        for to_sq in chess.SQUARES:
            moves.append(chess.Move(from_sq, to_sq).uci())
    move_to_idx = {m: i for i, m in enumerate(moves)}
    return move_to_idx

move_to_idx = build_move_dict()
NUM_MOVES = len(move_to_idx)

def board_to_tensor(board):
    """Converts a chess.Board into an 8x8x12 spatial tensor."""
    tensor = np.zeros((8, 8, 12), dtype=np.float32)
    for square, piece in board.piece_map().items():
        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)
        piece_type = piece.piece_type - 1
        color_offset = 0 if piece.color == chess.WHITE else 6
        tensor[row, col, piece_type + color_offset] = 1
    return tensor

# ============================================================
# 3. STANDALONE DATA LOADER (CREATES REAL & FAKE DATA)
# ============================================================
def load_and_simulate_data(pgn_path, max_games=1000):
    """
    Reads the Lichess PGN and creates a balanced dataset:
    - Label 1: The actual move played by the human.
    - Label 0: A random legal move simulating a bad AI generator.
    """
    boards, moves, elos, labels = [], [], [], []
    
    with open(pgn_path) as pgn:
        for _ in range(max_games):
            game = chess.pgn.read_game(pgn)
            if game is None: break
                
            try:
                w_elo_str = game.headers.get("WhiteElo", "0")
                b_elo_str = game.headers.get("BlackElo", "0")
                white_elo = int(w_elo_str) if w_elo_str.isdigit() else 0
                black_elo = int(b_elo_str) if b_elo_str.isdigit() else 0
            except ValueError:
                continue
            
            if white_elo < 1000 or black_elo < 1000:
                continue

            board = game.board()
            for move in game.mainline_moves():
                current_elo = white_elo if board.turn == chess.WHITE else black_elo
                elo_class = get_elo_class(current_elo)
                
                b_tensor = board_to_tensor(board)
                
                # --- 1. RECORD THE REAL MOVE (Label = 1) ---
                if move.uci() in move_to_idx:
                    boards.append(b_tensor)
                    moves.append(move_to_idx[move.uci()])
                    elos.append(elo_class)
                    labels.append(1.0)
                
                # --- 2. GENERATE A FAKE MOVE (Label = 0) ---
                legal_moves = list(board.legal_moves)
                if len(legal_moves) > 1:
                    fake_move = random.choice(legal_moves)
                    while fake_move == move:
                        fake_move = random.choice(legal_moves)
                        
                    if fake_move.uci() in move_to_idx:
                        boards.append(b_tensor)
                        moves.append(move_to_idx[fake_move.uci()])
                        elos.append(elo_class)
                        labels.append(0.0)
                
                board.push(move)
                
    return np.array(boards), np.array(moves), np.array(elos), np.array(labels)

# ============================================================
# 4. CONDITIONAL DISCRIMINATOR ARCHITECTURE
# ============================================================
def build_cgan_discriminator():
    board_input = tf.keras.Input(shape=(8, 8, 12), name="board")
    move_input = tf.keras.Input(shape=(NUM_MOVES,), name="move_onehot")
    elo_input = tf.keras.Input(shape=(1,), name="elo_class")
    
    elo_embedding = tf.keras.layers.Embedding(input_dim=NUM_ELOS, output_dim=12)(elo_input)
    x_elo = tf.keras.layers.Flatten()(elo_embedding)
    x_elo = tf.keras.layers.Dense(8 * 8)(x_elo)
    x_elo = tf.keras.layers.Reshape((8, 8, 1))(x_elo)
    
    combined_board = tf.keras.layers.Concatenate(axis=-1)([board_input, x_elo])
    
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(combined_board)
    x = tf.keras.layers.MaxPooling2D(2)(x) 
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    
    combined_all = tf.keras.layers.Concatenate()([x, move_input])
    
    combined_all = tf.keras.layers.Dense(256, activation="relu")(combined_all)
    combined_all = tf.keras.layers.Dropout(0.3)(combined_all) 
    
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="real_or_fake")(combined_all)
    
    return tf.keras.Model(inputs=[board_input, move_input, elo_input], outputs=output)

# ============================================================
# 5. EXECUTION, SAVING, & SANITY CHECK
# ============================================================
if __name__ == "__main__":
    print("Parsing Lichess PGN and generating Fake vs Real dataset...")
    pgn_file = "lichess_db_standard_rated_2013-01.pgn" 
    
    boards, moves, elos, labels = load_and_simulate_data(pgn_file, max_games=500)
    
    print(f"Dataset generated! Total samples: {len(boards)} (50% Real, 50% Fake)")
    
    moves_onehot = tf.one_hot(moves, NUM_MOVES)
    
    discriminator = build_cgan_discriminator()
    discriminator.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    print("\n--- TRAINING ---")
    discriminator.fit(
        x={"board": boards, "move_onehot": moves_onehot, "elo_class": elos},
        y=labels,
        batch_size=64,
        epochs=30,
        validation_split=0.2 
    )
    
    print("\n--- SAVING MODEL ---")
    model_filename = "cgan_discriminator_v2_epoch=30.keras"
    discriminator.save(model_filename)
    print(f"Model successfully saved to '{model_filename}'!")
"""
    # NEW: Run a manual sanity check
    print("\n--- SANITY CHECK (MANUAL INFERENCE) ---")
    
    # We grab 5 random samples from our dataset to test
    sample_boards = boards[:5]
    sample_moves = moves_onehot[:5]
    sample_elos = elos[:5]
    actual_labels = labels[:5]

    # We pass these samples to the trained model using the .predict() method
    predictions = discriminator.predict({
        "board": sample_boards, 
        "move_onehot": sample_moves, 
        "elo_class": sample_elos
    })

    print("\nResults:")
    for i in range(5):
        # 1.0 means it was actually a human move, 0.0 means it was our fake random move
        actual_status = "REAL" if actual_labels[i] == 1.0 else "FAKE"
        
        # The model outputs a probability between 0 and 1. We translate that to English:
        model_guess = "REAL" if predictions[i][0] >= 0.5 else "FAKE"
        
        print(f"Sample {i+1} (Actually {actual_status}): Model predicted {predictions[i][0]:.4f} -> Guessed {model_guess}")
"""