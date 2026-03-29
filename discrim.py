"""
Chess Authorship Attribution - Sequence Model (Capstone Final)
--------------------------------------------------------------
This model learns the long-term strategic style of two players by evaluating 
the first 40 moves of their games using a TimeDistributed Dense network 
feeding into stacked LSTMs.
"""

import sys
import json
import io
import random
import tensorflow as tf
from tensorflow.keras import regularizers
import numpy as np
import chess
import chess.pgn

# Ensure compatibility
assert sys.version_info < (3, 14), "TensorFlow requires Python 3.13 or older."

SEQ_LENGTH = 40 

# ============================================================
# 1. BOARD & MOVE ENCODING
# ============================================================
def build_move_dict():
    moves = []
    promotion_pieces = [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]

    for from_sq in chess.SQUARES:
        for to_sq in chess.SQUARES:
            for promo in promotion_pieces:
                try:
                    move = chess.Move(from_sq, to_sq, promotion=promo)
                    moves.append(move.uci())
                except Exception:
                    pass

    moves = sorted(set(moves))
    return {m: i for i, m in enumerate(moves)}

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
# 2. JSON DATA LOADER (SEQUENCES)
# ============================================================
def load_json_game_sequences(json_path, player_name, label_value, max_games=500):
    """
    Extracts games as sequences of up to SEQ_LENGTH moves. 
    Returns: Shape (num_games, SEQ_LENGTH, 8, 8, 12) for boards
             Shape (num_games, SEQ_LENGTH) for moves
    """
    game_boards_master = []
    game_moves_master = []
    
    with open(json_path, 'r') as f:
        games = json.load(f)
        
    for i, game_data in enumerate(games):
        if i >= max_games: break
            
        is_player_white = (game_data.get("white") == player_name)
        pgn_io = io.StringIO(game_data["moves"])
        game = chess.pgn.read_game(pgn_io)
        
        if game is None: continue
            
        board = game.board()
        current_game_boards = []
        current_game_moves = []
        
        for move in game.mainline_moves():
            is_white_turn = (board.turn == chess.WHITE)
            
            if (is_white_turn and is_player_white) or (not is_white_turn and not is_player_white):
                if move.uci() in move_to_idx:
                    current_game_boards.append(board_to_tensor(board))
                    current_game_moves.append(move_to_idx[move.uci()])
                    
            board.push(move)
            
        # 1. Truncate if the game is longer than our SEQ_LENGTH
        current_game_boards = current_game_boards[:SEQ_LENGTH]
        current_game_moves = current_game_moves[:SEQ_LENGTH]
        
        # 2. Pad with zeros if the game is shorter than SEQ_LENGTH
        while len(current_game_boards) < SEQ_LENGTH:
            current_game_boards.append(np.zeros((8, 8, 12), dtype=np.float32))
            current_game_moves.append(0) 
            
        game_boards_master.append(current_game_boards)
        game_moves_master.append(current_game_moves)
            
    labels = [label_value] * len(game_boards_master)
    return game_boards_master, game_moves_master, labels

# ============================================================
# 3. SEQUENCE CLASSIFIER ARCHITECTURE (The "Hourglass")
# ============================================================
def build_style_classifier():
    board_seq_input = tf.keras.Input(shape=(SEQ_LENGTH, 8, 8, 12), name="board_seq")
    move_seq_input = tf.keras.Input(shape=(SEQ_LENGTH, NUM_MOVES), name="move_seq")
    
    # --- GLOBAL BOARD EXTRACTOR ---
    # 512-width layers with L2 Regularization to prevent overfitting
    dense_base = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(8, 8, 12)),
        tf.keras.layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.001))
    ], name="global_board_extractor")
    
    # Process the sequence of boards
    encoded_boards = tf.keras.layers.TimeDistributed(dense_base)(board_seq_input)
    
    # Combine the processed board state with the actual move chosen
    combined_seq = tf.keras.layers.Concatenate(axis=-1)([encoded_boards, move_seq_input])
    
    # --- LONG-TERM MEMORY (LSTM) ---
    # Stepping down the node count to avoid exploding parameters
    x = tf.keras.layers.LSTM(256, return_sequences=True)(combined_seq)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.LSTM(128)(x) 
    
    # --- FINAL DECISION BLOCK ---
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    
    # Final Output: 1 = Player A, 0 = Player B
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="player_a_prob")(x)
    
    return tf.keras.Model(inputs=[board_seq_input, move_seq_input], outputs=output)

# ============================================================
# 4. TRAINING EXECUTION
# ============================================================
if __name__ == "__main__":
    print(f"Loading Player A (mrkeshavarz2025) games as sequences of {SEQ_LENGTH} moves...")
    b_A, m_A, l_A = load_json_game_sequences(
        "mrkeshavarz2025_games.json", 
        player_name="mrkeshavarz2025", 
        label_value=1.0, 
        max_games=443
    )
    
    print(f"Loading Player B (asghar7arab) games as sequences of {SEQ_LENGTH} moves...")
    b_B, m_B, l_B = load_json_game_sequences(
        "asghar7arab_games.json", 
        player_name="asghar7arab", 
        label_value=0.0, 
        max_games=443
    )
    
    # Combine datasets
    raw_boards = b_A + b_B
    raw_moves = m_A + m_B
    raw_labels = l_A + l_B
    
    # SHUFFLE THE DATA
    # This prevents validation_split from testing exclusively on Player B
    combined = list(zip(raw_boards, raw_moves, raw_labels))
    random.shuffle(combined)
    raw_boards, raw_moves, raw_labels = zip(*combined)

    all_boards = np.array(raw_boards)
    all_moves = np.array(raw_moves)
    all_labels = np.array(raw_labels)
    
    print(f"\nDataset Ready! Total GAMES to learn from: {len(all_labels)}")
    
    # One-hot encode the sequence of moves
    all_moves_onehot = tf.one_hot(all_moves, NUM_MOVES)
    
    classifier = build_style_classifier()
    classifier.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    print("\n--- TRAINING SEQUENCE CLASSIFIER ---")
    
    # Early Stopping: Halts training if validation loss rises for 4 consecutive epochs
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    )

    # Checkpoint: Saves the model at its mathematical peak
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "best_by_game_discriminator.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    classifier.fit(
        x={"board_seq": all_boards, "move_seq": all_moves_onehot},
        y=all_labels,
        batch_size=32, 
        epochs=30, # High epoch count; early_stop will safely cut it off
        validation_split=0.2,
        callbacks=[early_stop, checkpoint]
    )
    
    print("\nTraining Complete. Best weights saved to 'best_by_game_discriminator.keras'.")