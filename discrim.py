import sys  
import json 
import io  
import random  
import tensorflow as tf  
from tensorflow.keras import regularizers  
import numpy as np  
import chess  # chess logic
import chess.pgn  # reads PGN

# Ensure TensorFlow works with current Python version
assert sys.version_info < (3, 14), "TensorFlow requires Python 3.13 or older."

# Maximum number of moves per game (sequence length)
SEQ_LENGTH = 40

# ==============================
# 1. MOVE DICTIONARY
# ==============================
# This builds a mapping from every possible move to a number

def build_move_dict():
    moves = []
    
    # Promotion pieces (pawn reaching end of board)
    promotion_pieces = [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]

    # Loop over ALL possible squares on the board
    for from_sq in chess.SQUARES:
        for to_sq in chess.SQUARES:
            for promo in promotion_pieces:
                try:
                    # Create a move (like e2e4)
                    move = chess.Move(from_sq, to_sq, promotion=promo)
                    moves.append(move.uci())  # convert to string format
                except Exception:
                    pass  # ignore invalid moves

    # Remove duplicates and sort
    moves = sorted(set(moves))
    
    # Map each move string → unique integer
    return {m: i for i, m in enumerate(moves)}

# Create dictionary
move_to_idx = build_move_dict()
NUM_MOVES = len(move_to_idx)  # total number of possible moves

# ==============================
# 2. BOARD ENCODING
# ==============================
# Converts a chess board into numbers
# Output shape: (8, 8, 12)
# Explanation:
# - 8x8 = chess board
# - 12 channels = 6 piece types × 2 colors

def board_to_tensor(board):
    tensor = np.zeros((8, 8, 12), dtype=np.float32)
    
    # piece_map gives all pieces currently on the board
    for square, piece in board.piece_map().items():
        
        # Convert square index → row/column
        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)
        
        # piece_type: pawn=1, knight=2, etc → shift to 0-based index
        piece_type = piece.piece_type - 1
        
        # Separate white and black pieces into different channels
        color_offset = 0 if piece.color == chess.WHITE else 6
        
        # Set that position to 1 (one-hot encoding)
        tensor[row, col, piece_type + color_offset] = 1

    return tensor

# ==============================
# 3. DATA LOADING
# ==============================
# Turns JSON games into sequences the model can learn from

def load_json_game_sequences(json_path, player_name, label_value, max_games=500):
    
    game_boards_master = []  # all board sequences
    game_moves_master = []   # all move sequences
    
    with open(json_path, 'r') as f:
        games = json.load(f)
        
    for i, game_data in enumerate(games):
        if i >= max_games:
            break
            
        # Check if player is white
        is_player_white = (game_data.get("white") == player_name)
        
        # Convert move string into readable format
        pgn_io = io.StringIO(game_data["moves"])
        game = chess.pgn.read_game(pgn_io)
        
        if game is None:
            continue
            
        board = game.board()
        current_game_boards = []
        current_game_moves = []
        
        # Go through every move in the game
        for move in game.mainline_moves():
            is_white_turn = (board.turn == chess.WHITE)
            
            # Only record moves made by the target player
            if (is_white_turn and is_player_white) or (not is_white_turn and not is_player_white):
                
                if move.uci() in move_to_idx:
                    current_game_boards.append(board_to_tensor(board))
                    current_game_moves.append(move_to_idx[move.uci()])
                    
            board.push(move)  # update board
            
        # Truncate to max length
        current_game_boards = current_game_boards[:SEQ_LENGTH]
        current_game_moves = current_game_moves[:SEQ_LENGTH]
        
        # Pad if too short
        while len(current_game_boards) < SEQ_LENGTH:
            current_game_boards.append(np.zeros((8, 8, 12), dtype=np.float32))
            current_game_moves.append(0)
            
        game_boards_master.append(current_game_boards)
        game_moves_master.append(current_game_moves)
        
    labels = [label_value] * len(game_boards_master)
    return game_boards_master, game_moves_master, labels

# ==============================
# 4. MODEL ARCHITECTURE
# ==============================

def build_style_classifier():
    
    # Inputs
    board_seq_input = tf.keras.Input(shape=(SEQ_LENGTH, 8, 8, 12), name="board_seq")
    move_seq_input = tf.keras.Input(shape=(SEQ_LENGTH, NUM_MOVES), name="move_seq")
    
    # ------------------------------
    # BOARD FEATURE EXTRACTOR
    # ------------------------------
    # A neural network that processes ONE board
    dense_base = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(8, 8, 12)),
        
        # Dense layer = fully connected layer
        # Every input connects to every output
        tf.keras.layers.Dense(512, activation="relu",
                              kernel_regularizer=regularizers.l2(0.001)),
        
        # Dropout = randomly turns off neurons to prevent overfitting
        tf.keras.layers.Dropout(0.4),
        
        tf.keras.layers.Dense(512, activation="relu",
                              kernel_regularizer=regularizers.l2(0.001))
    ])
    
    # Apply this to EACH timestep (each move)
    encoded_boards = tf.keras.layers.TimeDistributed(dense_base)(board_seq_input)
    
    # Combine board features + move information
    combined_seq = tf.keras.layers.Concatenate(axis=-1)([encoded_boards, move_seq_input])
    
    # ------------------------------
    # SEQUENCE MODEL (LSTM)
    # ------------------------------
    # LSTM = a type of neural network for sequences (like time series)
    # It remembers past information
    
    x = tf.keras.layers.LSTM(256, return_sequences=True)(combined_seq)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    # This LSTM compresses entire sequence → one vector
    x = tf.keras.layers.LSTM(128)(x)
    
    # ------------------------------
    # FINAL CLASSIFIER
    # ------------------------------
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    
    # Sigmoid = outputs number between 0 and 1 (probability)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    
    return tf.keras.Model(inputs=[board_seq_input, move_seq_input], outputs=output)


# ============================================================
# 4. TRAINING EXECUTION
# ============================================================
if __name__ == "__main__":
    print(f"Loading Player A (Pancugolo) games as sequences of {SEQ_LENGTH} moves...")
    b_A, m_A, l_A = load_json_game_sequences(
        "Pancugolo_games.json", 
        player_name="Pancugolo", 
        label_value=1.0, 
        max_games=318
    )
    
    print(f"Loading Player B (Mouna007) games as sequences of {SEQ_LENGTH} moves...")
    b_B, m_B, l_B = load_json_game_sequences(
        "Mouna007_games.json", 
        player_name="Mouna007", 
        label_value=0.0, 
        max_games=318
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
        epochs=20, # High epoch count; early_stop will safely cut it off
        validation_split=0.2,
        callbacks=[early_stop, checkpoint]
    )
    
    print("\nTraining Complete. Best weights saved to 'best_by_game_discriminator.keras'.")