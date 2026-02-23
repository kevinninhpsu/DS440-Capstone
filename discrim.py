import sys
import json
import io
import tensorflow as tf
import numpy as np
import chess
import chess.pgn

# Ensure compatibility
assert sys.version_info < (3, 14), "TensorFlow requires Python 3.13 or older."

# 1. BOARD & MOVE ENCODING
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

# 2. JSON DATA LOADER
def load_json_player_data(json_path, player_name, label_value, max_games=500):
    """
    Reads the JSON array, determines if the target player is White or Black 
    for each game, and extracts only their moves.
    """
    boards, moves = [], []
    
    with open(json_path, 'r') as f:
        games = json.load(f)
        
    for i, game_data in enumerate(games):
        if i >= max_games: break
            
        # 1. Determine which color the target player is controlling
        is_player_white = (game_data.get("white") == player_name)
        
        # 2. Trick the PGN reader into parsing the raw string of moves
        pgn_io = io.StringIO(game_data["moves"])
        game = chess.pgn.read_game(pgn_io)
        
        if game is None: continue
            
        board = game.board()
        for move in game.mainline_moves():
            # 3. Check if it is currently our target player's turn
            is_white_turn = (board.turn == chess.WHITE)
            
            if (is_white_turn and is_player_white) or (not is_white_turn and not is_player_white):
                if move.uci() in move_to_idx:
                    boards.append(board_to_tensor(board))
                    moves.append(move_to_idx[move.uci()])
                    
            board.push(move)
            
    labels = [label_value] * len(boards)
    return boards, moves, labels

# 3. STYLE CLASSIFIER ARCHITECTURE
def build_style_classifier():
    board_input = tf.keras.Input(shape=(8, 8, 12), name="board")
    move_input = tf.keras.Input(shape=(NUM_MOVES,), name="move_onehot")
    
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(board_input)
    x = tf.keras.layers.MaxPooling2D(2)(x) 
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    
    combined = tf.keras.layers.Concatenate()([x, move_input])
    
    combined = tf.keras.layers.Dense(256, activation="relu")(combined)
    combined = tf.keras.layers.Dropout(0.4)(combined)
    
    # Output: 1 = Player A (mrkeshavarz2025), 0 = Player B (asghar7arab)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="player_a_prob")(combined)
    
    return tf.keras.Model(inputs=[board_input, move_input], outputs=output)

# 4. JSON GAME EVALUATION FUNCTION
def predict_json_game_author(model, game_data, target_player_color):
    """
    Evaluates an entire JSON game and returns the average probability 
    that the moves made by the specified color belong to Player A.
    """
    boards, moves = [], []
    
    pgn_io = io.StringIO(game_data["moves"])
    game = chess.pgn.read_game(pgn_io)
    board = game.board()
    
    for move in game.mainline_moves():
        if board.turn == target_player_color:
            if move.uci() in move_to_idx:
                boards.append(board_to_tensor(board))
                moves.append(move_to_idx[move.uci()])
        board.push(move)
        
    if len(boards) == 0:
        return 0.5 
        
    b_array = np.array(boards)
    m_onehot = tf.one_hot(np.array(moves), NUM_MOVES)
    
    predictions = model.predict({"board": b_array, "move_onehot": m_onehot}, verbose=0)
    return np.mean(predictions)

# 5. Training
# 443 is max games since asghar7arab has only 443 games
if __name__ == "__main__":
    print("Loading Player A (mrkeshavarz2025) as Label 1...")
    b_A, m_A, l_A = load_json_player_data(
        "mrkeshavarz2025_games.json", 
        player_name="mrkeshavarz2025", 
        label_value=1.0, 
        max_games=443
    )
    
    print("Loading Player B (asghar7arab) as Label 0...")
    b_B, m_B, l_B = load_json_player_data(
        "asghar7arab_games.json", 
        player_name="asghar7arab", 
        label_value=0.0, 
        max_games=443
    )
    
    all_boards = np.array(b_A + b_B)
    all_moves = np.array(m_A + m_B)
    all_labels = np.array(l_A + l_B)
    
    print(f"\nDataset Ready! Total moves to learn from: {len(all_labels)}")
    print(f"mrkeshavarz2025 moves: {len(l_A)}")
    print(f"asghar7arab moves: {len(l_B)}\n")
    
    all_moves_onehot = tf.one_hot(all_moves, NUM_MOVES)
    
    classifier = build_style_classifier()
    classifier.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    print("--- TRAINING CLASSIFIER ---")
    classifier.fit(
        x={"board": all_boards, "move_onehot": all_moves_onehot},
        y=all_labels,
        batch_size=64,
        epochs=15,
        validation_split=0.2 
    )
    
