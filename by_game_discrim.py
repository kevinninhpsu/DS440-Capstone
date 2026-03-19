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

# 2. JSON DATA LOADER
def load_json_player_games(json_path, player_name, label_value, max_games=500):
    """
    Returns a list of per-game dictionaries.
    Each item contains all move samples from one game.
    """
    game_samples = []

    with open(json_path, "r") as f:
        games = json.load(f)

    for i, game_data in enumerate(games):
        if i >= max_games:
            break

        is_player_white = (game_data.get("white") == player_name)

        moves_text = game_data.get("moves") or game_data.get("moves_san")
        if not moves_text:
            continue

        pgn_io = io.StringIO(f'[Event "?"]\n\n{moves_text}\n')
        game = chess.pgn.read_game(pgn_io)

        if game is None:
            continue

        board = game.board()
        boards, moves = [], []

        for move in game.mainline_moves():
            is_white_turn = (board.turn == chess.WHITE)

            if (is_white_turn and is_player_white) or (not is_white_turn and not is_player_white):
                if move.uci() in move_to_idx:
                    boards.append(board_to_tensor(board))
                    moves.append(move_to_idx[move.uci()])

            board.push(move)

        if len(boards) > 0:
            game_samples.append({
                "boards": boards,
                "moves": moves,
                "labels": [label_value] * len(boards)
            })

    return game_samples

# 3. SPLIT LIST OF GAMES (chronological split)
def split_games(game_list, train_ratio=0.75):
    split_idx = int(len(game_list) * train_ratio)
    train_games = game_list[:split_idx]
    val_games = game_list[split_idx:]
    return train_games, val_games

# 4. FLATTEN GAME SAMPLES
def flatten_games(game_list):
    boards, moves, labels = [], [], []

    for game in game_list:
        boards.extend(game["boards"])
        moves.extend(game["moves"])
        labels.extend(game["labels"])

    return np.array(boards), np.array(moves), np.array(labels)

# 5. STYLE CLASSIFIER ARCHITECTURE
def build_style_classifier():
    reg = tf.keras.regularizers.l2(1e-4)
    board_input = tf.keras.Input(shape=(8, 8, 12), name="board")
    move_input = tf.keras.Input(shape=(NUM_MOVES,), name="move_onehot")

    x = tf.keras.layers.Conv2D(64, 3, padding="same",
        activation="relu",
        kernel_regularizer=reg)(board_input)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same",
        activation="relu",
        kernel_regularizer=reg)(x)
    x = tf.keras.layers.Flatten()(x)

    combined = tf.keras.layers.Concatenate()([x, move_input])
    combined = tf.keras.layers.Dense(256, activation="relu")(combined)
    combined = tf.keras.layers.Dropout(0.4)(combined)

    output = tf.keras.layers.Dense(1, activation="sigmoid", name="player_a_prob")(combined)

    return tf.keras.Model(inputs=[board_input, move_input], outputs=output)

# 6. JSON GAME EVALUATION FUNCTION
def predict_json_game_author(model, game_data, target_player_color):
    """
    Evaluates an entire JSON game and returns the average probability
    that the moves made by the specified color belong to Player A.
    """
    boards, moves = [], []

    moves_text = game_data.get("moves") or game_data.get("moves_san")
    if not moves_text:
        return 0.5

    pgn_io = io.StringIO(f'[Event "?"]\n\n{moves_text}\n')
    game = chess.pgn.read_game(pgn_io)
    if game is None:
        return 0.5

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

# 7. Training
if __name__ == "__main__":
    print("Loading Player A (mrkeshavarz2025) as Label 1...")
    games_A = load_json_player_games(
        "Bijay_1549_games.json",
        player_name="Bijay_1549",
        label_value=1.0,
        max_games=443
    )

    print("Loading Player B (asghar7arab) as Label 0...")
    games_B = load_json_player_games(
        "Mouna007_games.json",
        player_name="Mouna007",
        label_value=0.0,
        max_games=443
    )

    print("Player A games loaded:", len(games_A))
    print("Player B games loaded:", len(games_B))

    train_A, val_A = split_games(games_A, train_ratio=0.75)
    train_B, val_B = split_games(games_B, train_ratio=0.75)

    train_games = train_A + train_B
    val_games = val_A + val_B

    rng = np.random.default_rng(42)
    rng.shuffle(train_games)
    rng.shuffle(val_games)

    x_train_boards, x_train_moves, y_train = flatten_games(train_games)
    x_val_boards, x_val_moves, y_val = flatten_games(val_games)

    print(f"\nTrain games: {len(train_games)}")
    print(f"Validation games: {len(val_games)}")
    print(f"Train moves: {len(y_train)}")
    print(f"Validation moves: {len(y_val)}\n")

    x_train_moves_onehot = tf.one_hot(x_train_moves, NUM_MOVES)
    x_val_moves_onehot = tf.one_hot(x_val_moves, NUM_MOVES)

    classifier = build_style_classifier()
    classifier.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    print("--- TRAINING CLASSIFIER ---")

    # Use early stopping to prevent overfitting and save the best model based on validation loss, model stops when val_loss stops improving

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "best_by_game_discriminator.keras",
        monitor="val_loss",
        save_best_only=True
    )

    classifier.fit(
        x={"board": x_train_boards, "move_onehot": x_train_moves_onehot},
        y=y_train,
        batch_size=64,
        epochs=15,
        validation_data=(
            {"board": x_val_boards, "move_onehot": x_val_moves_onehot},
            y_val
        ),
        callbacks=[early_stop, checkpoint]
    )


