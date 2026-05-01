import sys
import json
import io
import random
import os
import tensorflow as tf
from tensorflow.keras import regularizers
import numpy as np
import chess
import chess.pgn

assert sys.version_info < (3, 14), "TensorFlow requires Python 3.13 or older."

SEQ_LENGTH = 30

# ==============================
# 1. MOVE DICTIONARY
# ==============================
# Builds a mapping from every possible chess move string → a unique integer.
# Neural networks only understand numbers, so every move must be converted.

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


# ==============================
# 2. BOARD ENCODING
# ==============================
# Converts a chess board into a (8, 8, 12) array of numbers.
# 12 channels = 6 piece types x 2 colors (white pieces in 0-5, black in 6-11).
# Each cell is 1.0 if that piece is on that square, 0.0 if not.

def board_to_tensor(board):
    tensor = np.zeros((8, 8, 12), dtype=np.float32)
    for square, piece in board.piece_map().items():
        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)
        piece_type = piece.piece_type - 1
        color_offset = 0 if piece.color == chess.WHITE else 6
        tensor[row, col, piece_type + color_offset] = 1
    return tensor


# ==============================
# 3. ENDGAME DETECTION
# ==============================
# Endgame chess is logical and forced — it doesn't reflect a player's style.
# We stop recording moves once EITHER condition below is met.

PIECE_VALUES = {
    chess.QUEEN:  9,
    chess.ROOK:   5,
    chess.BISHOP: 3,
    chess.KNIGHT: 3,
    chess.PAWN:   1,
}

MATERIAL_CUTOFF = 32
MOVE_CUTOFF     = 30

def count_material(board):
    total = 0
    for piece in board.piece_map().values():
        total += PIECE_VALUES.get(piece.piece_type, 0)
    return total

def is_endgame(board, player_move_number):
    if player_move_number >= MOVE_CUTOFF:
        return True
    if count_material(board) <= MATERIAL_CUTOFF:
        return True
    return False


# ==============================
# 4. DATA LOADING
# ==============================
# Reads a JSON game file, replays each game move by move, and records
# board snapshots + move choices for the target player only.
# Stops recording once endgame is detected.
#
# max_games lets the caller cap how many games are loaded.
# We use this to enforce equal game counts between the two players —
# load all games first, find the smaller count, then trim both down.

def load_json_game_sequences(json_path, player_name, label_value, max_games=99999):
    game_boards_master = []
    game_moves_master  = []

    with open(json_path, 'r') as f:
        games = json.load(f)

    for i, game_data in enumerate(games):
        if i >= max_games:
            break

        is_player_white = (game_data.get("white") == player_name)

        pgn_io = io.StringIO(game_data["moves"])
        game   = chess.pgn.read_game(pgn_io)
        if game is None:
            continue

        board               = game.board()
        current_game_boards = []
        current_game_moves  = []
        player_move_number  = 0

        for move in game.mainline_moves():
            is_white_turn  = (board.turn == chess.WHITE)
            is_player_turn = (is_white_turn and is_player_white) or \
                             (not is_white_turn and not is_player_white)

            if is_player_turn:
                player_move_number += 1
                if is_endgame(board, player_move_number):
                    break
                if move.uci() in move_to_idx:
                    current_game_boards.append(board_to_tensor(board))
                    current_game_moves.append(move_to_idx[move.uci()])

            board.push(move)

        if len(current_game_boards) < 5:
            continue

        current_game_boards = current_game_boards[:SEQ_LENGTH]
        current_game_moves  = current_game_moves[:SEQ_LENGTH]

        while len(current_game_boards) < SEQ_LENGTH:
            current_game_boards.append(np.zeros((8, 8, 12), dtype=np.float32))
            current_game_moves.append(0)

        game_boards_master.append(current_game_boards)
        game_moves_master.append(current_game_moves)

    labels = [label_value] * len(game_boards_master)
    return game_boards_master, game_moves_master, labels


# ==============================
# 5. PER-GAME SIMILARITY SCORE
# ==============================
# Scores a single game dictionary from the JSON file.
# Returns a float between 0 and 1:
#   1.0 = this game strongly resembles Player A
#   0.0 = this game strongly resembles Player B

def predict_similarity(game_data, player_name, model_path="best_by_game_discriminator.keras", model=None):
    if model is None:
        model = tf.keras.models.load_model(model_path)

    is_player_white = (game_data.get("white") == player_name)

    pgn_io = io.StringIO(game_data["moves"])
    game   = chess.pgn.read_game(pgn_io)
    if game is None:
        return None

    board              = game.board()
    game_boards        = []
    game_moves         = []
    player_move_number = 0

    for move in game.mainline_moves():
        is_white_turn  = (board.turn == chess.WHITE)
        is_player_turn = (is_white_turn and is_player_white) or \
                         (not is_white_turn and not is_player_white)

        if is_player_turn:
            player_move_number += 1
            if is_endgame(board, player_move_number):
                break
            if move.uci() in move_to_idx:
                game_boards.append(board_to_tensor(board))
                game_moves.append(move_to_idx[move.uci()])

        board.push(move)

    if len(game_boards) < 5:
        return None

    game_boards = game_boards[:SEQ_LENGTH]
    game_moves  = game_moves[:SEQ_LENGTH]

    while len(game_boards) < SEQ_LENGTH:
        game_boards.append(np.zeros((8, 8, 12), dtype=np.float32))
        game_moves.append(0)

    board_input = np.expand_dims(np.array(game_boards), axis=0)
    move_input  = tf.one_hot(np.expand_dims(np.array(game_moves), axis=0), NUM_MOVES)

    score = float(model.predict(
        {"board_seq": board_input, "move_seq": move_input},
        verbose=0
    )[0][0])

    return score


# ==============================
# 5b. OVERALL PLAYSTYLE SIMILARITY
# ==============================
# Scores every game from both players and computes ONE overall similarity
# score representing how alike the two players' styles are across all games.
#
# HOW IT WORKS:
#   The model outputs 1.0 for Player A and 0.0 for Player B.
#   If styles are DIFFERENT: Player A's games average near 1.0,
#   Player B's games average near 0.0 — large gap between them.
#   If styles are SIMILAR: both players' games average near 0.5
#   because the model can't tell them apart — small gap.
#
#   Separation = avg score of A - avg score of B
#   Similarity = (1.0 - separation) x 100%
#
#   100% = indistinguishable styles
#     0% = completely different styles

def compute_overall_similarity(json_A, json_B, player_A, player_B, model):
    scores_A = []
    scores_B = []

    with open(json_A, 'r') as f:
        games_A = json.load(f)

    for game_data in games_A:
        score = predict_similarity(game_data, player_A, model=model)
        if score is not None:
            scores_A.append(score)

    with open(json_B, 'r') as f:
        games_B = json.load(f)

    for game_data in games_B:
        score = predict_similarity(game_data, player_B, model=model)
        if score is not None:
            scores_B.append(score)

    if not scores_A or not scores_B:
        return None

    avg_A      = np.mean(scores_A)
    avg_B      = np.mean(scores_B)
    separation = avg_A - avg_B

    # Return just the float — no printing, no dictionary.
    # 100.0 = indistinguishable styles, 0.0 = completely different.
    return (1.0 - separation) * 100.0


# ==============================
# 6. MODEL ARCHITECTURE
# ==============================

def build_style_classifier():
    board_seq_input = tf.keras.Input(shape=(SEQ_LENGTH, 8, 8, 12), name="board_seq")
    move_seq_input  = tf.keras.Input(shape=(SEQ_LENGTH, NUM_MOVES),  name="move_seq")

    dense_base = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(8, 8, 12)),
        tf.keras.layers.Dense(512, activation="relu",
                              kernel_regularizer=regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(512, activation="relu",
                              kernel_regularizer=regularizers.l2(0.001))
    ])
    encoded_boards = tf.keras.layers.TimeDistributed(dense_base)(board_seq_input)
    combined_seq   = tf.keras.layers.Concatenate(axis=-1)([encoded_boards, move_seq_input])
    masked_seq     = tf.keras.layers.Masking(mask_value=0.0)(combined_seq)

    x = tf.keras.layers.LSTM(256, return_sequences=True)(masked_seq)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.LSTM(128)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)

    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs=[board_seq_input, move_seq_input], outputs=output)


# ==============================
# 7. RESULTS PRINTER
# ==============================
# Shared function that prints the final similarity report.
# Called both after training and in standalone simScore mode
# so the output format is always identical.

def print_similarity_report(result, player_A, player_B):
    print(f"\n{'=' * 48}")
    print(f"  OVERALL PLAYSTYLE SIMILARITY")
    print(f"  {player_A}  vs  {player_B}")
    print(f"{'=' * 48}")
    print(f"  Games scored ({player_A}):".ljust(36) + f"{result['games_scored_A']}")
    print(f"  Games scored ({player_B}):".ljust(36) + f"{result['games_scored_B']}")
    print(f"  Avg model score ({player_A}):".ljust(36) + f"{result['avg_score_A']:.4f}")
    print(f"  Avg model score ({player_B}):".ljust(36) + f"{result['avg_score_B']:.4f}")
    print(f"  Style separation:".ljust(36) +            f"{result['separation']:.4f}")
    print(f"{'-' * 48}")
    print(f"  Overall playstyle similarity:".ljust(36) + f"{result['similarity_%']:.2f}%")
    print(f"{'=' * 48}\n")


# ============================================================
# 8. MAIN
# ============================================================
# Usage:
#   python discrim_v3_overall.py train    <player_A> <player_B>
#   python discrim_v3_overall.py simScore <player_A> <player_B>
#
# train    -> trains the discriminator on equal game counts from both
#             players, saves the best model, then AUTOMATICALLY runs
#             simScore and prints the overall similarity report.
#
# simScore -> loads the already-saved model and prints the overall
#             similarity report without retraining.
# ============================================================
if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("Usage:   python discrim_v3_overall.py <mode> <player_A> <player_B>")
        print("Modes:   train    — train the discriminator on two players")
        print("         simScore — compute overall playstyle similarity")
        print("Example: python discrim_v3_overall.py train Mouna007 Pancugolo")
        print("Example: python discrim_v3_overall.py simScore Mouna007 Pancugolo")
        sys.exit(1)

    mode     = sys.argv[1]
    player_A = sys.argv[2]
    player_B = sys.argv[3]
    json_A   = f"{player_A}_games.json"
    json_B   = f"{player_B}_games.json"

    if mode not in ("train", "simScore"):
        print(f"Error: unknown mode '{mode}'. Use 'train' or 'simScore'.")
        sys.exit(1)

    model_path = "best_by_game_discriminator.keras"

    if mode == "train":
        print(f"\n--- TRAIN MODE: {player_A} vs {player_B} ---\n")

        print(f"Loading all games for Player A ({player_A})...")
        b_A, m_A, l_A = load_json_game_sequences(
            json_A, player_name=player_A, label_value=1.0
        )

        print(f"Loading all games for Player B ({player_B})...")
        b_B, m_B, l_B = load_json_game_sequences(
            json_B, player_name=player_B, label_value=0.0
        )

        min_games = min(len(l_A), len(l_B))
        b_A, m_A, l_A = b_A[:min_games], m_A[:min_games], l_A[:min_games]
        b_B, m_B, l_B = b_B[:min_games], m_B[:min_games], l_B[:min_games]

        print(f"\nPlayer A ({player_A}) usable games: {len(l_A)}")
        print(f"Player B ({player_B}) usable games: {len(l_B)}")
        print(f"Training on {min_games} games per player ({min_games * 2} total)\n")

        raw_boards = b_A + b_B
        raw_moves  = m_A + m_B
        raw_labels = l_A + l_B

        combined = list(zip(raw_boards, raw_moves, raw_labels))
        random.shuffle(combined)
        raw_boards, raw_moves, raw_labels = zip(*combined)

        all_boards = np.array(raw_boards)
        all_moves  = np.array(raw_moves)
        all_labels = np.array(raw_labels)

        print(f"Dataset ready — {len(all_labels)} total games\n")

        all_moves_onehot = tf.one_hot(all_moves, NUM_MOVES)

        classifier = build_style_classifier()
        classifier.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        print("--- TRAINING SEQUENCE CLASSIFIER ---\n")

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True, verbose=1
        )
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            model_path, monitor="val_loss", save_best_only=True, verbose=1
        )

        classifier.fit(
            x={"board_seq": all_boards, "move_seq": all_moves_onehot},
            y=all_labels,
            batch_size=32,
            epochs=20,
            validation_split=0.2,
            callbacks=[early_stop, checkpoint]
        )

        print(f"\nTraining complete. Best weights saved to '{model_path}'.")

        # ---------------------------------------------------------------
        # AUTO SIMSCORE — runs immediately after training completes.
        # Reloads the best saved weights before scoring.
        # ---------------------------------------------------------------
        print(f"\nAuto-running similarity score on saved best model...\n")
        best_model = tf.keras.models.load_model(model_path)
        similarity = compute_overall_similarity(json_A, json_B, player_A, player_B, best_model)

        if similarity is not None:
            print(f"Overall playstyle similarity: {similarity:.2f}%")
        else:
            print("Could not compute similarity — not enough scoreable games.")

    elif mode == "simScore":
        print(f"\n--- SIMSCORE MODE: {player_A} vs {player_B} ---\n")

        if not os.path.exists(model_path):
            print(f"Error: no trained model found at '{model_path}'.")
            print(f"Run 'python discrim_v3_overall.py train {player_A} {player_B}' first.")
            sys.exit(1)

        print(f"Loading model from '{model_path}'...\n")
        model      = tf.keras.models.load_model(model_path)
        similarity = compute_overall_similarity(json_A, json_B, player_A, player_B, model)

        if similarity is not None:
            print(f"Overall playstyle similarity: {similarity:.2f}%")
        else:
            print("Could not compute similarity — not enough scoreable games.")
