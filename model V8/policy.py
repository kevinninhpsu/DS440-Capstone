import chess.pgn
import json
import numpy as np
import os
import tensorflow as tf
from stockfish import Stockfish
import pickle
print('POLICY V8 (JOINT)')

symbol_map = {
    0:'P',1:'N',2:'B',3:'R',4:'Q',5:'K',
    6:'p',7:'n',8:'b',9:'r',10:'q',11:'k'
}

def load_pgn(pgn_file):
    games = []
    with open(pgn_file) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            games.append(game)
    return games

import re
from tqdm import tqdm
DIV_RE = re.compile(r"\s?(\d)+\.\s|\s")

def load_json(json_file, n=-1):
    with open(json_file, "r") as f:
        data = json.load(f)

    games = []
    for g in tqdm(data, desc="Loading games"):
        if n == 0:
            break
        n -= 1

        game = chess.pgn.Game()
        game.headers.update({
            "Event":  g.get("event", "?"),
            "White":  g.get("white", "?"),
            "Black":  g.get("black", "?"),
            "Result": g.get("result", "*")
        })

        moves = DIV_RE.split(g.get("moves", ""))
        moves = [m.strip() for m in moves if m and not m.strip().isdigit()]

        board = chess.Board()
        node = game
        for move in moves:
            try:
                parsed = board.push_san(move)
                node = node.add_variation(parsed)
            except Exception:
                break

        games.append(game)

    return games

def promo_encode(move):
    if move.promotion is None:
        return 0
    return {
        chess.QUEEN: 1,
        chess.ROOK: 2,
        chess.BISHOP: 3,
        chess.KNIGHT: 4
    }[move.promotion]

def encode_boards(board):
    tensor = np.zeros((8, 8, 12), dtype=np.float32)
    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8
        plane = piece.piece_type - 1
        if piece.color == chess.BLACK:
            plane += 6
        tensor[row, col, plane] = 1.0
    return tensor

def move_to_conf(move):
    from_row = move.from_square // 8
    from_col = move.from_square % 8
    to_row   = move.to_square // 8
    to_col   = move.to_square % 8

    if move.promotion:
        promo_offset = {
            chess.KNIGHT: 8,
            chess.BISHOP: 9,
            chess.ROOK:   10,
            chess.QUEEN:  11
        }[move.promotion]
        return from_row, from_col, to_row, promo_offset
    else:
        return from_row, from_col, to_row, to_col

def move_to_conf_matrix(move):
    matrix = np.zeros((8, 8, 8, 12), dtype=np.float32)
    matrix[move_to_conf(move)] = 1.0
    return matrix

class Agent:
    def __init__(self, id, stockfish_path=""):
        self.id = id

        sf = Stockfish(stockfish_path)
        sf.set_depth(4)
        sf.set_skill_level(4)
        self.sf = sf

        self.eval_cache = {}

    def cached_eval_state(self, board):
        fen = board.fen()
        if fen in self.eval_cache:
            return self.eval_cache[fen]

        self.sf.set_fen_position(fen)
        info = self.sf.get_evaluation()

        if info is None or "type" not in info or "value" not in info:
            self.eval_cache[fen] = 0
            print(fen)
            return 0

        if info["type"] == "mate":
            value = 500 if info["value"] > 0 else -500
        else:
            value = np.clip(info["value"], -500, 500)

        self.eval_cache[fen] = value
        return value

    def act(self, board):
        x = encode_boards(board)
        x = np.expand_dims(x, axis=0)

        move_logits = self.model.predict(x, verbose=0)[0]
        legal_moves = list(board.legal_moves)

        scores = np.array([move_logits[move_to_conf(m)] for m in legal_moves], dtype=np.float32)
        scores = scores - np.max(scores)
        probs = np.exp(scores)
        probs /= probs.sum()

        chosen_idx = np.random.choice(len(legal_moves), p=probs)
        return legal_moves[chosen_idx]

    def train(self, games, alpha=0.7, batch_size=32, epochs=20, save_path="./hybrid_dataset/data.pkl"):

        samples = self._build_hybrid_dataset(games, alpha, save_path=save_path)

        n = len(samples)

        def gen():
            while True:
                for i in range(0, n, batch_size):
                    batch = samples[i:i + batch_size]
                    X = np.array([b["board"] for b in batch], dtype=np.float32)
                    Y = np.array([b["policy"] for b in batch], dtype=np.float32)
                    yield X, Y

        model = self.build_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )

        steps = (n + batch_size - 1) // batch_size
        model.fit(gen(), steps_per_epoch=steps, epochs=epochs)

        self.model = model

    def build_model(self):
        inputs = tf.keras.layers.Input(shape=(8, 8, 12))  # 1 board x 12 planes
        x = tf.keras.layers.Reshape((64, 12))(inputs)

        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.LayerNormalization()(x)

        attn1 = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=32)(x, x)
        x = tf.keras.layers.Add()([x, attn1])
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dropout(0.1)(x)

        attn2 = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=32)(x, x)
        x = tf.keras.layers.Add()([x, attn2])
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dropout(0.1)(x)

        attn3 = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=32)(x, x)
        x = tf.keras.layers.Add()([x, attn3])
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dropout(0.1)(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(2048, activation='relu')(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dropout(0.1)(x)

        x = tf.keras.layers.Dense(8 * 8 * 8 * 12, activation='softmax')(x)
        outputs = tf.keras.layers.Reshape((8, 8, 8, 12), name="policy")(x)

        return tf.keras.Model(inputs, outputs)

    def _build_hybrid_dataset(self, games, alpha, save_path="./hybrid_dataset/data.pkl"):
        samples = []
        for game in games:
            board = game.board()
            for move in game.mainline_moves():
                if board.is_game_over():
                    break
                board_tensor = encode_boards(board)
                legal_moves = list(board.legal_moves)

                human_probs = np.zeros(len(legal_moves), dtype=np.float32)
                for j, m in enumerate(legal_moves):
                    if m.uci() == move.uci():
                        human_probs[j] = 1.0
                        break

                base_eval = self.cached_eval_state(board)
                top_moves = self.get_top_moves(board, top_k=5)
                sf_scores = np.full(len(legal_moves), -1e4, dtype=np.float32)
                legal_uci_map = {x.uci(): i for i, x in enumerate(legal_moves)}

                for m in top_moves:
                    if m.uci() in legal_uci_map:
                        idx = legal_uci_map[m.uci()]
                        board.push(m)
                        eval_after = self.cached_eval_state(board) if not board.is_game_over() else base_eval
                        board.pop()
                        sf_scores[idx] = eval_after - base_eval

                sf_probs = tf.convert_to_tensor(sf_scores, dtype=tf.float32)
                sf_probs_normalized = tf.nn.softmax(sf_probs).numpy()

                final_probs = alpha * human_probs + (1 - alpha) * sf_probs_normalized
                final_probs = final_probs / (final_probs.sum() + 1e-8)
                target = np.zeros((8, 8, 8, 12), dtype=np.float32)
                for m, p in zip(legal_moves, final_probs):
                    target[move_to_conf(m)] = p
                if target.sum() < 1e-6:
                    board.push(move)
                    continue
                samples.append({
                    "board": board_tensor,
                    "policy": target
                })
                board.push(move)


        return samples

    def get_top_moves(self, board, top_k):
        self.sf.set_fen_position(board.fen())
        top = self.sf.get_top_moves(top_k)
        return [chess.Move.from_uci(m["Move"]) for m in top]