import chess.pgn
import json
import numpy as np
import tensorflow as tf

print('POLICY V7')
symbol_map = {
    0:'P',1:'N',2:'B',3:'R',4:'Q',5:'K',   # white
    6:'p',7:'n',8:'b',9:'r',10:'q',11:'k'   # black
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

def load_json(json_file, n = -1):
    with open(json_file, "r") as f:
        data = json.load(f)
    
    games = []
    for g in tqdm(data, desc="Loading games"):

        if n == 0:
            break
        n -= 1
        game = chess.pgn.Game()
        
        # Batch header assignment
        game.headers.update({
            "Event":  g.get("event", "?"),
            "White":  g.get("white", "?"),
            "Black":  g.get("black", "?"),
            "Result": g.get("result", "*")
        })

        # Parse moves more efficiently
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

def pgn_to_player_samples(games, player_id=""):
    samples = []
    player_id = player_id.lower().strip()

    for game in games:
        white = game.headers.get("White", "").lower()
        black = game.headers.get("Black", "").lower()

        if player_id == "":
            include_white = True
            include_black = True
        else:
            include_white = (player_id == white)
            include_black = (player_id == black)
            if not (include_white or include_black):
                continue  # skip
        board = game.board()

        for move in game.mainline_moves():
            if (board.turn == chess.WHITE and include_white) or (board.turn == chess.BLACK and include_black):
                x = encode_board(board)
                y = move_to_conf_matrix(move)
                samples.append((x, y))

            board.push(move)

    return samples
    
def encode_board(board):
    tensor = np.zeros((8, 8, 12), bool)

    for square, piece in board.piece_map().items():
        
        row = 7 - (square // 8)
        col = square % 8
        plane = piece.piece_type - 1
        if piece.color == chess.BLACK:
            plane += 6
        
        tensor[row, col, plane] = 1.0
    return tensor




def move_to_conf_matrix(move):
    
    matrix = np.zeros((8, 8, 8, 12), dtype=np.float32)
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
        matrix[from_row, from_col, to_row, promo_offset] = 1.0
    else:
        matrix[from_row, from_col, to_row, to_col] = 1.0

    return matrix

    
class Agent:
    def __init__(self, id):
        print('Agent V2')
        print(tf.config.list_physical_devices('GPU'))
        self.id = id

    def is_blunder(self, board, move, sf, threshold=200):
    
        sf.set_fen_position(board.fen())
        before = sf.get_evaluation()['value']
    
        # Get eval after move
        board_copy = board.copy()
        board_copy.push(move)
        sf.set_fen_position(board_copy.fen())
        after = -sf.get_evaluation()['value']
    
        if sf.get_evaluation()['type'] == "mate":
            print('mate')
            return False
            
        print(before, after, board, move)
        return (before - after) > threshold
    def act(self, state, sf=None, blunder_threshold=200):
        x = encode_board(state)
        x = np.expand_dims(x, axis=0)
        probs = self.model.predict(x, verbose=0)[0]  # shape (8,8,8,12)

        legal_moves = list(state.legal_moves)
        print(state.legal_moves)
        scored_moves = []
        for move in legal_moves:
            
            m = move_to_conf_matrix(move)
            print(m)
            score = np.sum(probs * m)
            scored_moves.append((score, move))

        scored_moves.sort(key=lambda x: x[0], reverse=True)

        blocked = 0
        print('selecting move', sf)
        for score, move in scored_moves:
            if sf is None or not self.is_blunder(state, move, sf, 100):
                return [move, blocked]
            else:
                blocked += 1
                print('Move blocked:', move)

        
        return [scored_moves[0][1], blocked] if scored_moves else ValueError('Error: No Legal Move in Matrix')

    def train(self, games):
        samples = pgn_to_player_samples(games, self.id)
        batch_size = 32
        epochs = 1
        n = len(samples)
    
        def gen():
            for i in range(0, n, batch_size):
                batch = samples[i:i+batch_size]
                X = np.array([s[0] for s in batch], dtype=np.float32)
                Y = np.array([s[1] for s in batch], dtype=np.float32)  # (batch, 8,8,8,12)
                yield X, Y
    
        def build_model():
            inputs = tf.keras.layers.Input(shape=(8, 8, 12))
            
            x = tf.keras.layers.Reshape((64, 12))(inputs)
            
            x = tf.keras.layers.Dense(256, activation='relu')(x)
            x = tf.keras.layers.LayerNormalization()(x)

            # attention blocks
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
            
            # dense layers
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(4096, activation='relu')(x)
            x = tf.keras.layers.LayerNormalization()(x)
            x = tf.keras.layers.Dropout(0.2)(x)
        
            x = tf.keras.layers.Dense(2048, activation='relu')(x)
            x = tf.keras.layers.LayerNormalization()(x)
            x = tf.keras.layers.Dropout(0.2)(x)
        
            x = tf.keras.layers.Dense(1024, activation='relu')(x)
            x = tf.keras.layers.LayerNormalization()(x)
            x = tf.keras.layers.Dropout(0.1)(x)
        
            x = tf.keras.layers.Dense(512, activation='relu')(x)
            
            # Output
            x = tf.keras.layers.Dense(8*8*8*12, activation='softmax')(x)
            outputs = tf.keras.layers.Reshape((8, 8, 8, 12))(x)
            
            return tf.keras.Model(inputs, outputs)
        
        self.model = build_model()
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
        steps = (n + batch_size - 1) // batch_size
        self.model.fit(gen(), steps_per_epoch=steps, epochs=epochs)
