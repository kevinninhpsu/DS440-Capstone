import chess.pgn
import json
import numpy as np
import tensorflow as tf
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
DIV_RE = re.compile(r"(\s?(\d)+\.\s|\s)")
def load_json(json_file):
    games = []
    with open(json_file, "r") as f:
        data = json.load(f)
    for g in data:
        game = chess.pgn.Game()
        game.headers["Event"] = g.get("event", "?")
        game.headers["White"] = g.get("white", "?")
        game.headers["Black"] = g.get("black", "?")
        game.headers["Result"] = g.get("result", "*")
        board = chess.Board()
        node = game
        moves = DIV_RE.sub('|',g.get('moves')).split('|')[1:-1]
        for move in moves:
            move = board.push_san(move.strip(" "))
            node = node.add_variation(move)
            
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
                y = move_to_index(move)
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
    
def move_to_index(move):
    if move.promotion:
        promo_offset = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 3}[move.promotion]
        return 4096 + move.from_square * 4 + promo_offset
    else:
        return move.from_square * 64 + move.to_square

    
class Agent:
    def __init__(self, id):
        self.id = id

    def act(self, state):
        x = encode_board(state)         
        x = np.expand_dims(x, axis=0)
        probs = self.model.predict(x, verbose=0)[0]
        best_move = None
        best_score = -1.0    
        for move in state.legal_moves:
            idx = move_to_index(move)
            score = probs[idx]
            if score > best_score:
                best_score = score
                best_move = move
        return best_move
    
    def train(self,games):
        
        samples = pgn_to_player_samples(games,self.id)
        batch_size = 32
        epochs=1
        n = len(samples)
        def gen():
            for i in range(0, n, batch_size):
                batch = samples[i:i+batch_size]
                X = np.array([s[0] for s in batch], dtype=bool).astype(np.float32)
                Y = np.array([s[1] for s in batch], dtype=np.int32)
                yield X, Y
                
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(8,8,12)),         
            tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4672, activation='softmax')  
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        

        steps = (n + batch_size - 1) // batch_size
        self.model.fit(gen(), steps_per_epoch=steps, epochs=epochs)