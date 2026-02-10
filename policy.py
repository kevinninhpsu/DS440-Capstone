import chess.pgn
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
def pgn_to_player_samples(games, player_id):
    samples = []
    player_id = player_id.lower()

    for game in games:
        white = game.headers.get("White", "").lower()
        black = game.headers.get("Black", "").lower()

        if player_id == white:
            player_color = chess.WHITE
        elif player_id == black:
            player_color = chess.BLACK
        else:
            continue

        board = game.board()

        for move in game.mainline_moves():
            if board.turn == player_color:
                x = encode_board(board)      # ✅ numeric tensor
                y = move_to_index(move)      # ✅ integer
                samples.append((x, y))

            board.push(move)

    return samples
    
def encode_board(board):
    tensor = np.zeros((12, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8
        plane = piece.piece_type - 1
        if piece.color == chess.BLACK:
            plane += 6
        tensor[plane, row, col] = 1.0
    return tensor
    
def move_to_index(move): 
    return move.from_square * 64 + move.to_square
    
class Agent:
    def __init__(self, id):
        self.id = id

    def act(self, state):
        x = encode_board(board)         
        x = np.expand_dims(x, axis=0)
        probs = self.model.predict(x, verbose=0)[0]
        best_move = None
        best_score = -1.0    
        for move in state.legal_moves:
            idx = move.from_square * 64 + move.to_square
            score = probs[idx]
            if score > best_score:
                best_score = score
                best_move = move
        return best_move
        
    def train(self,games):
        samples = pgn_to_player_samples(games,self.id)
        
        X = np.array([s[0] for s in samples], dtype=np.float32)
        Y = np.array([s[1] for s in samples], dtype=np.int32)
        
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(12,8,8)),          # board tensor
            tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4672, activation='softmax')  # all possible moves
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model.fit(
            X, Y,
            batch_size=128,
            epochs=10,
            validation_split=0.1,
            shuffle=True
        ) 