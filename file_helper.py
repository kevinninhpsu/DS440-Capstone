

import json
from collections import defaultdict
import re
import io
player_games = defaultdict(set)
player_sum_elo = defaultdict(int)
player_sum_game_len = defaultdict(int)

player_max_elo = defaultdict(int)
player_min_elo = defaultdict(int)


EVAL_RE = re.compile(r"\{.*\}")
def count_entries(path):
	with open(path, "r", encoding="utf-8") as f:
		for i, line in enumerate(f):
			if i % 1_000_000 == 0 and i > 0:
				print(f"games processed: {i // 1_000_000}M")
				
			game = json.loads(line)

			
			player_games[game["white"]].add(game["moves"])

			
			e = int(game['elo'])
			player_sum_elo[game['white']] += e
			player_sum_game_len[game['white']] += game['moves'].count('.')

			if player_min_elo[game["white"]] == 0 or e < player_min_elo[game["white"]]:
				player_min_elo[game["white"]] = e
	
			if player_max_elo[game["white"]] == 0 or e > player_max_elo[game["white"]]:
				player_max_elo[game["white"]] = e



#count_entries("games.jsonl")

#import pgntofen
import chess.pgn
import re
DIV_RE = re.compile(r"(\s?(\d)+\.\s)")
state_move_pairs = set()

def extract_player_move_states(path, username):
	with open(path, "r", encoding="utf-8") as f:
		data = json.load(f)
		
	for i, game in enumerate(data):

		if game['white'] == username:
			
			board = chess.Board()

			#print(game['moves'])
			moves = DIV_RE.sub('|',game['moves']).split('|')[1:-1]
			
			for move_pair in moves:
				
				white, black = move_pair.split(' ')
				state_move_pairs.add((white, board.fen()))

				board.push_san(white)
				board.push_san(black)


						
				
def get_past_moves(state_move_pairs, state):
	moves = {}


	for i in state_move_pairs:
		print(i)
		if i[1] == state:
			if i[0] in moves:
				moves[i[0]] += 1
			else:
				moves[i[0]] = 1
	return moves


def extract_games_by_player(path, username):

	games = []
	with open(path, "r", encoding="utf-8") as f:
		for i, line in enumerate(f):
			game = json.loads(line)
			e = int(game['elo'])
			if e > 1400 and e < 1500:

				game['moves'] = EVAL_RE.sub("", game['moves'])
				if game['moves'].count('.') < 5:
					continue

				games.append(game)
	print("length:", len(games))
	#with open(r"player_games/" + username + "_games.json", "w") as f:
	#	json.dump(games, f, indent=1)
					
def getTruncatedGames(path):
	games = []
	with open(path, "r", encoding="utf-8") as f:

		count = 0
		data = json.load(f)
		for game in data:
			count += 1
			if count % 10_000 == 0:
				print(count)
		
			if game['moves'].count('.') >= 30:
				t = truncateGame(game)
				if t:
					game['moves'] = t
					games.append(game)
	with open(r"reduced_"+path, "w+") as f:
		json.dump(games, f, indent=1)
					


def truncateGame(game):
	PIECE_VALUES = {
		chess.PAWN: 1,
		chess.KNIGHT: 3,
		chess.BISHOP: 3,
		chess.ROOK: 5,
		chess.QUEEN: 9,
	}

	pgn = io.StringIO(game['moves'])
	pgn_game = chess.pgn.read_game(pgn)
	board = pgn_game.board()
	moves_played = []

	for i, move in enumerate(pgn_game.mainline_moves()):
		board.push(move)
		moves_played.append(move)
		if i == 59:
			break

	def material(color):
		return sum(
			PIECE_VALUES[pt] * len(board.pieces(pt, color))
			for pt in PIECE_VALUES
		)

	if material(chess.WHITE) < 16 or material(chess.BLACK) < 16:
		return False
		
	replay_board = pgn_game.board()
	move_tokens = []
	for i, move in enumerate(moves_played):
		if i % 2 == 0:
			move_tokens.append(f"{(i // 2) + 1}.")
		move_tokens.append(replay_board.san(move))
		replay_board.push(move)

	return ' '.join(move_tokens)
#count_entries('games_standard.jsonl')
#extract_player_move_states('asghar7arab_games.json','asghar7arab')

#extract_player_move_states('theuglyamerican_games.json','theuglyamerican')
#print(state_move_pairs)

#print(get_past_moves(state_move_pairs, chess.Board().fen()))


getTruncatedGames('1400-1500_games.json')


#extract_games_by_player('games_standard.jsonl','mrkeshavarz2025')

#['asghar7arab', 'mrkeshavarz2025']

#print(len(player_games))

#count = 1
#for player in sorted(player_games.keys(), key=lambda x: len(player_games[x]), reverse=True):
#	print(f'{count}. {player}: {len(player_games[player])} games')
#	print(f'  elo: {player_sum_elo[player] // len(player_games[player])} ({player_min_elo[player]}|{player_max_elo[player]})')
#	print(f'  avg # moves: {player_sum_game_len[player] // len(player_games[player])}')

#	count += 1
#	if count > 20:
#		break
