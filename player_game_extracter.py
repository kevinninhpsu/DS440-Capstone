

import json
from collections import defaultdict
import re
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
				if game['white'] == 'jmanette411':
					pass
					#print('new_min')

			if player_max_elo[game["white"]] == 0 or e > player_max_elo[game["white"]]:
				player_max_elo[game["white"]] = e
				if game['white'] == 'jmanette411':
					pass
					#print('new_max')
			
			#if game['white'] == 'jmanette411':
			#	if e == 999:
			#		print(game['elo'])
			#		print(player_min_elo[game["white"]], player_max_elo[game["white"]])
			#		input()

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
					



#count_entries('games_standard.jsonl')
#extract_player_move_states('asghar7arab_games.json','asghar7arab')

#extract_player_move_states('theuglyamerican_games.json','theuglyamerican')
print(state_move_pairs)

#print(get_past_moves(state_move_pairs, chess.Board().fen()))


extract_games_by_player('games_standard.jsonl','1400-1500')


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
