

import json
from collections import defaultdict

player_games = defaultdict(set)

def count_entries(path):
	with open(path, "r", encoding="utf-8") as f:
		for i, line in enumerate(f):
			if i % 1_000_000 == 0 and i > 0:
				print(f"games processed: {i // 1_000_000}M")
				
			game = json.loads(line)

			player_games[game["white"]].add(game["id"])

count_entries("games.jsonl")

print(len(player_games))

count = 1
for player in sorted(player_games.keys(), key=lambda x: len(player_games[x]), reverse=True):
	print(f'{count}. {player}: {len(player_games[player])} games')
	
	count += 1
	if count > 9:
		break
