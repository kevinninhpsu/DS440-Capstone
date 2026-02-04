import zstandard as zstd
import io
import json
import os
import signal
import sys
import re

# --------------------
# Files
# --------------------

INPUT_FILE = r"C:\Users\llsad\Downloads\lichess_db_standard_rated_2025-12.pgn.zst"
OUTPUT_FILE = "games.jsonl"
PROGRESS_LINE_FILE = "progress_line.txt"
PROGRESS_KEY_FILE = "progress_key.txt"

# --------------------
# Regex
# --------------------

RESULT_RE = re.compile(r"\b(1-0|0-1|1/2-1/2)\b")
CLOCK_RE = re.compile(r"\{\s*\[%clk [^\]]+\]\s*\}")
BLACK_MOVE_RE = re.compile(r"\b\d+\.\.\.\s*")

# --------------------
# Progress handling
# --------------------

def load_progress():
	if os.path.exists(PROGRESS_LINE_FILE):
		with open(PROGRESS_LINE_FILE, "r", encoding="utf-8") as f:
			line = f.read().strip()

	if os.path.exists(PROGRESS_KEY_FILE):
		with open(PROGRESS_KEY_FILE, "r", encoding="utf-8") as f:
			key = f.read().strip()
			
		
		return int(line) if line else None, key if key else None
	return 0, None


def save_progress(line, game_id):
	with open(PROGRESS_LINE_FILE, "w", encoding="utf-8") as f:
		f.write(str(line))
	with open(PROGRESS_KEY_FILE, "w", encoding="utf-8") as f:
		f.write(game_id)


last_saved_key = None
last_saved_line = 0

def handle_sigint(signum, frame):
	print("\nInterrupted — saving progress...")
	if last_saved_key:
		save_progress(last_saved_line,last_saved_key)
	sys.exit(0)

signal.signal(signal.SIGINT, handle_sigint)

# --------------------
# Game handler
# --------------------

def handle_game(headers, moves, out, resume_key, skipping):
	global last_saved_key, last_saved_line

	tc = headers.get('TimeControl')
	if not tc or tc != '180+2':
		return False
	game_id = headers.get("UTCDate", "") + "T" + headers.get("UTCTime", "") + '|' + headers.get('Site')[-8:]

	last_saved_key = game_id

	# Resume logic: skip until we pass the last processed game
	if skipping:
		if resume_key is not None:
			if game_id != resume_key:
				return False   # skipped
			else:
				resume_key = None  # resume complete

	moves_text = " ".join(moves)
	moves_text = CLOCK_RE.sub("", moves_text)
	moves_text = BLACK_MOVE_RE.sub("", moves_text)
	moves_text = " ".join(moves_text.split())
	record = {
		"id": game_id,
		"white": headers.get("White"),
		"time_control": tc,
		"elo": headers.get('WhiteElo'),
		"moves": moves_text,
	}

	out.write(json.dumps(record) + "\n")
	return True

# --------------------
# Main loop
# --------------------

resume_line, resume_key = load_progress()
if resume_key:
	print(f"Resuming after game: {resume_key}")

if resume_line:
	print(f"Resuming after line: {resume_line}")

processed = 0
skipped = 0
skipping = True

from itertools import islice

with open(INPUT_FILE, "rb") as compressed, \
	 open(OUTPUT_FILE, "a", encoding="utf-8") as out:

	dctx = zstd.ZstdDecompressor()
	with dctx.stream_reader(compressed) as reader:
		stream = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")

		headers = {}
		moves = []
		in_game = False

		game_count = 0
		line_count = 0

		for _ in islice(stream, resume_line):
			line_count += 1
			if line_count % 10_000_000 == 0:
				print('line count: ', line_count // 1_000_000,'M', sep ='')


		for line in stream:
			line = line.strip()

			line_count += 1
			if line_count % 1_000_000 == 0:
				print('line count: ', line_count // 1_000_000,'M', sep ='')

			#if line_count < resume_line:
			#	continue

			# New game starts
			if line.startswith("[Event "):
				game_count += 1

				if game_count % 100_000 == 0:
					print(game_count)
				if headers and moves:
					written = handle_game(headers, moves, out, resume_key, skipping)
					last_saved_line = line_count
					if written:
						skipping = False
						processed += 1
						if processed % 10_000 == 0:
							print(f"Processed {processed:,} games")
							save_progress(last_saved_line, last_saved_key)
					else:
						skipped += 1

				headers = {}
				moves = []
				in_game = True

			# Header lines
			if in_game and line.startswith("[") and line.endswith("]"):
				key, val = line[1:-1].split(" ", 1)
				headers[key] = val.strip('"')
				continue

			# Move text
			if in_game and line:
				moves.append(line)

				if RESULT_RE.search(line):
					written = handle_game(headers, moves, out, resume_key, skipping)
					if written:
						skipping = False
						processed += 1
						if processed % 10_000 == 0:
							print(f"Processed {processed:,} games")
							save_progress(last_saved_line,last_saved_key)
					else:
						skipped += 1

					headers = {}
					moves = []
					in_game = False

print(f"Done. Processed={processed:,}, Skipped={skipped:,}")
if last_saved_key:
	save_progress(last_saved_line,last_saved_key)
