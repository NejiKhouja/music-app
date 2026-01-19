#!/usr/bin/env python3
"""
fast_language_detector_mp_enhanced.py

Level-2 multiprocessing language tagging with:
- SQLite caching (search + lyrics) safe across processes & restarts
- Artist-level caching heuristics
- Genius token rotation (round-robin)
- Retries + exponential backoff + logging
- tqdm progress bars
- FastText fallback (optional, recommended)
"""

import requests
import re
import time
import sqlite3
import os
import logging
import multiprocessing as mp
from multiprocessing import Value, Lock
from functools import partial
from typing import Optional, Tuple
from tqdm import tqdm
import pandas as pd

# Optional fasttext import
try:
    import fasttext
    FASTTEXT_AVAILABLE = True
except Exception:
    FASTTEXT_AVAILABLE = False

# ----------------------------
# CONFIG
# ----------------------------
GENIUS_TOKENS = [
    # put one or more tokens here. If you have only one token, leave only one.
    "GzX5JJNervheF9V2BCRoIjVEQXzzA7hBH-i24Hie1De0PbucoVBd74eZGkbh4Dw0",
    "XE8f_QiK-cjoxOE4rQpvXAHLzaeQvs6zzWd-I_WtkkUd4Z8rez8SkQxRf1KMQxpI",
    "jzlw-GKhdugdCtKMFsicErlP0qZq7byAEnCd8RJaDIDe65lzD6ysxsgRkzDSNc5K"

    # "YOUR_TOKEN_2",
]
SQLITE_CACHE_PATH = "genius_cache.db"
FASTTEXT_MODEL_PATH = "lid.176.ftz"  # change if stored elsewhere
WORKERS = 1
REQUEST_TIMEOUT = 8
MAX_RETRIES = 3
BACKOFF_FACTOR = 20.0  # seconds
SEARCH_PER_PAGE = 1

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("language_detector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("langdet")

# ----------------------------
# SQLite cache helper (process-safe)
# ----------------------------
def init_cache(db_path=SQLITE_CACHE_PATH):
    """Create sqlite DB and tables if missing."""
    con = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS search_cache (
        key TEXT PRIMARY KEY,
        json TEXT,
        fetched_at REAL
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS lyrics_cache (
        song_url TEXT PRIMARY KEY,
        lyrics TEXT,
        fetched_at REAL
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS artist_cache (
        artist TEXT PRIMARY KEY,
        representative_url TEXT,
        fetched_at REAL
    )""")
    con.commit()
    return con

def cache_get(con: sqlite3.Connection, table: str, key_col: str, key_value: str) -> Optional[str]:
    cur = con.cursor()
    cur.execute(f"SELECT { 'json' if table=='search_cache' else ('lyrics' if table=='lyrics_cache' else 'representative_url') } FROM {table} WHERE {key_col} = ?", (key_value,))
    row = cur.fetchone()
    return row[0] if row else None

def cache_set(con: sqlite3.Connection, table: str, key_col: str, key_value: str, value: str):
    cur = con.cursor()
    col_val = 'json' if table=='search_cache' else ('lyrics' if table=='lyrics_cache' else 'representative_url')
    cur.execute(f"INSERT OR REPLACE INTO {table} ({key_col},{col_val},fetched_at) VALUES (?,?,?)", (key_value, value, time.time()))
    con.commit()

# ----------------------------
# Genius token rotation (shared counter)
# ----------------------------
def get_token_rotator(tokens):
    idx = Value('i', 0)
    lock = Lock()
    def next_token():
        with lock:
            i = idx.value
            token = tokens[i % len(tokens)]
            idx.value = (i + 1) % len(tokens)
            return token
    return next_token

NEXT_TOKEN = get_token_rotator(GENIUS_TOKENS)

# ----------------------------
# HTTP request with retries & backoff
# ----------------------------
def requests_get_with_retries(url, headers=None, params=None, timeout=REQUEST_TIMEOUT, max_retries=MAX_RETRIES):
    attempt = 0
    while attempt < max_retries:
        try:
            r = requests.get(url, headers=headers, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            # For rate limits or server errors, raise to retry
            if r.status_code in (429, 500, 502, 503, 504):
                logger.warning(f"HTTP {r.status_code} on {url}; attempt {attempt+1}/{max_retries}")
                raise requests.RequestException(f"HTTP {r.status_code}")
            # otherwise return (e.g., 404)
            return r
        except Exception as e:
            sleep_for = BACKOFF_FACTOR * (2 ** attempt)
            logger.debug(f"Request error: {e} - sleeping {sleep_for:.1f}s")
            time.sleep(sleep_for)
            attempt += 1
    # final attempt
    try:
        r = requests.get(url, headers=headers, params=params, timeout=timeout)
        return r
    except Exception as e:
        logger.error(f"Final request failed for {url}: {e}")
        return None

# ----------------------------
# Lyrics extraction & search (cached)
# ----------------------------
def genius_search_and_cache(con: sqlite3.Connection, track_name: str, artist: str) -> Optional[dict]:
    """Return Genius search result (as dict) or None. Uses sqlite cache."""
    key = f"{track_name}||{artist}"
    cached = cache_get(con, "search_cache", "key", key)
    if cached:
        try:
            import json
            return json.loads(cached)
        except Exception:
            pass

    token = NEXT_TOKEN()
    headers = {"Authorization": f"Bearer {token}"}
    url = "https://api.genius.com/search"
    params = {"q": f"{track_name} {artist}", "per_page": SEARCH_PER_PAGE}

    r = requests_get_with_retries(url, headers=headers, params=params)
    if r is None:
        return None
    if r.status_code != 200:
        logger.warning(f"Search non-200 ({r.status_code}) for '{track_name}' by '{artist}'")
        return None
    try:
        resp = r.json()
        hits = resp.get("response", {}).get("hits", [])
        if not hits:
            # store empty result to avoid repeated lookups
            import json
            cache_set(con, "search_cache", "key", key, json.dumps({}))
            return None
        best = hits[0]["result"]
        import json
        cache_set(con, "search_cache", "key", key, json.dumps(best))
        return best
    except Exception as e:
        logger.error(f"Error parsing Genius search JSON: {e}")
        return None

def fetch_lyrics_and_cache(con: sqlite3.Connection, song_url: str) -> str:
    cached = cache_get(con, "lyrics_cache", "song_url", song_url)
    if cached:
        return cached
    r = requests_get_with_retries(song_url)
    if r is None or r.status_code != 200:
        return ""
    html = r.text
    # Find Genius lyrics containers
    containers = re.findall(r'<div[^>]*class="[^"]*Lyrics__Container[^"]*"[^>]*>([\s\S]*?)</div>', html)
    if not containers:
        # fallback: take meta description or page inner text heuristics:
        # remove tags and get some text
        text = re.sub(r'<[^>]+>', '\n', html)
        # fallback short heuristics
        text_lines = [l.strip() for l in text.splitlines() if l.strip() and len(l.strip()) > 8]
        lyrics = " ".join(text_lines[:15])
        cache_set(con, "lyrics_cache", "song_url", song_url, lyrics)
        return lyrics
    lines = []
    for c in containers:
        text = re.sub(r'<[^>]*>', '\n', c)
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            if line.startswith('[') and line.endswith(']'):
                continue
            if line.isupper():
                continue
            if len(line) < 4:
                continue
            lines.append(line)
    lyrics = " ".join(lines[:30])
    cache_set(con, "lyrics_cache", "song_url", song_url, lyrics)
    return lyrics

# ----------------------------
# Language detection (fasttext fallback)
# ----------------------------
class RuleBasedDetector:
    def __init__(self):
        # script patterns for non-latin languages
        self.script_patterns = {
            'ar': [r'[\u0600-\u06FF]'],
            'ja': [r'[\u3040-\u309F]', r'[\u30A0-\u30FF]'],
            'ko': [r'[\uAC00-\uD7AF]'],
            'zh': [r'[\u4E00-\u9FFF]'],
            'ru': [r'[\u0400-\u04FF]'],
        }
        self.latin_indicators = {
            'en': ['the','and','you','that','with','this','have','your','just','into','from','what','is','are'],
            'pl': ['się','jest','tak','ale','być','moja','twoja','kiedy','dlaczego','nie','co','na','w','i'],
            'es': ['que','para','como','pero','porque','sobre','donde','quien'],
            'fr': ['que','est','pas','pour','dans','avec','elle','nous'],
            'de': ['und','ich','nicht','dass','ein','die','der','wir','mit'],
        }

    def detect(self, text: str) -> Tuple[str, float]:
        if not text or len(text.strip()) < 15:
            return ("unknown", 0.0)
        t = text.lower()
        # script detection
        for lang, pats in self.script_patterns.items():
            for p in pats:
                if re.search(p, t):
                    return (lang, 1.0)
        words = re.findall(r"\b[a-zA-Z]{3,}\b", t)
        if not words:
            return ("unknown", 0.0)
        scores = {}
        for lang, inds in self.latin_indicators.items():
            hits = sum(1 for w in words if w in inds)
            scores[lang] = hits / len(words)
        best_lang, best_score = max(scores.items(), key=lambda x: x[1])
        if best_score < 0.02:
            return ("unknown", best_score)
        return (best_lang, best_score)

# fasttext wrapper
FASTTEXT_MODEL = None
if FASTTEXT_AVAILABLE and os.path.exists(FASTTEXT_MODEL_PATH):
    try:
        FASTTEXT_MODEL = fasttext.load_model(FASTTEXT_MODEL_PATH)
        logger.info("Loaded fastText model.")
    except Exception as e:
        logger.warning(f"Failed to load fastText model: {e}")
        FASTTEXT_MODEL = None
else:
    if FASTTEXT_AVAILABLE:
        logger.warning(f"fasttext available but model not found at {FASTTEXT_MODEL_PATH}")

rule_detector = RuleBasedDetector()

def detect_language(text: str) -> Tuple[str, float]:
    """Returns (lang_code, confidence). Uses fastText when available, else rule-based."""
    if not text or len(text.strip()) < 10:
        return ("unknown", 0.0)
    # prefer fasttext if available
    if FASTTEXT_MODEL:
        try:
            preds = FASTTEXT_MODEL.predict(text.replace("\n", " "), k=2)
            labels, probs = preds
            # labels like "__label__en"
            label = labels[0].replace("__label__", "")
            prob = float(probs[0])
            # quick Arabic character override to avoid urdu/persian mistakes
            if re.search(r'[\u0600-\u06FF]', text) and label != "ar":
                return ("ar", 1.0)
            return (label, prob)
        except Exception as e:
            logger.debug(f"fasttext predict error: {e}")
            # fallback to rule-based
    return rule_detector.detect(text)

# ----------------------------
# Worker processing
# ----------------------------
def worker_process(row, db_path=SQLITE_CACHE_PATH):
    # Each worker opens its own sqlite connection
    con = init_cache(db_path)
    try:
        track_name = str(row.get("track_name", "")).strip()
        artist = str(row.get("artists", "")).strip()
        if not track_name:
            return "unknown"

        # Try cached search
        song_data = genius_search_and_cache(con, track_name, artist)
        if not song_data:
            # heuristics: if artist cached representative_url exists, skip search and use it
            art_cached = cache_get(con, "artist_cache", "artist", artist)
            if art_cached:
                lyrics = fetch_lyrics_and_cache(con, art_cached)
                if lyrics:
                    lang, conf = detect_language(lyrics)
                    if lang != "unknown":
                        return lang
            return "unknown"

        song_url = song_data.get("url")
        if not song_url:
            return "unknown"

        # Save representative_url for artist (artist-level caching)
        try:
            cache_set(con, "artist_cache", "artist", artist, song_url)
        except Exception:
            pass

        lyrics = fetch_lyrics_and_cache(con, song_url)
        if not lyrics:
            # fallback to title+artist combined
            fallback_text = f"{track_name} {artist}"
            lang, conf = detect_language(fallback_text)
            if lang != "unknown" and conf > 0.6:
                return lang
            # final fallback to unknown
            return "unknown"

        # Primary detection on lyrics
        lang, conf = detect_language(lyrics)
        # If fasttext returned low confidence and rule-based suggests something, prefer rule-based if confidences align
        if conf < 0.6:
            rule_lang, rule_conf = rule_detector.detect(lyrics)
            if rule_lang != "unknown" and rule_conf > 0.3:
                return rule_lang
        return lang
    except Exception as e:
        logger.exception(f"Worker exception for row {row}: {e}")
        return "unknown"
    finally:
        try:
            con.close()
        except:
            pass

# ----------------------------
# Multiprocessing driver
# ----------------------------
def process_dataset_mp(csv_path: str, output_path: str, workers: int = WORKERS, db_path: str = SQLITE_CACHE_PATH):
    # init db in main process
    con = init_cache(db_path)
    con.close()

    df = pd.read_csv(csv_path)
    rows = df.to_dict("records")

    logger.info(f"Starting processing {len(rows)} rows with {workers} workers.")
    results = []
    with mp.Pool(workers) as pool:
        for idx, (row, lang) in enumerate(tqdm(zip(rows, pool.imap_unordered(partial(worker_process, db_path=db_path), rows)), total=len(rows))):
            results.append(lang)
            
            # print progress every 200
            if (idx + 1) % 200 == 0:
                track = row.get("track_name", "")
                artist = row.get("artists", "")
                print(f"[{idx+1}] \"{track}\" - \"{artist}\" -> {lang}")

    df["language"] = results
    df.to_csv(output_path, index=False)
    logger.info(f"Finished. Saved to {output_path}")

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fast Language Detector")

    # Default paths encoded inside the script
    DEFAULT_INPUT = "/home/neji/python/music_recommender/data/raw/tracks.csv"
    DEFAULT_OUTPUT = "/home/neji/python/music_recommender/data/tracks_with_language.csv"
    DEFAULT_WORKERS = 12
    DEFAULT_DB = "/home/neji/python/music_recommender/genius_cache.sqlite"

    parser.add_argument("--input", "-i", default=DEFAULT_INPUT,
                        help=f"Input CSV file (default: {DEFAULT_INPUT})")

    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT,
                        help=f"Output CSV file (default: {DEFAULT_OUTPUT})")

    parser.add_argument("--workers", "-w", type=int, default=DEFAULT_WORKERS,
                        help=f"Number of parallel workers (default: {DEFAULT_WORKERS})")

    parser.add_argument("--db", default=DEFAULT_DB,
                        help=f"SQLite cache path (default: {DEFAULT_DB})")

    args = parser.parse_args()

    if len(GENIUS_TOKENS) == 0 or GENIUS_TOKENS[0].startswith("YOUR_TOKEN"):
        logger.error("Please configure GENIUS_TOKENS in the script before running.")
        raise SystemExit(1)

    if FASTTEXT_AVAILABLE and not os.path.exists(FASTTEXT_MODEL_PATH):
        logger.warning(f"fastText installed but model not found at {FASTTEXT_MODEL_PATH}. FastText fallback will be unavailable.")

    process_dataset_mp(args.input, args.output, workers=args.workers, db_path=args.db)
