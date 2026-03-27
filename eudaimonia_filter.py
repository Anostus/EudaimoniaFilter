#!/usr/bin/env python3
"""
Eudaimonia RSS Filter
=====================
Fetches RSS feeds, scores each entry against three philosophical criteria
using DeepSeek R1, and outputs a filtered RSS feed containing only items
that score highly on at least one criterion.

Designed to run on a schedule (4x/day) via GitHub Actions.
"""

import json
import logging
import os
import re
import sys
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import feedparser
from feedgen.feed import FeedGenerator
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RSS_FEEDS: list[str] = json.loads(os.getenv("RSS_FEEDS", json.dumps([
    "https://www.theguardian.com/us-news/rss",
    "https://www.theguardian.com/world/rss",
    "https://www.pbs.org/newshour/feeds/rss/headlines",
    "https://www.propublica.org/feeds/propublica/main",
])))

DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")

OUTPUT_FILE: str = os.getenv("OUTPUT_FILE", "docs/feed.xml")

# How many entries to pull from each feed (0 = all)
MAX_ENTRIES_PER_FEED: int = int(os.getenv("MAX_ENTRIES_PER_FEED", "25"))

# Batch size for LLM calls (each batch = 1 API call)
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "10"))

# Minimum score (1-10) on ANY single criterion to pass the filter
SCORE_THRESHOLD: int = int(os.getenv("SCORE_THRESHOLD", "7"))

# Path to a JSON file that tracks already-seen entry IDs across runs,
# so we don't re-judge articles we've already processed.
SEEN_DB_PATH: str = os.getenv("SEEN_DB_PATH", "data/seen.json")

# How many days to keep entries in the output feed before they age out
FEED_RETENTION_DAYS: int = int(os.getenv("FEED_RETENTION_DAYS", "7"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("eudaimonia")

# ---------------------------------------------------------------------------
# Seen-entries database (simple JSON file)
# ---------------------------------------------------------------------------


def _entry_id(entry) -> str:
    """Produce a stable ID for an RSS entry."""
    link = getattr(entry, "link", "") or ""
    title = getattr(entry, "title", "") or ""
    raw = link or title
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_seen_db() -> dict:
    """Load the seen-entries database. Returns {id: iso_timestamp}."""
    path = Path(SEEN_DB_PATH)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            log.warning("Corrupt seen DB; starting fresh.")
    return {}


def save_seen_db(db: dict) -> None:
    """Persist the seen-entries database."""
    path = Path(SEEN_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(db, indent=2))


# ---------------------------------------------------------------------------
# RSS Fetching
# ---------------------------------------------------------------------------


def fetch_entries() -> list:
    """Fetch and deduplicate entries from all configured RSS feeds."""
    all_entries = []
    seen_links = set()

    for url in RSS_FEEDS:
        log.info("Fetching feed: %s", url)
        try:
            feed = feedparser.parse(url)
            if feed.bozo and not feed.entries:
                log.warning("Feed parse error for %s: %s", url, feed.bozo_exception)
                continue

            entries = feed.entries
            if MAX_ENTRIES_PER_FEED > 0:
                entries = entries[:MAX_ENTRIES_PER_FEED]

            for entry in entries:
                link = getattr(entry, "link", "") or ""
                if link and link in seen_links:
                    continue
                seen_links.add(link)
                all_entries.append(entry)

            log.info("  Got %d entries from %s", len(entries), url)
        except Exception as e:
            log.error("Failed to fetch %s: %s", url, e)

    log.info("Total unique entries fetched: %d", len(all_entries))
    return all_entries


# ---------------------------------------------------------------------------
# LLM Scoring
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a philosophical filter for an RSS feed reader. Your job is to score \
each article on three independent criteria. An article only needs to score \
highly on ONE criterion to pass — it does not need to pass all three.

## Criteria

1. **Lasting Importance (lasting)**
   Will this still matter in 10 years? Will people remember it as a genuinely \
significant event, discovery, or development from this era — or is it a \
pseudo-event, rage-bait, ephemeral drama, or incremental news that will be \
forgotten in weeks?

2. **Personal Growth (growth)**
   Will reading or engaging with this make someone a better person? Does it \
cultivate moral insight, intellectual depth, empathy, practical wisdom, or \
meaningful skill — as opposed to mere entertainment, outrage, or trivia?

3. **Eudaimonia (eudaimonia)**
   Does this contribute to human flourishing? Does it help someone live with \
greater purpose, meaning, connection, or well-being — in the Aristotelian \
sense of a life well-lived?

## Instructions

For each article, return a JSON object with these fields:
- "index": the article index number
- "lasting": integer score 1-10
- "growth": integer score 1-10
- "eudaimonia": integer score 1-10
- "reason": a single sentence explaining why the highest-scoring criterion \
earned its score (keep it brief)

Return a JSON array of these objects — one per article. Return ONLY valid JSON, \
no markdown fences, no commentary outside the array.\
"""


def build_article_block(index: int, entry) -> str:
    """Format a single RSS entry for the LLM prompt."""
    title = getattr(entry, "title", "(no title)") or "(no title)"
    summary = getattr(entry, "summary", "") or ""
    # Strip HTML tags from summary
    summary = re.sub(r"<[^>]+>", " ", summary)
    summary = re.sub(r"\s+", " ", summary).strip()
    # Truncate to keep prompt size reasonable
    if len(summary) > 300:
        summary = summary[:297] + "..."
    return f"[{index}] {title}\n    {summary}"


def score_batch(client: OpenAI, batch: list[tuple[int, object]]) -> list[dict]:
    """
    Send a batch of articles to DeepSeek R1 for scoring.
    Returns a list of score dicts: {index, lasting, growth, eudaimonia, reason}.
    """
    prompt_parts = ["Score the following articles:\n"]
    for idx, entry in batch:
        prompt_parts.append(build_article_block(idx, entry))

    user_prompt = "\n".join(prompt_parts)

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )
            raw = response.choices[0].message.content.strip()

            # Strip markdown fences if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            scores = json.loads(raw)
            if isinstance(scores, list):
                return scores
            else:
                log.warning("LLM returned non-list JSON; retrying.")
        except json.JSONDecodeError as e:
            log.warning("JSON parse error on attempt %d: %s", attempt + 1, e)
            log.debug("Raw response: %s", raw[:500] if 'raw' in dir() else "(none)")
        except Exception as e:
            log.warning("API error on attempt %d: %s", attempt + 1, e)

        if attempt < 2:
            wait = 2 ** (attempt + 1)
            log.info("Retrying in %ds...", wait)
            time.sleep(wait)

    log.error("Failed to score batch after 3 attempts; skipping.")
    return []


def judge_articles(entries: list) -> list[tuple[object, dict]]:
    """
    Score all entries via the LLM in batches.
    Returns list of (entry, score_dict) tuples that pass the threshold.
    """
    if not DEEPSEEK_API_KEY:
        log.error("DEEPSEEK_API_KEY is not set. Exiting.")
        sys.exit(1)

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    indexed = list(enumerate(entries))
    all_scores: dict[int, dict] = {}

    for batch_start in range(0, len(indexed), BATCH_SIZE):
        batch = indexed[batch_start : batch_start + BATCH_SIZE]
        log.info(
            "Scoring batch %d-%d of %d...",
            batch_start,
            batch_start + len(batch) - 1,
            len(indexed),
        )
        scores = score_batch(client, batch)
        for s in scores:
            try:
                all_scores[int(s["index"])] = s
            except (KeyError, ValueError, TypeError):
                continue

        # Rate-limit courtesy pause between batches
        if batch_start + BATCH_SIZE < len(indexed):
            time.sleep(1)

    # Filter: pass if ANY criterion meets the threshold
    passed = []
    for i, entry in enumerate(entries):
        s = all_scores.get(i)
        if s is None:
            log.debug("No score for entry %d; skipping.", i)
            continue

        lasting = int(s.get("lasting", 0))
        growth = int(s.get("growth", 0))
        eudaimonia = int(s.get("eudaimonia", 0))
        best = max(lasting, growth, eudaimonia)

        title = getattr(entry, "title", "(no title)")
        if best >= SCORE_THRESHOLD:
            log.info(
                "  PASS [%d/%d/%d] %s — %s",
                lasting, growth, eudaimonia, title,
                s.get("reason", ""),
            )
            passed.append((entry, s))
        else:
            log.info("  FAIL [%d/%d/%d] %s", lasting, growth, eudaimonia, title)

    log.info("Passed filter: %d / %d entries", len(passed), len(entries))
    return passed


# ---------------------------------------------------------------------------
# Feed Generation
# ---------------------------------------------------------------------------


def load_existing_feed_entries() -> list[dict]:
    """
    Load entries from the existing output feed so we can merge new results
    with recent previous results (rolling window).
    """
    path = Path(OUTPUT_FILE)
    if not path.exists():
        return []

    feed = feedparser.parse(str(path))
    cutoff = datetime.now(timezone.utc).timestamp() - (FEED_RETENTION_DAYS * 86400)
    kept = []

    for entry in feed.entries:
        # Try to get the published date; skip if too old
        published = getattr(entry, "published_parsed", None)
        if published:
            entry_ts = time.mktime(published)
            if entry_ts < cutoff:
                continue

        kept.append({
            "title": getattr(entry, "title", ""),
            "link": getattr(entry, "link", ""),
            "description": getattr(entry, "summary", ""),
            "published": getattr(entry, "published", ""),
            "categories": [t.term for t in getattr(entry, "tags", [])],
        })

    log.info("Loaded %d existing entries (within retention window).", len(kept))
    return kept


def generate_feed(new_entries: list[tuple[object, dict]], existing: list[dict]) -> None:
    """Build the output RSS feed, merging new entries with existing ones."""
    fg = FeedGenerator()
    fg.title("Eudaimonia Feed")
    fg.link(href="https://github.com/your-username/eudaimonia-filter")
    fg.description(
        "RSS entries filtered for lasting importance, personal growth, "
        "and eudaimonia. Updated 4x daily."
    )
    fg.language("en")
    fg.lastBuildDate(datetime.now(timezone.utc))

    seen_links = set()
    now_str = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S +0000")

    # Add new entries first (most recent at top)
    for entry, scores in new_entries:
        link = getattr(entry, "link", "") or ""
        if link in seen_links:
            continue
        seen_links.add(link)

        fe = fg.add_entry()
        fe.title(getattr(entry, "title", "(no title)"))
        fe.link(href=link)

        summary = getattr(entry, "summary", "") or ""
        reason = scores.get("reason", "")
        lasting = scores.get("lasting", 0)
        growth = scores.get("growth", 0)
        eudaimonia = scores.get("eudaimonia", 0)

        # Append score context to the description
        score_line = (
            f"\n\n[Eudaimonia Filter — Lasting: {lasting}/10 | "
            f"Growth: {growth}/10 | Eudaimonia: {eudaimonia}/10"
            f"{(' — ' + reason) if reason else ''}]"
        )
        fe.description(summary + score_line)

        # Preserve original pub date if available
        pub = getattr(entry, "published", None)
        if pub:
            fe.pubDate(pub)
        else:
            fe.pubDate(now_str)

        # Tag with highest criterion
        best_name = max(
            [("lasting", lasting), ("growth", growth), ("eudaimonia", eudaimonia)],
            key=lambda x: x[1],
        )[0]
        fe.category(term=best_name, label=best_name.title())

    # Merge in existing entries that aren't duplicates
    for item in existing:
        link = item.get("link", "")
        if link in seen_links:
            continue
        seen_links.add(link)

        fe = fg.add_entry()
        fe.title(item.get("title", ""))
        fe.link(href=link)
        fe.description(item.get("description", ""))
        if item.get("published"):
            fe.pubDate(item["published"])
        for cat in item.get("categories", []):
            fe.category(term=cat)

    # Write output
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fg.rss_file(str(output_path))
    log.info("Wrote %d entries to %s", len(seen_links), OUTPUT_FILE)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    log.info("=" * 60)
    log.info("Eudaimonia Filter — run started at %s", datetime.now(timezone.utc).isoformat())
    log.info("=" * 60)

    # Load seen DB
    seen_db = load_seen_db()
    log.info("Seen DB has %d entries.", len(seen_db))

    # Fetch
    all_entries = fetch_entries()
    if not all_entries:
        log.warning("No entries fetched. Exiting.")
        return

    # Filter out already-seen entries
    new_entries = []
    for entry in all_entries:
        eid = _entry_id(entry)
        if eid not in seen_db:
            new_entries.append(entry)
        else:
            log.debug("Skipping already-seen: %s", getattr(entry, "title", "?"))

    log.info("New (unseen) entries: %d / %d", len(new_entries), len(all_entries))

    # Score new entries via LLM
    passed = []
    if new_entries:
        passed = judge_articles(new_entries)
    else:
        log.info("Nothing new to score.")

    # Mark all fetched entries as seen (pass or fail)
    now_iso = datetime.now(timezone.utc).isoformat()
    for entry in new_entries:
        seen_db[_entry_id(entry)] = now_iso

    # Prune old entries from seen DB (older than 2x retention)
    cutoff_iso = datetime.fromtimestamp(
        time.time() - FEED_RETENTION_DAYS * 86400 * 2, tz=timezone.utc
    ).isoformat()
    pruned = {k: v for k, v in seen_db.items() if v > cutoff_iso}
    if len(pruned) < len(seen_db):
        log.info("Pruned %d old entries from seen DB.", len(seen_db) - len(pruned))
    save_seen_db(pruned)

    # Load existing feed entries and merge
    existing = load_existing_feed_entries()
    generate_feed(passed, existing)

    log.info("Done. %d new entries passed the filter.", len(passed))


if __name__ == "__main__":
    main()
