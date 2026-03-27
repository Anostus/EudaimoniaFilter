# Eudaimonia RSS Filter

An LLM-powered RSS filter that keeps only articles worth your attention. Runs automatically 4× daily via GitHub Actions, using DeepSeek R1 to score every entry against three philosophical criteria. An article passes if it scores highly on **any one** of them.

## The Three Criteria

| Criterion | Question |
|---|---|
| **Lasting Importance** | Will this still matter in 10 years? Or is it a pseudo-event that will be forgotten in weeks? |
| **Personal Growth** | Will reading this make me a better person — morally, intellectually, practically? |
| **Eudaimonia** | Does this contribute to human flourishing, purpose, and a life well-lived? |

## How It Works

1. **Fetch** — Pulls entries from your configured RSS feeds.
2. **Deduplicate** — Skips entries already scored in previous runs (tracked in `data/seen.json`).
3. **Score** — Sends entries to DeepSeek R1 in batches. Each entry gets three independent 1–10 scores.
4. **Filter** — Keeps entries where *any single score* meets the threshold (default: 7/10).
5. **Merge** — Combines new results with the existing feed (rolling 7-day window).
6. **Publish** — Writes `docs/feed.xml`, which you can subscribe to in any RSS reader.

## Setup

### 1. Fork or clone this repo

### 2. Add your DeepSeek API key

Go to **Settings → Secrets and variables → Actions → Secrets** and add:

- `DEEPSEEK_API_KEY` — your DeepSeek API key

### 3. Configure your feeds

Go to **Settings → Secrets and variables → Actions → Variables** and optionally add:

| Variable | Default | Description |
|---|---|---|
| `RSS_FEEDS` | Guardian (US & World), PBS NewsHour, ProPublica | JSON array of RSS feed URLs |
| `SCORE_THRESHOLD` | `7` | Minimum score (1–10) on any single criterion to pass |
| `MAX_ENTRIES_PER_FEED` | `25` | Max entries to pull from each feed per run |
| `BATCH_SIZE` | `10` | Articles per LLM API call |
| `FEED_RETENTION_DAYS` | `7` | Days to keep entries in the output feed |

### 4. Enable GitHub Pages (optional)

To get a public URL for your feed:

1. Go to **Settings → Pages**
2. Set source to **Deploy from a branch**
3. Select the `main` branch and `/docs` folder
4. Your feed will be available at `https://<username>.github.io/<repo>/feed.xml`

### 5. Subscribe

Add the feed URL to your RSS reader of choice (NetNewsWire, Feedly, Miniflux, etc.).

## Running Locally

```bash
export DEEPSEEK_API_KEY="your-key-here"
pip install -r requirements.txt
python eudaimonia_filter.py
```

The default feeds (Guardian US & World, PBS NewsHour, ProPublica) are built into the script. To override them:

```bash
export RSS_FEEDS='["https://example.com/feed", "https://other.com/rss"]'
python eudaimonia_filter.py
```

## Cost

DeepSeek R1 is inexpensive. With the defaults (25 entries/feed, batches of 10), each run processes roughly 2–5 batches depending on how many feeds you have and how many entries are new. At 4 runs/day this typically costs well under $1/month for a handful of feeds.

## Project Structure

```
├── .github/workflows/eudaimonia.yml   # GitHub Actions schedule
├── eudaimonia_filter.py               # Main script
├── requirements.txt                   # Python dependencies
├── data/seen.json                     # Tracks already-scored entries (auto-generated)
└── docs/feed.xml                      # Output RSS feed (auto-generated)
```
