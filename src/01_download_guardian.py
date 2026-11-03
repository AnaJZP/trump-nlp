"""Retrieves the corpus from The Guardian Content API.

Query "Donald Trump", 1 Oct 2025 – 25 Oct 2026, in quarterly blocks to stay
within the API's pagination limits; fields headline, bodyText and trailText.

Usage:  export GUARDIAN_API_KEY=...   (free developer key: open-platform.theguardian.com)
        python src/01_download_guardian.py
"""
import os
import time

import pandas as pd
import requests

from common import CORPUS_CSV, DATA, QUARTERS, QUERY

URL = "https://content.guardianapis.com/search"
API_KEY = os.environ["GUARDIAN_API_KEY"]


def fetch_block(date_from, date_to):
    params = {"api-key": API_KEY, "q": QUERY, "from-date": date_from, "to-date": date_to,
              "page-size": 200, "show-fields": "headline,bodyText,trailText", "order-by": "oldest", "page": 1}
    first = requests.get(URL, params=params, timeout=60)
    first.raise_for_status()
    resp = first.json()["response"]
    rows, pages = list(resp["results"]), resp["pages"]
    print(f"{date_from} → {date_to}: {resp['total']} items, {pages} pages")
    for page in range(2, pages + 1):
        params["page"] = page
        r = requests.get(URL, params=params, timeout=60)
        r.raise_for_status()
        rows.extend(r.json()["response"]["results"])
        time.sleep(0.2)
    return rows


def main():
    DATA.mkdir(exist_ok=True)
    rows = [row for a, b in QUARTERS for row in fetch_block(a, b)]
    df = pd.DataFrame(rows).drop_duplicates(subset="id")
    df["webPublicationDate"] = pd.to_datetime(df["webPublicationDate"], errors="coerce")
    df = df.sort_values("webPublicationDate").reset_index(drop=True)
    df.to_csv(CORPUS_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df):,} records; duplicate IDs: {df['id'].duplicated().sum()}, "
          f"duplicate URLs: {df['webUrl'].duplicated().sum()}")
    print(df["type"].value_counts())


if __name__ == "__main__":
    main()
