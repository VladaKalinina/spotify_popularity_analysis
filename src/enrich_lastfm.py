"""Обогащение репрезентативной выборки исполнителей Spotify данными Last.fm API.

Запустите этот скрипт перед аналитическим ноутбуком. API-ключ считывается из
переменной окружения ``LASTFM_API_KEY`` и не должен попадать в Git.

Пример
------
python src/enrich_lastfm.py \
    --input data/raw/dataset.csv \
    --output data/processed/artists_lastfm.csv \
    --sample-size 3000
"""

from __future__ import annotations

import argparse
import ast
import logging
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv
from requests import Session
from requests.exceptions import RequestException

BASE_URL = "https://ws.audioscrobbler.com/2.0/"
DEFAULT_REQUEST_INTERVAL = 0.25
LOGGER = logging.getLogger(__name__)


def extract_primary_artist(value: object) -> str | None:
    """Возвращает первого исполнителя из значения ``artists`` набора Spotify."""
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    if text.startswith(("[", "(")) and text.endswith(("]", ")")):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)) and parsed:
                text = str(parsed[0])
        except (ValueError, SyntaxError):
            pass

    artist = text.split(";")[0].strip().strip("[]'\"").strip()
    return artist or None


def normalize_artist_name(value: object) -> str | None:
    """Создаёт стабильный ключ объединения, сохраняя отображаемое имя."""
    artist = extract_primary_artist(value)
    if artist is None:
        return None
    return " ".join(artist.casefold().split())


def select_representative_artists(
    tracks: pd.DataFrame,
    sample_size: int,
) -> pd.DataFrame:
    """Отбирает популярных исполнителей с сохранением жанрового разнообразия.

    Отбор детерминирован. Сначала лучшие исполнители выбираются отдельно
    внутри каждого жанра. Оставшиеся места заполняются из общего рейтинга.
    Рейтинг учитывает максимальную популярность в Spotify, охват жанров и
    количество треков в исходном наборе данных.
    """
    required_columns = {"artists", "track_genre", "popularity"}
    missing_columns = required_columns.difference(tracks.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Input dataset is missing required columns: {missing}")
    if sample_size <= 0:
        raise ValueError("sample_size must be a positive integer")

    working = tracks.copy()
    if "track_id" in working.columns:
        working = working.drop_duplicates(subset="track_id")

    working["artist_name"] = working["artists"].map(extract_primary_artist)
    working["artist_key"] = working["artist_name"].map(normalize_artist_name)
    working["popularity"] = pd.to_numeric(
        working["popularity"], errors="coerce"
    )
    working = working.dropna(
        subset=["artist_name", "artist_key", "track_genre", "popularity"]
    )

    artist_summary = (
        working.groupby(["artist_key", "artist_name"], as_index=False)
        .agg(
            max_spotify_popularity=("popularity", "max"),
            mean_spotify_popularity=("popularity", "mean"),
            track_count=("popularity", "size"),
            genre_count=("track_genre", "nunique"),
        )
        .sort_values(
            [
                "max_spotify_popularity",
                "genre_count",
                "track_count",
                "artist_name",
            ],
            ascending=[False, False, False, True],
        )
    )

    genre_ranked = (
        working.sort_values(
            ["track_genre", "popularity", "artist_name"],
            ascending=[True, False, True],
        )
        .drop_duplicates(subset=["track_genre", "artist_key"])
    )
    genres = sorted(genre_ranked["track_genre"].unique())
    base_quota, remainder = divmod(sample_size, max(len(genres), 1))

    genre_parts = []
    for genre_position, genre in enumerate(genres):
        quota = base_quota + int(genre_position < remainder)
        if quota == 0:
            continue
        genre_parts.append(
            genre_ranked.loc[genre_ranked["track_genre"] == genre].head(quota)
        )

    if genre_parts:
        genre_candidates = pd.concat(genre_parts, ignore_index=True)
        selected_keys = set(genre_candidates["artist_key"])
    else:
        selected_keys = set()

    selected = artist_summary[
        artist_summary["artist_key"].isin(selected_keys)
    ]
    selected = selected.sort_values(
        [
            "max_spotify_popularity",
            "genre_count",
            "track_count",
            "artist_name",
        ],
        ascending=[False, False, False, True],
    )

    if len(selected) < sample_size:
        additional = artist_summary[
            ~artist_summary["artist_key"].isin(selected_keys)
        ].head(sample_size - len(selected))
        selected = pd.concat([selected, additional], ignore_index=True)

    selected = selected.head(sample_size).copy()

    genre_map = (
        working[working["artist_key"].isin(selected["artist_key"])]
        .groupby("artist_key")["track_genre"]
        .agg(lambda values: ", ".join(sorted(set(values))))
    )
    selected["source_genres"] = selected["artist_key"].map(genre_map)
    selected.insert(0, "sample_rank", range(1, len(selected) + 1))
    return selected.reset_index(drop=True)


def get_artist_info(
    session: Session,
    artist_name: str,
    api_key: str,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Запрашивает слушателей, прослушивания и главные теги исполнителя."""
    params = {
        "method": "artist.getinfo",
        "artist": artist_name,
        "api_key": api_key,
        "autocorrect": 1,
        "format": "json",
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(
                BASE_URL,
                params=params,
                timeout=15,
            )
            response.raise_for_status()
            payload = response.json()

            if "error" in payload:
                return {
                    "listeners": None,
                    "playcount": None,
                    "tags": None,
                    "lastfm_name": None,
                    "api_status": f"lastfm_error_{payload['error']}",
                }

            artist = payload.get("artist")
            if not artist:
                return {
                    "listeners": None,
                    "playcount": None,
                    "tags": None,
                    "lastfm_name": None,
                    "api_status": "artist_not_found",
                }

            stats = artist.get("stats", {})
            tags_raw = artist.get("tags", {}).get("tag", [])
            tags = ", ".join(
                tag.get("name", "")
                for tag in tags_raw[:3]
                if tag.get("name")
            ) or None

            return {
                "listeners": pd.to_numeric(
                    stats.get("listeners"), errors="coerce"
                ),
                "playcount": pd.to_numeric(
                    stats.get("playcount"), errors="coerce"
                ),
                "tags": tags,
                "lastfm_name": artist.get("name"),
                "api_status": "ok",
            }
        except (RequestException, ValueError) as error:
            if attempt == max_retries:
                LOGGER.warning(
                    "Request failed for %s after %s attempts: %s",
                    artist_name,
                    max_retries,
                    error,
                )
                return {
                    "listeners": None,
                    "playcount": None,
                    "tags": None,
                    "lastfm_name": None,
                    "api_status": "request_failed",
                }
            time.sleep(2 ** (attempt - 1))

    raise RuntimeError("Unreachable retry state")


def enrich_with_lastfm(
    input_csv: Path,
    output_csv: Path,
    sample_size: int = 3000,
    request_interval: float = DEFAULT_REQUEST_INTERVAL,
) -> pd.DataFrame:
    """Создаёт репрезентативную выборку и обогащает её данными Last.fm.

    Если существует ранее созданный выходной файл, значения API повторно
    используются по ключу ``artist_key``. Также поддерживается переход с
    исходного формата CSV из четырёх столбцов. Неудачные сетевые запросы
    повторяются при следующем запуске, а постоянные ошибки Last.fm сохраняются,
    чтобы не запрашивать неизвестных исполнителей многократно.
    """
    load_dotenv()
    api_key = os.getenv("LASTFM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "LASTFM_API_KEY is not set. Copy .env.example to .env "
            "and add your own Last.fm API key."
        )

    LOGGER.info("Loading source dataset: %s", input_csv)
    tracks = pd.read_csv(input_csv, index_col=0)
    sample = select_representative_artists(tracks, sample_size)
    LOGGER.info(
        "Selected %s artists across %s source genres",
        len(sample),
        tracks["track_genre"].nunique(),
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    api_columns = [
        "listeners",
        "playcount",
        "tags",
        "lastfm_name",
        "api_status",
    ]

    if output_csv.exists():
        existing = pd.read_csv(output_csv)
        if "artist_key" not in existing.columns:
            if "artist_name" not in existing.columns:
                raise ValueError(
                    "Existing output has neither artist_key nor artist_name."
                )
            existing["artist_key"] = existing["artist_name"].map(
                normalize_artist_name
            )

        for column in api_columns:
            if column not in existing.columns:
                existing[column] = None

        legacy_success = (
            existing["listeners"].notna()
            | existing["playcount"].notna()
            | existing["tags"].notna()
        )
        existing.loc[
            existing["api_status"].isna() & legacy_success,
            "api_status",
        ] = "ok"

        existing_api = (
            existing[["artist_key", *api_columns]]
            .drop_duplicates(subset="artist_key", keep="last")
        )
        result = sample.merge(existing_api, on="artist_key", how="left")
        reused = result["api_status"].notna().sum()
        LOGGER.info("Reused %s existing API records", reused)
    else:
        result = sample.copy()
        for column in api_columns:
            result[column] = None

    retryable_statuses = {None, "", "request_failed"}
    pending_mask = result["api_status"].map(
        lambda value: pd.isna(value) or value in retryable_statuses
    )
    pending_indices = result.index[pending_mask].tolist()

    with requests.Session() as session:
        session.headers.update(
            {"User-Agent": "spotify-popularity-course-project/1.0"}
        )

        for position, row_index in enumerate(pending_indices, start=1):
            artist_name = str(result.at[row_index, "artist_name"])
            api_data = get_artist_info(
                session=session,
                artist_name=artist_name,
                api_key=api_key,
            )
            for column, value in api_data.items():
                result.at[row_index, column] = value

            if position % 50 == 0 or position == len(pending_indices):
                LOGGER.info(
                    "Processed %s/%s pending artists",
                    position,
                    len(pending_indices),
                )
                result.sort_values("sample_rank").to_csv(
                    output_csv,
                    index=False,
                )

            time.sleep(request_interval)

    result = result.sort_values("sample_rank").reset_index(drop=True)
    result.to_csv(output_csv, index=False)

    successful = int((result["api_status"] == "ok").sum())
    LOGGER.info(
        "Saved %s rows to %s; successful API matches: %s",
        len(result),
        output_csv,
        successful,
    )
    return result

def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Enrich a representative artist sample through Last.fm."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/dataset.csv"),
        help="Path to the Kaggle Spotify tracks CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/artists_lastfm.csv"),
        help="Path for the enriched CSV.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=3000,
        help="Number of artists to enrich (default: 3000).",
    )
    parser.add_argument(
        "--request-interval",
        type=float,
        default=DEFAULT_REQUEST_INTERVAL,
        help="Seconds between requests (default: 0.25).",
    )
    return parser.parse_args()


def main() -> None:
    """Точка входа интерфейса командной строки."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    enrich_with_lastfm(
        input_csv=args.input,
        output_csv=args.output,
        sample_size=args.sample_size,
        request_interval=args.request_interval,
    )


if __name__ == "__main__":
    main()
