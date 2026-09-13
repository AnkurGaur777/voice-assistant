"""
Local Jarvis - Interaction Analytics CSV Exporter

Exports recorded interactions from the SQLite interactions table (analysis/interactions.db)
to a standardized CSV file (analysis/interactions_export.csv) for easy analysis in
Jupyter / pandas notebooks.

Usage via CLI:
    python -m src.analytics.export_csv
    python -m src.analytics.export_csv --db custom_path.db --output custom_export.csv
"""

import argparse
import csv
from pathlib import Path
import sqlite3
import sys
from typing import Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "analysis" / "interactions.db"
DEFAULT_CSV_PATH = PROJECT_ROOT / "analysis" / "interactions_export.csv"

CSV_COLUMNS = [
    "id",
    "timestamp",
    "user_text",
    "assistant_response",
    "tools_called",
    "stt_seconds",
    "brain_seconds",
    "tts_seconds",
    "total_seconds",
    "success",
    "trigger_mode",
    "is_test",
]


def export_to_csv(
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
    output_csv_path: Union[str, Path] = DEFAULT_CSV_PATH,
    exclude_tests: bool = False,
) -> int:
    """
    Dumps rows from the interactions table in the specified SQLite database
    to a CSV file.

    :param db_path: Path to the SQLite database.
    :param output_csv_path: Target CSV export destination.
    :param exclude_tests: If True, only exports real interactions (where is_test=0).
    :return: Number of rows successfully exported.
    """
    db_file = Path(db_path).resolve()
    csv_file = Path(output_csv_path).resolve()

    if not db_file.exists():
        print(f"[Analytics Export Warning] Database file not found: {db_file}")
        return 0

    csv_file.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_file), timeout=10.0)
    try:
        cursor = conn.cursor()

        # Verify table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='interactions';"
        )
        if not cursor.fetchone():
            print(f"[Analytics Export Warning] Table 'interactions' not found in: {db_file}")
            return 0

        cursor.execute("PRAGMA table_info(interactions);")
        existing_cols = {col[1] for col in cursor.fetchall()}
        has_is_test = "is_test" in existing_cols

        if has_is_test:
            query = """
            SELECT
                id,
                timestamp,
                user_text,
                assistant_response,
                tools_called,
                stt_seconds,
                brain_seconds,
                tts_seconds,
                total_seconds,
                success,
                trigger_mode,
                is_test
            FROM interactions
            """
            if exclude_tests:
                query += " WHERE is_test = 0"
            query += " ORDER BY id ASC;"
            cursor.execute(query)
            rows = cursor.fetchall()
            header = CSV_COLUMNS
        else:
            cursor.execute(
                """
                SELECT
                    id,
                    timestamp,
                    user_text,
                    assistant_response,
                    tools_called,
                    stt_seconds,
                    brain_seconds,
                    tts_seconds,
                    total_seconds,
                    success,
                    trigger_mode
                FROM interactions
                ORDER BY id ASC;
                """
            )
            raw_rows = cursor.fetchall()
            # Append 0 for is_test if column did not exist
            rows = [list(r) + [0] for r in raw_rows]
            header = CSV_COLUMNS
    finally:
        conn.close()

    with open(str(csv_file), mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)
        writer.writerows(rows)

    filter_note = " (excluding automated test runs)" if exclude_tests else ""
    print(f"[Analytics Export] Exported {len(rows)} interaction records{filter_note} to {csv_file}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Local Jarvis interaction logs from SQLite to CSV.",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=str(DEFAULT_DB_PATH),
        help=f"Path to interactions.db (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_CSV_PATH),
        help=f"Path to output CSV (default: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument(
        "--exclude-tests",
        action="store_true",
        help="Exclude automated unit test rows (where is_test=1) from export",
    )
    args = parser.parse_args()

    count = export_to_csv(
        db_path=args.db,
        output_csv_path=args.output,
        exclude_tests=args.exclude_tests,
    )
    print(f"Done. {count} rows exported.")


if __name__ == "__main__":
    main()
