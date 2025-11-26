#!/usr/bin/env python
"""Download WRDS OptionMetrics tables for a single date/ticker.

This follows the working flow from `notebooks/data_wrds_old.ipynb`:
1) resolve secid via optionmnames
2) pull opprcd/secprd/vsurfd/stdopd/distrprojd/zerocd/securd
3) save each table to parquet under the raw data directory
"""

import argparse
import os
from pathlib import Path
from typing import List

import pandas as pd
from dotenv import find_dotenv, load_dotenv


def get_data_df(db, secid: float, date: str, label: str) -> pd.DataFrame:
    """Mirror the queries from the old notebook for a specific table label."""
    secid_str = str(secid)
    year = date[:4]
    date_str = f"'{date}'"

    if label in ["opprcd", "secprd", "vsurfd", "stdopd", "distrprojd"]:
        table = f"optionm.{label}{year}"
        return db.raw_sql(f"SELECT * FROM {table} WHERE secid = {secid_str} AND date = {date_str}")
    if label == "zerocd":
        return db.raw_sql(f"SELECT * FROM optionm.{label} WHERE date = {date_str}")
    if label == "securd":
        return db.raw_sql(f"SELECT * FROM optionm.{label} WHERE secid = {secid_str}")
    raise ValueError(f"Invalid label: {label}")


def concat_and_save(dfs: List[pd.DataFrame], out_path: Path, label: str) -> pd.DataFrame:
    """Concatenate the per-secid pulls and persist to parquet."""
    if not dfs:
        raise ValueError(f"No data returned for {label}")
    df = pd.concat(dfs, ignore_index=True)
    df.to_parquet(out_path, engine="pyarrow", compression="snappy")
    print(f"  {label:<18} {len(df):>8} rows -> {out_path}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch WRDS OptionMetrics tables.")
    parser.add_argument("--date", required=True, help="Trade date (YYYY-MM-DD)")
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol")
    parser.add_argument(
        "--raw-dir",
        default=None,
        help="Output directory for parquet files (default: $RAW_DATA_PATH or data/raw)",
    )
    parser.add_argument(
        "--username",
        default=None,
        help="WRDS username (default: $WRDS_USERNAME or WRDS config)",
    )
    args = parser.parse_args()

    load_dotenv(find_dotenv())

    raw_dir = Path(args.raw_dir or os.getenv("RAW_DATA_PATH") or "data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    date_name = args.date.replace("-", "")

    print("=" * 70)
    print(f"WRDS download for {args.symbol} @ {args.date}")
    print(f"Saving parquet to: {raw_dir}")
    print("=" * 70)

    try:
        import wrds
    except ImportError as exc:
        raise SystemExit("wrds package not installed. Run `pip install wrds`.") from exc

    print("Connecting to WRDS...")
    db = wrds.Connection(wrds_username=args.username or os.getenv("WRDS_USERNAME"))
    print("✓ Connected.")

    # Resolve secid
    name_to_secid_df = db.raw_sql(f"SELECT * FROM optionm.optionmnames WHERE ticker = '{args.symbol}'")
    name_to_secid_df = name_to_secid_df.dropna(subset=["optionid"])
    secid_list = name_to_secid_df["secid"].unique().tolist()
    if not secid_list:
        raise SystemExit(f"No secid found for ticker {args.symbol}")
    print(f"secid list: {secid_list}")

    # Pull tables
    op_path = raw_dir / f"{args.symbol}{date_name}_option_price.parquet"
    sec_path = raw_dir / f"{args.symbol}{date_name}_security_price.parquet"
    vsurf_path = raw_dir / f"{args.symbol}{date_name}_volatility_surface.parquet"
    stdop_path = raw_dir / f"{args.symbol}{date_name}_stdoption_price.parquet"
    distr_path = raw_dir / f"{args.symbol}{date_name}_distr_proj.parquet"
    zc_path = raw_dir / f"{date_name}_zero_curve.parquet"
    securd_path = raw_dir / f"{args.symbol}_securd.parquet"

    option_price_df = concat_and_save(
        [get_data_df(db, secid, args.date, "opprcd") for secid in secid_list],
        op_path,
        "option price",
    )
    security_price_df = concat_and_save(
        [get_data_df(db, secid, args.date, "secprd") for secid in secid_list],
        sec_path,
        "security price",
    )
    concat_and_save(
        [get_data_df(db, secid, args.date, "vsurfd") for secid in secid_list],
        vsurf_path,
        "vol surface",
    )
    concat_and_save(
        [get_data_df(db, secid, args.date, "stdopd") for secid in secid_list],
        stdop_path,
        "std option px",
    )
    concat_and_save(
        [get_data_df(db, secid, args.date, "distrprojd") for secid in secid_list],
        distr_path,
        "distr proj",
    )

    zero_curve_df = get_data_df(db, secid_list[0], args.date, "zerocd")
    zero_curve_df.to_parquet(zc_path, engine="pyarrow", compression="snappy")
    print(f"  zero curve         {len(zero_curve_df):>8} rows -> {zc_path}")

    concat_and_save(
        [get_data_df(db, secid, args.date, "securd") for secid in secid_list],
        securd_path,
        "securd",
    )

    # Simple spot printout for sanity
    try:
        spot_close = float(security_price_df["close"].iloc[0])
        print(f"\nSpot (close) on {args.date}: {spot_close:.4f}")
    except Exception:
        pass

    db.close()
    print("\nAll tables downloaded.")


if __name__ == "__main__":
    main()
