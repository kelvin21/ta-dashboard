"""
Update OHLCV data from AmiBroker into MongoDB.

The script keeps the AmiBroker OLE workflow (Analysis.Run + Export) but swaps
the storage layer from SQLite to MongoDB and adds CLI controls for:
  - date range: --from-date YYYY-MM-DD --to-date YYYY-MM-DD
  - symbols: --tickers "AAA,BBB" or --all to pull everything

Usage examples
--------------
# load all symbols from 2020-01-01 to today
python ami_project_to_db.py --from-date 2020-01-01 --all

# load specific tickers and write to a custom MongoDB database/collection
python ami_project_to_db.py --tickers AAPL,MSFT --mongo-db market --collection ohlcv
"""

import argparse
import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Set

import pandas as pd
import pymongo
import win32com.client
from amibroker_apx_editor import modify_analysis

# Optional: load .env like db_adapter.py
try:
    from dotenv import load_dotenv

    SCRIPT_DIR = Path(__file__).parent
    ENV_PATH = SCRIPT_DIR / ".env"
    load_dotenv(dotenv_path=ENV_PATH, verbose=False)
except ImportError:
    # python-dotenv not required; fall back to OS env only
    pass

# Directory constants
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PROJECT_DIR = r"C:\Program Files (x86)\AmiBroker\Projects"
FORMULA_DIR = r"C:\Program Files (x86)\AmiBroker\Formulas\Custom\Screen"
EXPORT_DIR = r"C:\Program Files (x86)\AmiBroker\Export"

# Defaults / Mongo settings (align with db_adapter style)
DEFAULT_PROJECT = "ExportData.apx"

# Optional hardcoded URI (only used if MONGODB_URI env not set)
HARDCODED_MONGODB_URI = ""

# Prefer env, then hardcoded, then localhost
DEFAULT_MONGO_URI = os.getenv("MONGODB_URI") or HARDCODED_MONGODB_URI or "mongodb://localhost:27017"
DEFAULT_DB = os.getenv("MONGODB_DB_NAME", "macd_reversal")
DEFAULT_COLLECTION = os.getenv("MONGODB_OHLCV_COLLECTION", "ohlcv")


class AmiProjectToMongo:
    """Handle AmiBroker export and loading into MongoDB."""

    def __init__(self, mongo_uri=DEFAULT_MONGO_URI, db_name=DEFAULT_DB, collection=DEFAULT_COLLECTION):
        self.mongo_uri = mongo_uri
        self.client = pymongo.MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection]
        os.makedirs(DATA_DIR, exist_ok=True)
        self.ensure_indexes()

    def ensure_indexes(self):
        """Ensure indexes for fast upsert by project/ticker/date."""
        self.collection.create_index(
            [("project", pymongo.ASCENDING), ("ticker", pymongo.ASCENDING), ("date", pymongo.ASCENDING)],
            unique=True,
            name="idx_project_ticker_date",
        )
        self.collection.create_index([("ticker", pymongo.ASCENDING), ("date", pymongo.ASCENDING)], name="idx_ticker_date")

    def create_dummy_project(self, project_path: str) -> bool:
        """Create a simple APX if none exists."""
        print(f"Creating dummy project file: {project_path}")

        from_date = datetime(2010, 1, 1)
        current_date = datetime.now()

        apx_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<ANALYSIS>
    <GENERAL>
        <FORMULAPATH></FORMULAPATH>
        <FORMULANAME>Data Export</FORMULANAME>
        <SYMBOL>*</SYMBOL>
        <SHOWTRADES>0</SHOWTRADES>
        <SHOWSIGNALS>0</SHOWSIGNALS>
        <SHOWRESULTS>1</SHOWRESULTS>
    </GENERAL>
    <PERIODS>
        <FROMDATE>{from_date.strftime('%Y-%m-%d')}</FROMDATE>
        <TODATE>{current_date.strftime('%Y-%m-%d')}</TODATE>
        <FROMTIME>000000</FROMTIME>
        <TOTIME>235959</TOTIME>
        <PERIODICITY>0</PERIODICITY>
        <TIMEFRAME>86400</TIMEFRAME>
    </PERIODS>
    <BACKTEST>
        <INITIALEQUITY>10000000</INITIALEQUITY>
        <COMMISSION>0.15</COMMISSION>
        <MARGIN>100</MARGIN>
        <INTERESTRATE>0</INTERESTRATE>
        <PYRAMIDING>1</PYRAMIDING>
    </BACKTEST>
</ANALYSIS>"""

        try:
            with open(project_path, "w", encoding="utf-8") as f:
                f.write(apx_content)
            print(f"Created dummy project: {project_path}")
            return True
        except Exception as e:  # pragma: no cover - defensive
            print(f"Error creating dummy project: {e}")
            return False

    def ensure_project_exists(self, project_name: str) -> str:
        """Return path to project, creating a dummy one when missing."""
        project_path = os.path.join(PROJECT_DIR, project_name)

        if not os.path.exists(project_path):
            print(f"Project file not found: {project_path}")
            if not self.create_dummy_project(project_path):
                raise FileNotFoundError(f"Could not create project file: {project_path}")

        return project_path

    def update_project_config(self, project_path: str, from_date: datetime, to_date: datetime, symbols: Optional[str]):
        """Update date range and symbol filter inside the temp APX file."""
        kwargs = {"date_range": (from_date, to_date)}
        if symbols:
            # Some versions of modify_analysis may not support symbols; fail-soft.
            kwargs["symbols"] = symbols

        try:
            modify_analysis(project_path, **kwargs)
            print(f"Updated project date range {from_date:%Y-%m-%d} → {to_date:%Y-%m-%d}")
            if symbols:
                print(f"Updated symbols to: {symbols}")
        except TypeError:
            # Fallback for older modify_analysis signatures
            modify_analysis(project_path, date_range=(from_date, to_date))
            print(f"Updated project date range {from_date:%Y-%m-%d} → {to_date:%Y-%m-%d} (symbol edit skipped)")
        except Exception as e:
            print(f"Warning: Could not update project settings: {e}")

    def run_analysis_and_export(self, project_path: str, from_date: datetime, to_date: datetime, symbols: Optional[str]) -> str:
        """Run AmiBroker analysis and export results to CSV via OLE."""
        try:
            ab = win32com.client.Dispatch("Broker.Application")
        except Exception as e:
            print(f"Error connecting to AmiBroker: {e}")
            raise

        temp_dir = tempfile.gettempdir()
        base_name = os.path.splitext(os.path.basename(project_path))[0]
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_apx = os.path.join(temp_dir, f"{base_name}_{date_str}_temp.apx")

        shutil.copy2(project_path, temp_apx)
        self.update_project_config(temp_apx, from_date, to_date, symbols)

        try:
            analysis = ab.AnalysisDocs.Open(temp_apx)
            print("Running analysis...")
            analysis.Run(1)  # 1 = scan

            start_time = datetime.now()
            timeout = timedelta(minutes=10)

            while analysis.IsBusy:
                if (datetime.now() - start_time) > timeout:
                    raise TimeoutError("Analysis took too long to complete")
                print(".", end="", flush=True)
                import time

                time.sleep(1)

            print("\nAnalysis complete")

            export_filename = f"{base_name}_{date_str}_results.csv"
            export_path = os.path.join(EXPORT_DIR, export_filename)
            os.makedirs(EXPORT_DIR, exist_ok=True)

            analysis.Export(export_path, "csv")
            print(f"Exported results to {export_path}")

            return export_path

        finally:
            try:
                analysis.Close()
            except Exception:
                pass
            try:
                os.remove(temp_apx)
            except Exception:
                pass

    def load_export_to_mongo(self, export_path: str, project_name: str, ticker_filter: Optional[Set[str]]) -> int:
        """Load exported CSV data into MongoDB with cleanup and upserts."""
        if not os.path.exists(export_path):
            print(f"Export file not found: {export_path}")
            return 0

        try:
            print(f"📊 Loading CSV file: {export_path}")
            file_size = os.path.getsize(export_path)
            print(f"📁 File size: {file_size / (1024 * 1024):.1f} MB")

            start_time = datetime.now()
            df = pd.read_csv(export_path)
            load_time = (datetime.now() - start_time).total_seconds()
            print(f"✅ CSV loaded in {load_time:.1f}s - {len(df)} rows, columns: {list(df.columns)}")

            if df.empty:
                print("❌ CSV file is empty")
                return 0

            df.columns = [col.lower().strip().replace("/", "_").replace(" ", "_") for col in df.columns]

            column_mapping = {
                "ticker": "ticker",
                "name": "ticker",
                "symbol": "ticker",
                "date_time": "date",
                "datetime": "date",
                "date": "date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
                "vol": "volume",
                "open_interest": "open_interest",
                "openint": "open_interest",
            }

            for old_col, new_col in column_mapping.items():
                if old_col in df.columns and old_col != new_col:
                    df = df.rename(columns={old_col: new_col})

            required_columns = ["ticker", "date"]
            optional_columns = ["open", "high", "low", "close", "volume"]

            missing_required = [col for col in required_columns if col not in df.columns]
            if missing_required:
                print(f"❌ Missing required columns: {missing_required}")
                return 0

            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                invalid_dates = df["date"].isna().sum()
                if invalid_dates > 0:
                    print(f"⚠️ Found {invalid_dates} invalid dates")
            else:
                print("❌ No date column found")
                return 0

            for col in optional_columns:
                if col not in df.columns:
                    df[col] = 0.0 if col != "volume" else 0
            if "open_interest" not in df.columns:
                df["open_interest"] = 0

            df = df.dropna(subset=["ticker", "date"])
            df = df[df["ticker"].astype(str).str.strip() != ""]

            if ticker_filter:
                df = df[df["ticker"].str.upper().isin(ticker_filter)]
                print(f"Filtered to tickers {sorted(ticker_filter)}, remaining rows: {len(df)}")

            cleaned_count = len(df)
            if cleaned_count == 0:
                print("❌ No valid data remaining after cleanup")
                return 0

            print("📋 Sample rows:")
            for _, row in df.head(3).iterrows():
                print(
                    f"  {row['ticker']} | {row['date'].date()} | "
                    f"O:{row.get('open', 'N/A')} H:{row.get('high', 'N/A')} "
                    f"L:{row.get('low', 'N/A')} C:{row.get('close', 'N/A')} V:{row.get('volume', 'N/A')}"
                )

            operations = []
            for idx, row in df.iterrows():
                ticker = str(row["ticker"]).strip().upper()
                doc_date = pd.to_datetime(row["date"]).to_pydatetime()

                operations.append(
                    pymongo.UpdateOne(
                        {"project": project_name, "ticker": ticker, "date": doc_date},
                        {
                            "$set": {
                                "project": project_name,
                                "ticker": ticker,
                                "date": doc_date,
                                "open": float(row["open"]) if pd.notna(row["open"]) else None,
                                "high": float(row["high"]) if pd.notna(row["high"]) else None,
                                "low": float(row["low"]) if pd.notna(row["low"]) else None,
                                "close": float(row["close"]) if pd.notna(row["close"]) else None,
                                "volume": int(row["volume"]) if pd.notna(row["volume"]) else 0,
                                "open_interest": int(row["open_interest"]) if pd.notna(row["open_interest"]) else 0,
                                "exported_at": datetime.utcnow(),
                            }
                        },
                        upsert=True,
                    )
                )

                if (idx + 1) % 1000 == 0:
                    print(f"  📦 Prepared {idx + 1}/{len(df)} docs")

            if not operations:
                print("❌ No operations to send")
                return 0

            print(f"🚀 Writing {len(operations)} records to MongoDB...")
            result = self.collection.bulk_write(operations, ordered=False)
            inserted = result.upserted_count + result.modified_count
            print(f"🎉 MongoDB write complete: {inserted} upserted/updated")

            try:
                os.remove(export_path)
                print("🧹 Removed temporary export file")
            except Exception as e:
                print(f"⚠️ Could not remove export file: {e}")

            return inserted

        except pd.errors.EmptyDataError:
            print("❌ CSV file is empty or corrupted")
            return 0
        except pd.errors.ParserError as e:
            print(f"❌ CSV parsing error: {e}")
            return 0
        except MemoryError:
            print("❌ Not enough memory to load CSV file")
            return 0
        except Exception as e:
            print(f"❌ Unexpected error loading CSV to MongoDB: {e}")
            import traceback

            traceback.print_exc()
            return 0

    def close(self):
        """Close Mongo client."""
        self.client.close()


def parse_date(date_str: Optional[str], default: datetime) -> datetime:
    if not date_str:
        return default
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date format for {date_str}, expected YYYY-MM-DD") from exc


def main():
    parser = argparse.ArgumentParser(
        description="Update OHLCV data from AmiBroker to MongoDB (supports date range and tickers)."
    )
    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT, help=f"Project file name (default: {DEFAULT_PROJECT})")
    parser.add_argument("--mongo-uri", type=str, default=DEFAULT_MONGO_URI, help=f"Mongo connection string (default: {DEFAULT_MONGO_URI})")
    parser.add_argument("--mongo-db", type=str, default=DEFAULT_DB, help=f"Mongo database name (default: {DEFAULT_DB})")
    parser.add_argument("--collection", type=str, default=DEFAULT_COLLECTION, help=f"Mongo collection name (default: {DEFAULT_COLLECTION})")
    parser.add_argument("--from-date", dest="from_date", type=str, help="Start date YYYY-MM-DD (default: earliest)")
    parser.add_argument("--to-date", dest="to_date", type=str, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--tickers", type=str, help='Comma-separated tickers, e.g. "AAPL,MSFT"')
    parser.add_argument("--all", dest="all_tickers", action="store_true", help="Use all tickers (*)")
    parser.add_argument("--no-run", action="store_true", help="Skip AmiBroker run, only parse latest export file")
    parser.add_argument("--export-file", type=str, help="Use an existing CSV export instead of running AmiBroker")

    args = parser.parse_args()

    from_date = parse_date(args.from_date, datetime(2010, 1, 1))
    to_date = parse_date(args.to_date, datetime.now())
    if from_date > to_date:
        parser.error("--from-date cannot be after --to-date")

    ticker_filter = None
    symbol_string = "*"
    if args.tickers:
        tickers_clean = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        ticker_filter = set(tickers_clean)
        symbol_string = ",".join(tickers_clean)
    elif args.all_tickers:
        symbol_string = "*"

    loader = AmiProjectToMongo(args.mongo_uri, args.mongo_db, args.collection)

    try:
        project_name = args.project
        project_path = loader.ensure_project_exists(project_name)
        print(f"📁 Using project file: {project_path}")

        if args.export_file:
            export_path = args.export_file
            print(f"Using provided export file: {export_path}")
        elif args.no_run:
            parser.error("--no-run requires --export-file")
        else:
            print("⚡ Starting AmiBroker analysis...")
            start = datetime.now()
            export_path = loader.run_analysis_and_export(project_path, from_date, to_date, symbol_string)
            print(f"✅ Analysis + export finished in {(datetime.now() - start).total_seconds():.1f}s")

        base_name = os.path.splitext(project_name)[0]
        inserted = loader.load_export_to_mongo(export_path, base_name, ticker_filter)
        print(f"Done. Inserted/updated {inserted} documents into MongoDB collection '{args.collection}'.")

    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
    finally:
        loader.close()


if __name__ == "__main__":
    main()
