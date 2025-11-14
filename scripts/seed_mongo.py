import os
from typing import Dict, List

# Load .env if present so MONGODB_URI/MONGODB_DB are picked up automatically
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

try:
    from pymongo import MongoClient
except Exception as exc:
    raise SystemExit("pymongo not installed. Add it to requirements and pip install.")


def get_db():
    uri = os.environ.get("MONGODB_URI", "").strip()
    if not uri:
        raise SystemExit("Set MONGODB_URI env var (or put it in .env) to your MongoDB Atlas connection string")
    db_name = os.environ.get("MONGODB_DB", "sectorscope").strip() or "sectorscope"
    
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        # Validate connection
        client.admin.command("ping")
        return client[db_name]
    except Exception as e:
        raise SystemExit(f"Failed to connect to MongoDB: {e}")


def upsert_sectors(db, sectors: List[Dict]):
    col = db["sectors"]
    for s in sectors:
        col.update_one({"code": s["code"]}, {"$set": s}, upsert=True)


def upsert_stocks(db, stocks: List[Dict]):
    col = db["stocks"]
    for st in stocks:
        col.update_one({"symbol": st["symbol"]}, {"$set": st}, upsert=True)


def main():
    db = get_db()

    sectors = [
        {"code": "IT", "name": "Information Technology", "etf_symbol": "XLK"},
        {"code": "Pharma", "name": "Pharmaceuticals & Healthcare", "etf_symbol": "XLV"},
        {"code": "Banking", "name": "Financials & Banking", "etf_symbol": "XLF"},
        {"code": "FMCG", "name": "Consumer Staples (FMCG)", "etf_symbol": "XLP"},
    ]

    stocks = [
        {"sector_code": "IT", "symbol": "AAPL", "name": "Apple Inc.", "is_active": True},
        {"sector_code": "IT", "symbol": "MSFT", "name": "Microsoft Corporation", "is_active": True},
        {"sector_code": "IT", "symbol": "NVDA", "name": "NVIDIA Corporation", "is_active": True},

        {"sector_code": "Pharma", "symbol": "PFE", "name": "Pfizer Inc.", "is_active": True},
        {"sector_code": "Pharma", "symbol": "JNJ", "name": "Johnson & Johnson", "is_active": True},
        {"sector_code": "Pharma", "symbol": "MRK", "name": "Merck & Co., Inc.", "is_active": True},

        {"sector_code": "Banking", "symbol": "JPM", "name": "JPMorgan Chase & Co.", "is_active": True},
        {"sector_code": "Banking", "symbol": "BAC", "name": "Bank of America Corporation", "is_active": True},
        {"sector_code": "Banking", "symbol": "C", "name": "Citigroup Inc.", "is_active": True},

        {"sector_code": "FMCG", "symbol": "PG", "name": "Procter & Gamble Company", "is_active": True},
        {"sector_code": "FMCG", "symbol": "KO", "name": "The Coca-Cola Company", "is_active": True},
        {"sector_code": "FMCG", "symbol": "PEP", "name": "PepsiCo, Inc.", "is_active": True},
    ]

    upsert_sectors(db, sectors)
    upsert_stocks(db, stocks)
    print("Mongo seed complete: sectors and stocks upserted.")


if __name__ == "__main__":
    main()


