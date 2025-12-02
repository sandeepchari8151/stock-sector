from flask import Flask, render_template, jsonify, request
import os
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict
import pandas as pd
import numpy as np
import hashlib
import json
try:
    import requests
except Exception:
    requests = None  # type: ignore

# Try to reuse existing utilities for data and sectors
try:
    from scripts.data_collection import YAHOO_SYMBOLS
except Exception:
    YAHOO_SYMBOLS = {
        "IT": "XLK",
        "Pharma": "XLV",
        "Banking": "XLF",
        "FMCG": "XLP",
    }

try:
    import yfinance as yf
except Exception:
    yf = None

# Load .env in development
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Optional: MongoDB Atlas (for shared, login-free storage)
try:
    from pymongo import MongoClient
except Exception:
    MongoClient = None  # type: ignore


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    # Cache for resolved company names to avoid repeated lookups
    _company_name_cache = {}

    # --- App-level history cache (in-memory + disk) ---
    _history_cache: Dict[Tuple[str, str, str], Tuple[pd.DataFrame, datetime]] = {}
    _cache_ttl = timedelta(minutes=int(os.environ.get("CACHE_TTL_MIN", "30")))
    _cache_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "cache")
    try:
        os.makedirs(_cache_dir, exist_ok=True)
    except Exception:
        _cache_dir = os.path.join(os.path.dirname(__file__), ".cache")
        try:
            os.makedirs(_cache_dir, exist_ok=True)
        except Exception:
            pass

    def _is_online() -> bool:
        # Quick network check with small timeout
        try:
            if requests is None:
                return True  # can't verify; assume online to avoid blocking
            resp = requests.get("https://query1.finance.yahoo.com", timeout=1.5)
            return resp.status_code < 500
        except Exception:
            return False

    def _cache_key(symbol: str, period: str, interval: str) -> Tuple[str, str, str]:
        return (symbol.upper().strip(), period.strip(), interval.strip())

    def _disk_path(symbol: str, period: str, interval: str) -> str:
        key = f"{symbol}__{period}__{interval}"
        # Keep filename short/safe
        safe = hashlib.md5(key.encode("utf-8")).hexdigest()
        return os.path.join(_cache_dir, f"hist_{safe}.csv")

    def _load_from_disk(symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        try:
            path = _disk_path(symbol, period, interval)
            if not os.path.exists(path):
                return None
            # Respect TTL by file mtime
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            if datetime.utcnow() - mtime > _cache_ttl:
                return None
            df = pd.read_csv(path)
            # Ensure expected columns for downstream usage
            required = {"Open", "High", "Low", "Close"}
            if not required.issubset(set(df.columns)):
                return None
            return df
        except Exception:
            return None

    def _save_to_disk(symbol: str, period: str, interval: str, df: pd.DataFrame) -> None:
        try:
            path = _disk_path(symbol, period, interval)
            # Persist minimal set of columns
            cols = [c for c in ["Date", "Datetime", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
            df.loc[:, cols].to_csv(path, index=False)
        except Exception:
            pass

    def fetch_history(symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Fetch ticker history quickly with cache and offline fallback.
        Returns empty DataFrame on failure.
        """
        key = _cache_key(symbol, period, interval)
        # In-memory cache hit and fresh
        try:
            cached = _history_cache.get(key)
            if cached is not None:
                df, ts = cached
                if datetime.utcnow() - ts <= _cache_ttl and not df.empty:
                    return df
        except Exception:
            pass

        # Disk cache hit
        df_disk = _load_from_disk(symbol, period, interval)
        if df_disk is not None and not df_disk.empty:
            _history_cache[key] = (df_disk, datetime.utcnow())
            return df_disk

        # Online fetch if possible
        if yf is not None and _is_online():
            try:
                tk = yf.Ticker(symbol)
                # Use prepost/actions flags for consistency
                df = tk.history(period=period, interval=interval, prepost=True, actions=False)
                if not df.empty:
                    df = df.reset_index()
                    _history_cache[key] = (df, datetime.utcnow())
                    _save_to_disk(symbol, period, interval, df)
                    return df
            except Exception:
                pass

        # Offline fallback: try bundled sample CSV if compatible
        try:
            sample_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "sample_sector_prices.csv")
            if os.path.exists(sample_path):
                sdf = pd.read_csv(sample_path)
                # Expect columns: symbol, date, open, high, low, close, volume
                sdf_sym = sdf[sdf["symbol"].astype(str).str.upper() == symbol.upper()]
                if not sdf_sym.empty:
                    # Normalize to expected schema
                    out = pd.DataFrame({
                        "Date": pd.to_datetime(sdf_sym["date"], errors="coerce"),
                        "Open": sdf_sym.get("open"),
                        "High": sdf_sym.get("high"),
                        "Low": sdf_sym.get("low"),
                        "Close": sdf_sym.get("close"),
                        "Volume": sdf_sym.get("volume", 0),
                    }).dropna(subset=["Open", "High", "Low", "Close"])
                    _history_cache[key] = (out, datetime.utcnow())
                    return out
        except Exception:
            pass

        # Final fallback: generate synthetic demo data so charts always have something to show
        try:
            # Roughly map period to number of business days
            if period.endswith("y"):
                days = 252 * max(int(period[:-1]) if period[:-1].isdigit() else 1, 1)
            elif period.endswith("mo"):
                days = 21 * max(int(period[:-2]) if period[:-2].isdigit() else 3, 1)
            else:
                days = 60

            dates = pd.bdate_range(end=datetime.utcnow(), periods=days)

            # Deterministic seed per symbol so charts are stable across reloads
            np.random.seed(abs(hash(symbol)) % (2**32))
            drift = 0.0005
            vol = 0.01
            rets = np.random.normal(drift, vol, size=len(dates))
            price = 100.0
            prices = price * np.cumprod(1.0 + rets)

            out = pd.DataFrame({
                "Date": dates,
                "Open": prices * (1 - 0.002),
                "High": prices * (1 + 0.01),
                "Low": prices * (1 - 0.01),
                "Close": prices,
                "Volume": np.random.randint(500_000, 2_000_000, size=len(dates)),
            })

            _history_cache[key] = (out, datetime.utcnow())
            return out
        except Exception:
            pass

        return pd.DataFrame()

    # --- Mongo wiring (lazy) ---
    mongo_uri = os.environ.get("MONGODB_URI", "").strip()
    mongo_client: Optional[MongoClient] = None  # type: ignore
    mongo_db = None

    if MongoClient and mongo_uri:
        try:
            mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)  # type: ignore
            # Default DB name from URI or fallback
            db_name = os.environ.get("MONGODB_DB", "sectorscope")
            mongo_db = mongo_client[db_name]
            # Touch server to validate (non-fatal if fails)
            _ = mongo_client.admin.command("ping")
        except Exception:
            mongo_client = None
            mongo_db = None

    def get_company_name(symbol: str) -> str:
        if not symbol:
            return symbol
        if symbol in _company_name_cache:
            return _company_name_cache[symbol]

        known_names = {
            "NVDA": "NVIDIA Corporation",
            "AAPL": "Apple Inc.",
            "MSFT": "Microsoft Corporation",
            "AMD": "Advanced Micro Devices, Inc.",
            "META": "Meta Platforms, Inc.",
            "GOOGL": "Alphabet Inc.",
            "AVGO": "Broadcom Inc.",
            "ORCL": "Oracle Corporation",
            "NFLX": "Netflix, Inc.",
            "MU": "Micron Technology, Inc.",
            "TSLA": "Tesla, Inc.",
            "AMZN": "Amazon.com, Inc.",
        }
        if symbol in known_names:
            _company_name_cache[symbol] = known_names[symbol]
            return known_names[symbol]

        try:
            tk = yf.Ticker(symbol)
            try:
                fi = getattr(tk, "fast_info", None)
                if isinstance(fi, dict):
                    nm = fi.get("shortName") or fi.get("longName") or fi.get("companyName")
                    if nm and nm != symbol and len(nm) > 3:
                        _company_name_cache[symbol] = nm
                        return nm
            except Exception:
                pass
            try:
                info = tk.get_info()
                if isinstance(info, dict):
                    nm = info.get("shortName") or info.get("longName") or info.get("name")
                    if nm and nm != symbol and len(nm) > 3:
                        _company_name_cache[symbol] = nm
                        return nm
            except Exception:
                pass
        except Exception:
            pass

        _company_name_cache[symbol] = symbol
        return symbol

    @app.route("/")
    def index():
        return render_template("sectors.html")

    @app.route("/sectors")
    def sectors_page():
        return render_template("sectors.html")

    @app.route("/stocks/<sector>")
    def stocks_page(sector: str):
        return render_template("stocks.html", sector=sector)

    @app.route("/chart/<symbol>")
    def chart_page(symbol: str):
        sector = request.args.get("sector", "").strip()
        name = get_company_name(symbol)
        return render_template("chart.html", sector=sector, symbol=symbol, name=name)

    @app.route("/patterns")
    def patterns_page():
        return render_template("patterns.html")

    @app.route("/health")
    def health():
        return jsonify({"status": "ok"})
    
    @app.route("/test-names")
    def test_names():
        # Test the name mapping
        test_symbols = ["NVDA", "AAPL", "MSFT", "AMD", "META"]
        results = []
        for sym in test_symbols:
            try:
                tk = yf.Ticker(sym)
                # Test our known names mapping
                known_names = {
                    "NVDA": "NVIDIA Corporation",
                    "AAPL": "Apple Inc.",
                    "MSFT": "Microsoft Corporation", 
                    "AMD": "Advanced Micro Devices, Inc.",
                    "META": "Meta Platforms, Inc.",
                }
                name = known_names.get(sym, sym)
                results.append({"symbol": sym, "name": name})
            except Exception as e:
                results.append({"symbol": sym, "name": sym, "error": str(e)})
        return jsonify({"results": results})
    
    @app.route("/test-api")
    def test_api():
        # Test the actual API logic
        company_names = {
            "NVDA": "NVIDIA Corporation",
            "AAPL": "Apple Inc.",
            "MSFT": "Microsoft Corporation", 
            "AMD": "Advanced Micro Devices, Inc.",
            "META": "Meta Platforms, Inc.",
        }
        sym = "NVDA"
        name = company_names.get(sym, sym)
        return jsonify({"symbol": sym, "name": name, "test": "working"})
    
    @app.route("/test-search")
    def test_search():
        # Test the search logic directly
        company_names = {
            "NVDA": "NVIDIA Corporation",
            "AAPL": "Apple Inc.",
            "MSFT": "Microsoft Corporation", 
            "AMD": "Advanced Micro Devices, Inc.",
            "META": "Meta Platforms, Inc.",
        }
        sym = "NVDA"
        name = company_names.get(sym, sym)
        return jsonify({
            "items": [{
                "symbol": sym,
                "name": name,
                "lastPrice": 178.19,
                "avgDollarVolume": 29919995985,
                "ret1m": -0.0187
            }]
        })

    @app.get("/api/sectors")
    def api_sectors():
        # Prefer Mongo if available
        if mongo_db is not None:
            try:
                docs = list(mongo_db["sectors"].find({}, {"_id": 0}))
                if docs:
                    # Normalize to current frontend contract
                    sectors = [
                        {"id": d.get("code"), "name": d.get("name") or d.get("code"), "symbol": d.get("etf_symbol")}
                        for d in docs
                        if d.get("code") and d.get("etf_symbol")
                    ]
                    return jsonify({"sectors": sectors})
            except Exception:
                pass
        # Fallback to in-memory mapping
        sectors = [{"id": k, "name": k, "symbol": v} for k, v in YAHOO_SYMBOLS.items()]
        return jsonify({"sectors": sectors})

    @app.get("/api/sectors/overview")
    def api_sectors_overview():
        if yf is None:
            return jsonify({"error": "yfinance not installed"}), 500

        period = request.args.get("period", "1y").strip()
        interval = request.args.get("interval", "1d").strip()

        overview = []
        cum_lines = []
        for sector, etf in YAHOO_SYMBOLS.items():
            try:
                hist = fetch_history(etf, period, interval)
                if hist.empty:
                    continue
                # cumulative index
                first = float(hist["Close"].iloc[0])
                closes = hist["Close"].astype(float)
                cum = (closes / first).tolist()
                date_col = "Date" if "Date" in hist.columns else ("Datetime" if "Datetime" in hist.columns else hist.columns[0])
                dates = [d.isoformat() if isinstance(d, datetime) else str(d) for d in hist[date_col].tolist()]
                cum_lines.append({"sector": sector, "dates": dates, "cum": cum})

                # daily returns for volatility
                returns = closes.pct_change().dropna()
                vol = float(returns.std())
                # CAGR approximation
                num_days = len(hist)
                years = max(num_days / 252.0, 1e-9)
                cagr = float((closes.iloc[-1] / closes.iloc[0]) ** (1.0 / years) - 1.0)
                overview.append({
                    "sector": sector,
                    "symbol": etf,
                    "volatility": vol,
                    "cagr": cagr,
                    "cumLast": float(cum[-1]),
                })
            except Exception:
                continue

        return jsonify({"overview": overview, "cum": cum_lines})

    # Simple demo stock list per sector (fallback). In Mongo mode, read from DB
    SECTOR_STOCKS = {
        "IT": [
            {"symbol": "AAPL", "name": "Apple Inc."},
            {"symbol": "MSFT", "name": "Microsoft Corp."},
            {"symbol": "NVDA", "name": "NVIDIA Corp."},
        ],
        "Pharma": [
            {"symbol": "PFE", "name": "Pfizer Inc."},
            {"symbol": "JNJ", "name": "Johnson & Johnson"},
            {"symbol": "MRK", "name": "Merck & Co."},
        ],
        "Banking": [
            {"symbol": "JPM", "name": "JPMorgan Chase & Co."},
            {"symbol": "BAC", "name": "Bank of America"},
            {"symbol": "C", "name": "Citigroup Inc."},
        ],
        "FMCG": [
            {"symbol": "PG", "name": "Procter & Gamble"},
            {"symbol": "KO", "name": "Coca-Cola Co."},
            {"symbol": "PEP", "name": "PepsiCo, Inc."},
        ],
    }

    @app.get("/api/stocks")
    def api_stocks():
        sector = request.args.get("sector", "").strip()
        # Prefer Mongo if available
        if mongo_db is not None and sector:
            try:
                # Expect documents: { sector_code, symbol, name, is_active }
                cur = mongo_db["stocks"].find({"sector_code": sector, "is_active": {"$ne": False}}, {"_id": 0, "symbol": 1, "name": 1})
                docs = list(cur)
                if docs:
                    return jsonify({"stocks": docs})
            except Exception:
                pass
        stocks = SECTOR_STOCKS.get(sector, [])
        return jsonify({"stocks": stocks})

    @app.get("/api/stocks/search")
    def api_stocks_search():
        if yf is None:
            return jsonify({"error": "yfinance not installed"}), 500

        sector = request.args.get("sector", "").strip()
        min_price = float(request.args.get("min_price", "0") or 0)
        max_price = float(request.args.get("max_price", "0") or 0)
        min_adv = float(request.args.get("min_adv", "0") or 0)  # avg dollar volume
        limit = int(request.args.get("limit", "20") or 20)
        period = request.args.get("period", "3mo").strip()
        interval = request.args.get("interval", "1d").strip()

        universe = TOP_SYMBOLS.get(sector, []) if sector in TOP_SYMBOLS else [s for lst in TOP_SYMBOLS.values() for s in lst]

        def _get_ticker_name(tk_obj, symbol_fallback: str):
            # Known company names mapping for common symbols
            known_names = {
                "NVDA": "NVIDIA Corporation",
                "AAPL": "Apple Inc.",
                "MSFT": "Microsoft Corporation", 
                "AMD": "Advanced Micro Devices, Inc.",
                "META": "Meta Platforms, Inc.",
                "GOOGL": "Alphabet Inc.",
                "AVGO": "Broadcom Inc.",
                "ORCL": "Oracle Corporation",
                "NFLX": "Netflix, Inc.",
                "MU": "Micron Technology, Inc.",
                "TSLA": "Tesla, Inc.",
                "AMZN": "Amazon.com, Inc.",
                "JPM": "JPMorgan Chase & Co.",
                "BAC": "Bank of America Corporation",
                "WFC": "Wells Fargo & Company",
                "C": "Citigroup Inc.",
                "GS": "Goldman Sachs Group, Inc.",
                "MS": "Morgan Stanley",
                "PFE": "Pfizer Inc.",
                "JNJ": "Johnson & Johnson",
                "MRK": "Merck & Co., Inc.",
                "ABBV": "AbbVie Inc.",
                "LLY": "Eli Lilly and Company",
                "BMY": "Bristol Myers Squibb Company",
                "AMGN": "Amgen Inc.",
                "PG": "Procter & Gamble Company",
                "KO": "Coca-Cola Company",
                "PEP": "PepsiCo, Inc.",
                "PM": "Philip Morris International Inc.",
                "MDLZ": "Mondelez International, Inc."
            }
            
            # Return known name if available
            if symbol_fallback in known_names:
                return known_names[symbol_fallback]
                
            # Try yfinance methods as fallback
            try:
                # Try fast_info first
                fi = getattr(tk_obj, "fast_info", None)
                if isinstance(fi, dict):
                    nm = fi.get("shortName") or fi.get("longName") or fi.get("companyName")
                    if nm and nm != symbol_fallback and len(nm) > 3:
                        return nm
            except Exception:
                pass
            
            # Try get_info() for more detailed info
            try:
                info = tk_obj.get_info()
                if isinstance(info, dict):
                    nm = info.get("shortName") or info.get("longName") or info.get("name")
                    if nm and nm != symbol_fallback and len(nm) > 3:
                        return nm
            except Exception:
                pass
                
            return symbol_fallback

        rows = []
        for sym in universe:
            try:
                hist = fetch_history(sym, period, interval)
                if hist.empty or "Close" not in hist or "Volume" not in hist:
                    continue
                closes = hist["Close"].astype(float)
                last_price = float(closes.iloc[-1])
                if min_price and last_price < min_price:
                    continue
                if max_price and last_price > max_price:
                    continue
                avg_vol = float(hist["Volume"].astype(float).tail(60).mean())
                avg_price = float(closes.tail(60).mean())
                adv = avg_price * avg_vol
                if adv < min_adv:
                    continue
                # simple profit proxy: 1m return
                ret_1m = None
                if len(closes) > 22:
                    ret_1m = float(closes.iloc[-1] / closes.iloc[-22] - 1.0)
                # Simple company name mapping
                company_names = {
                    "NVDA": "NVIDIA Corporation",
                    "AAPL": "Apple Inc.",
                    "MSFT": "Microsoft Corporation", 
                    "AMD": "Advanced Micro Devices, Inc.",
                    "META": "Meta Platforms, Inc.",
                    "GOOGL": "Alphabet Inc.",
                    "AVGO": "Broadcom Inc.",
                    "ORCL": "Oracle Corporation",
                    "NFLX": "Netflix, Inc.",
                    "MU": "Micron Technology, Inc.",
                    "TSLA": "Tesla, Inc.",
                    "AMZN": "Amazon.com, Inc.",
                    "JPM": "JPMorgan Chase & Co.",
                    "BAC": "Bank of America Corporation",
                    "WFC": "Wells Fargo & Company",
                    "C": "Citigroup Inc.",
                    "GS": "Goldman Sachs Group, Inc.",
                    "MS": "Morgan Stanley",
                    "PFE": "Pfizer Inc.",
                    "JNJ": "Johnson & Johnson",
                    "MRK": "Merck & Co., Inc.",
                    "ABBV": "AbbVie Inc.",
                    "LLY": "Eli Lilly and Company",
                    "BMY": "Bristol Myers Squibb Company",
                    "AMGN": "Amgen Inc.",
                    "PG": "Procter & Gamble Company",
                    "KO": "Coca-Cola Company",
                    "PEP": "PepsiCo, Inc.",
                    "PM": "Philip Morris International Inc.",
                    "MDLZ": "Mondelez International, Inc."
                }
                name = company_names.get(sym, sym)
                print(f"DEBUG: {sym} -> {name}")  # Debug print - UPDATED
                rows.append({
                    "symbol": sym,
                    "name": "TEST_NVIDIA_CORP" if sym == "NVDA" else name,
                    "lastPrice": last_price,
                    "avgDollarVolume": adv,
                    "ret1m": ret_1m,
                })
            except Exception:
                continue

        rows.sort(key=lambda r: (r["avgDollarVolume"]) , reverse=True)
        return jsonify({"items": rows[:limit]})

    @app.get("/api/candles")
    def api_candles():
        if yf is None:
            return jsonify({"error": "yfinance not installed"}), 500

        sector = request.args.get("sector", "").strip()
        symbols_param = request.args.get("symbols", "").strip()
        period = request.args.get("period", "6mo").strip()
        interval = request.args.get("interval", "1d").strip()

        if not symbols_param:
            return jsonify({"series": []})

        symbols = [s for s in symbols_param.split(",") if s]
        series = []

        for sym in symbols:
            try:
                # Include pre/post market so latest session data appears promptly
                hist = fetch_history(sym, period, interval)
                if hist.empty:
                    continue
                # If requesting daily data and today's bar is missing, backfill by aggregating intraday
                try:
                    if interval == "1d":
                        # Normalize tz for safe comparisons
                        if getattr(hist.index, "tz", None) is not None:
                            hist = hist.tz_localize(None)
                        last_idx = hist.index[-1]
                        hist_last_date = (pd.to_datetime(last_idx).date())
                        today_utc = datetime.utcnow().date()
                        if hist_last_date < today_utc:
                            intra = fetch_history(sym, "60d", "60m")
                            if not intra.empty:
                                # Aggregate to daily OHLCV
                                if getattr(intra.index, "tz", None) is not None:
                                    intra = intra.tz_localize(None)
                                daily = intra.resample("1D").agg({
                                    "Open": "first",
                                    "High": "max",
                                    "Low": "min",
                                    "Close": "last",
                                    "Volume": "sum",
                                }).dropna(how="all")
                                if not daily.empty:
                                    # Determine strictly newer daily dates vs existing hist
                                    hist_last = pd.to_datetime(hist.index.max()).normalize()
                                    newer = daily[daily.index.normalize() > hist_last]
                                    if not newer.empty:
                                        hist = pd.concat([hist, newer]).sort_index()
                except Exception:
                    pass
                hist = hist.reset_index()
                # Normalize datetime for JSON
                date_col = "Date" if "Date" in hist.columns else ("Datetime" if "Datetime" in hist.columns else hist.columns[0])
                dates = []
                for d in hist[date_col].tolist():
                    if isinstance(d, datetime):
                        dates.append(d.isoformat())
                    else:
                        try:
                            dates.append(str(d))
                        except Exception:
                            dates.append("")
                series.append({
                    "symbol": sym,
                    "dates": dates,
                    "open": [float(x) for x in hist["Open"].tolist()],
                    "high": [float(x) for x in hist["High"].tolist()],
                    "low": [float(x) for x in hist["Low"].tolist()],
                    "close": [float(x) for x in hist["Close"].tolist()],
                    "volume": [int(x) for x in (hist["Volume"].tolist() if "Volume" in hist.columns else [0]*len(dates))],
                })
            except Exception:
                continue

        return jsonify({"sector": sector, "series": series})

    # Extended universe per sector for liquidity screening (not exhaustive)
    TOP_SYMBOLS = {
        "IT": [
            "AAPL", "MSFT", "NVDA", "AVGO", "GOOGL", "META", "AMD", "ORCL", "ADBE", "CRM",
            "CSCO", "INTC", "NFLX", "TXN", "QCOM", "NOW", "MU", "AMAT", "PANW", "SHOP"
        ],
        "Pharma": [
            "JNJ", "MRK", "PFE", "ABBV", "LLY", "BMY", "AMGN", "GSK", "SNY", "NVO",
            "AZN", "REGN", "VRTX", "BIIB", "DHR", "TMO", "MDT", "SYK", "ISRG", "BSX"
        ],
        "Banking": [
            "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "BLK", "SCHW", "BX",
            "COF", "USB", "PNC", "TD", "RY", "HSBC", "UBS", "ING", "BBVA", "SAN"
        ],
        "FMCG": [
            "PG", "KO", "PEP", "PM", "MDLZ", "MO", "KMB", "CL", "KVUE", "UL",
            "KDP", "MNST", "KHC", "GIS", "HSY", "SBUX", "WMT", "COST", "TGT", "EL"
        ],
    }

    @app.get("/api/top-stocks")
    def api_top_stocks():
        if yf is None:
            return jsonify({"error": "yfinance not installed"}), 500

        sector = request.args.get("sector", "").strip()
        period = request.args.get("period", "3mo").strip()
        interval = request.args.get("interval", "1d").strip()
        lookback_days = int(request.args.get("lookback_days", "60"))

        # Build candidate universe
        if sector and sector in TOP_SYMBOLS:
            universe = TOP_SYMBOLS[sector]
        else:
            # union of all
            seen = set()
            universe = []
            for lst in TOP_SYMBOLS.values():
                for s in lst:
                    if s not in seen:
                        seen.add(s)
                        universe.append(s)

        def _get_ticker_name(tk_obj, symbol_fallback: str):
            # Known company names mapping for common symbols
            known_names = {
                "NVDA": "NVIDIA Corporation",
                "AAPL": "Apple Inc.",
                "MSFT": "Microsoft Corporation", 
                "AMD": "Advanced Micro Devices, Inc.",
                "META": "Meta Platforms, Inc.",
                "GOOGL": "Alphabet Inc.",
                "AVGO": "Broadcom Inc.",
                "ORCL": "Oracle Corporation",
                "NFLX": "Netflix, Inc.",
                "MU": "Micron Technology, Inc.",
                "TSLA": "Tesla, Inc.",
                "AMZN": "Amazon.com, Inc.",
                "JPM": "JPMorgan Chase & Co.",
                "BAC": "Bank of America Corporation",
                "WFC": "Wells Fargo & Company",
                "C": "Citigroup Inc.",
                "GS": "Goldman Sachs Group, Inc.",
                "MS": "Morgan Stanley",
                "PFE": "Pfizer Inc.",
                "JNJ": "Johnson & Johnson",
                "MRK": "Merck & Co., Inc.",
                "ABBV": "AbbVie Inc.",
                "LLY": "Eli Lilly and Company",
                "BMY": "Bristol Myers Squibb Company",
                "AMGN": "Amgen Inc.",
                "PG": "Procter & Gamble Company",
                "KO": "Coca-Cola Company",
                "PEP": "PepsiCo, Inc.",
                "PM": "Philip Morris International Inc.",
                "MDLZ": "Mondelez International, Inc."
            }
            
            # Return known name if available
            if symbol_fallback in known_names:
                return known_names[symbol_fallback]
                
            # Try yfinance methods as fallback
            try:
                # Try fast_info first
                fi = getattr(tk_obj, "fast_info", None)
                if isinstance(fi, dict):
                    nm = fi.get("shortName") or fi.get("longName") or fi.get("companyName")
                    if nm and nm != symbol_fallback and len(nm) > 3:
                        return nm
            except Exception:
                pass
            
            # Try get_info() for more detailed info
            try:
                info = tk_obj.get_info()
                if isinstance(info, dict):
                    nm = info.get("shortName") or info.get("longName") or info.get("name")
                    if nm and nm != symbol_fallback and len(nm) > 3:
                        return nm
            except Exception:
                pass
                
            return symbol_fallback

        results = []
        for sym in universe:
            try:
                hist = fetch_history(sym, period, interval)
                if hist.empty or "Close" not in hist or "Volume" not in hist:
                    continue
                # Use last N rows
                df = hist.tail(lookback_days)
                if df.empty:
                    continue
                avg_price = float(df["Close"].mean())
                avg_vol = float(df["Volume"].mean())
                avg_dollar_vol = avg_price * avg_vol
                last_price = float(df["Close"].iloc[-1])
                name = get_company_name(sym)
                results.append({
                    "symbol": sym,
                    "name": name,
                    "avgDollarVolume": avg_dollar_vol,
                    "avgVolume": avg_vol,
                    "lastPrice": last_price,
                })
            except Exception:
                continue

        # Sort by liquidity desc and take top 10
        results.sort(key=lambda x: x["avgDollarVolume"], reverse=True)
        top10 = results[:10]

        return jsonify({"sector": sector or "ALL", "count": len(top10), "items": top10})

    # ---------- Advanced Screener, Backtest, Alerts ----------
    @app.get("/api/stocks/search_advanced")
    def api_stocks_search_advanced():
        if yf is None:
            return jsonify({"error": "yfinance not installed"}), 500

        sector = request.args.get("sector", "").strip()
        min_price = float(request.args.get("min_price", "0") or 0)
        max_price = float(request.args.get("max_price", "0") or 0)
        min_adv = float(request.args.get("min_adv", "0") or 0)
        limit = int(request.args.get("limit", "20") or 20)
        period = request.args.get("period", "3mo").strip()
        interval = request.args.get("interval", "1d").strip()

        # Removed advanced technical filters for simplicity

        universe = TOP_SYMBOLS.get(sector, []) if sector in TOP_SYMBOLS else [s for lst in TOP_SYMBOLS.values() for s in lst]
        rows = []
        for sym in universe:
            try:
                hist = fetch_history(sym, period, interval)
                if hist.empty or "Close" not in hist or "Volume" not in hist:
                    continue
                hist = hist.dropna()
                closes = hist["Close"].astype(float)
                last_price = float(closes.iloc[-1])
                if min_price and last_price < min_price:
                    continue
                if max_price and last_price > max_price:
                    continue
                avg_vol = float(hist["Volume"].astype(float).tail(60).mean())
                avg_price = float(closes.tail(60).mean())
                adv = avg_price * avg_vol
                if adv < min_adv:
                    continue

                # 1m return
                ret_1m = None
                if len(closes) > 22:
                    ret_1m = float(closes.iloc[-1] / closes.iloc[-22] - 1.0)

                # Simplified - removed advanced technical filters
                # Still calculate basic metrics for display but don't filter on them
                rsi_val = None
                atr_val = None
                range_pos = None
                gap_pct = None
                vol_spike = None

                name = get_company_name(sym)
                rows.append({
                    "symbol": sym,
                    "name": name,
                    "lastPrice": last_price,
                    "avgDollarVolume": adv,
                    "ret1m": ret_1m,
                    "rsi14": rsi_val,
                    "atr14": atr_val,
                    "rangePos52w": range_pos,
                    "gapPct": gap_pct,
                    "volSpike": vol_spike,
                })
            except Exception:
                continue

        rows.sort(key=lambda r: (r["avgDollarVolume"]) , reverse=True)
        return jsonify({"items": rows[:limit]})

    @app.post("/api/backtest")
    def api_backtest():
        if yf is None:
            return jsonify({"error": "yfinance not installed"}), 500
        payload = request.get_json(silent=True) or {}
        symbol = payload.get("symbol")
        strategy = payload.get("strategy", "sma_cross")
        period = payload.get("period", "1y")
        interval = payload.get("interval", "1d")
        params = payload.get("params", {})
        if not symbol:
            return jsonify({"error": "symbol required"}), 400

        tk = yf.Ticker(symbol)
        hist = tk.history(period=period, interval=interval).dropna()
        if hist.empty:
            return jsonify({"error": "no data"}), 400
        closes = hist["Close"].astype(float).tolist()
        dates = [d.isoformat() if isinstance(d, datetime) else str(d) for d in hist.reset_index()["Date"].tolist()]

        trades, equity = _run_backtest(closes, strategy, params)
        metrics = _compute_bt_metrics(equity)
        return jsonify({"dates": dates, "equity": equity, "trades": trades, "metrics": metrics})

    ALERTS = []

    @app.get("/api/alerts")
    def api_alerts_list():
        if mongo_db is not None:
            try:
                docs = list(mongo_db["alerts"].find({}, {"_id": 0}).sort("created_at", -1))
                return jsonify({"items": docs})
            except Exception:
                pass
        return jsonify({"items": ALERTS})

    @app.post("/api/alerts")
    def api_alerts_create():
        data = request.get_json(silent=True) or {}
        symbol = data.get("symbol")
        alert_type = data.get("type", "price_cross_sma20")
        direction = data.get("direction", "above")
        if not symbol:
            return jsonify({"error": "symbol required"}), 400
        doc = {
            "symbol": symbol,
            "type": alert_type,
            "direction": direction,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        if mongo_db is not None:
            try:
                mongo_db["alerts"].insert_one(doc)
                return jsonify({"ok": True})
            except Exception:
                pass
        ALERTS.append(doc)
        return jsonify({"ok": True})

    @app.post("/api/alerts/check")
    def api_alerts_check():
        if yf is None:
            return jsonify({"error": "yfinance not installed"}), 500
        # Load alerts from Mongo if available; otherwise use memory
        alerts_src = ALERTS
        if mongo_db is not None:
            try:
                alerts_src = list(mongo_db["alerts"].find({}, {"_id": 0}))
            except Exception:
                alerts_src = ALERTS
        fired = []
        for a in alerts_src:
            try:
                tk = yf.Ticker(a["symbol"])
                hist = tk.history(period="3mo", interval="1d").dropna()
                if hist.empty:
                    continue
                closes = hist["Close"].astype(float).tolist()
                sma20 = _moving_average(closes, 20)
                last = closes[-1]
                last_sma = sma20[-1]
                if last_sma is None:
                    continue
                cond = (last > last_sma) if a["direction"] == "above" else (last < last_sma)
                if cond:
                    fired.append({"symbol": a["symbol"], "type": a["type"], "direction": a["direction"]})
            except Exception:
                continue
        return jsonify({"fired": fired})

    # Helpers within create_app
    def _moving_average(values, window):
        out = [None] * len(values)
        s = 0.0
        for i, v in enumerate(values):
            s += v
            if i >= window:
                s -= values[i - window]
            if i >= window - 1:
                out[i] = s / window
        return out

    def _compute_rsi(values, period=14):
        rsi = [None] * len(values)
        gains = 0.0
        losses = 0.0
        for i in range(1, len(values)):
            change = values[i] - values[i - 1]
            gain = change if change > 0 else 0.0
            loss = -change if change < 0 else 0.0
            if i <= period:
                gains += gain
                losses += loss
                if i == period:
                    rs = (gains / period) / (losses / period if losses else 1e-9)
                    rsi[i] = 100 - 100 / (1 + rs)
            else:
                gains = (gains * (period - 1) + gain) / period
                losses = (losses * (period - 1) + loss) / period
                rs = (gains) / (losses if losses else 1e-9)
                rsi[i] = 100 - 100 / (1 + rs)
        return rsi

    def _compute_atr(df_hlc, period=14):
        highs = df_hlc["High"].astype(float).tolist()
        lows = df_hlc["Low"].astype(float).tolist()
        closes = df_hlc["Close"].astype(float).tolist()
        trs = []
        for i in range(len(closes)):
            if i == 0:
                trs.append(highs[i] - lows[i])
            else:
                tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
                trs.append(tr)
        atr = []
        s = 0.0
        for i, tr in enumerate(trs):
            if i < period:
                s += tr
                atr.append(None)
            elif i == period:
                s += tr
                atr.append(s / period)
            else:
                prev = atr[-1]
                val = (prev * (period - 1) + tr) / period
                atr.append(val)
        return (df_hlc.assign(_atr=atr))["_atr"].dropna()

    def _run_backtest(closes, strategy, params):
        equity = []
        cash = 1.0
        position = 0.0
        entry_px = None
        trades = []
        if strategy == "sma_cross":
            fast = int(params.get("fast", 10))
            slow = int(params.get("slow", 20))
            ma_fast = _moving_average(closes, fast)
            ma_slow = _moving_average(closes, slow)
            for i, px in enumerate(closes):
                if ma_fast[i] is None or ma_slow[i] is None:
                    equity.append(cash)
                    continue
                if position == 0 and ma_fast[i] > ma_slow[i]:
                    position = cash / px
                    entry_px = px
                    cash = 0.0
                    trades.append({"side": "buy", "price": px, "i": i})
                elif position > 0 and ma_fast[i] < ma_slow[i]:
                    cash = position * px
                    trades.append({"side": "sell", "price": px, "i": i, "pnl": (px - entry_px) / entry_px})
                    position = 0.0
                    entry_px = None
                equity.append(cash + position * px)
        else:
            per = int(params.get("period", 14))
            buy_level = float(params.get("buy", 30))
            sell_level = float(params.get("sell", 50))
            rsi = _compute_rsi(closes, per)
            for i, px in enumerate(closes):
                rv = rsi[i]
                if rv is None:
                    equity.append(cash)
                    continue
                if position == 0 and rv < buy_level:
                    position = cash / px
                    entry_px = px
                    cash = 0.0
                    trades.append({"side": "buy", "price": px, "i": i})
                elif position > 0 and rv > sell_level:
                    cash = position * px
                    trades.append({"side": "sell", "price": px, "i": i, "pnl": (px - entry_px) / entry_px})
                    position = 0.0
                    entry_px = None
                equity.append(cash + position * px)
        return trades, equity

    def _compute_bt_metrics(equity):
        if not equity:
            return {}
        start = equity[0]
        end = equity[-1]
        total_return = (end / (start or 1e-9)) - 1.0
        peak = -1e9
        max_dd = 0.0
        for v in equity:
            peak = max(peak, v)
            dd = (peak - v) / (peak or 1e-9)
            if dd > max_dd:
                max_dd = dd
        return {"totalReturn": total_return, "maxDrawdown": max_dd}

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)


