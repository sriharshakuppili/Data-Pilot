# app.py
"""
DataPilot — Generic Text→SQL + Dashboard prototype (Streamlit)
- Multi-format ingest: CSV/TSV/Excel/JSON/Parquet/SQL
- Optional Google Sheets (public CSV export)
- Schema introspection
- Optional OpenAI NL→SQL engine (temperature=0)
- Local NL→SQL fallback (generic)
- Safe SQL execution (SELECT-only)
- Quick Dashboard (generic heuristics)
"""

import streamlit as st
import pandas as pd
import sqlite3
import re
import os
import json
from datetime import datetime
import plotly.express as px

# Optional: load OpenAI if API available
try:
    from openai import OpenAI
except:
    OpenAI = None

from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="DataPilot — Text→SQL", layout="wide")
st.title("🛩️ DataPilot — Text→SQL & Quick Dashboard")
st.caption("Upload any data → Ask NL → Get SQL → Execute safely → Build simple dashboards.")

# ─────────────────────────────────────────────────────────────
# INIT OPENAI
# ─────────────────────────────────────────────────────────────
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = None
if OPENAI_KEY and OpenAI is not None:
    try:
        client = OpenAI()
    except:
        client = None

# ─────────────────────────────────────────────────────────────
# SAFETY
# ─────────────────────────────────────────────────────────────
FORBIDDEN = ["insert","update","delete","drop","alter","create","truncate","replace","attach","detach"]

def is_destructive(sql: str) -> bool:
    if not sql:
        return False
    low = sql.lower()
    return any(re.search(rf"\b{kw}\b", low) for kw in FORBIDDEN)

def extract_sql_codeblock(text: str) -> str:
    m = re.search(r"```sql(.*?)```", text, re.S)
    return m.group(1).strip() if m else text.strip()

# ─────────────────────────────────────────────────────────────
# SQLITE IN-MEMORY DB
# ─────────────────────────────────────────────────────────────
conn = sqlite3.connect(":memory:", check_same_thread=False)
cursor = conn.cursor()

# ─────────────────────────────────────────────────────────────
# FILE LOADERS
# ─────────────────────────────────────────────────────────────
def load_file(uploaded):
    name = uploaded.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded)
        if name.endswith(".tsv") or name.endswith(".txt"):
            return pd.read_csv(uploaded, sep="\t")
        if name.endswith(".xlsx") or name.endswith(".xls"):
            return pd.read_excel(uploaded)
        if name.endswith(".json"):
            return pd.read_json(uploaded)
        if name.endswith(".parquet"):
            import pyarrow.parquet as pq
            return pq.read_table(uploaded).to_pandas()
        if name.endswith(".sql"):
            return uploaded.read().decode("utf-8")
    except Exception as e:
        st.error(f"Error reading {uploaded.name}: {e}")
    return None

def load_google_sheet(url):
    if not url:
        return None
    try:
        if "docs.google.com" in url and "/edit" in url:
            url = url.replace("/edit", "/export?format=csv")
        return pd.read_csv(url)
    except Exception as e:
        st.error(f"Google Sheet load failed: {e}")
        return None

# ─────────────────────────────────────────────────────────────
# UPLOAD UI
# ─────────────────────────────────────────────────────────────
st.header("1) Upload your data")
uploads = st.file_uploader("Upload CSV/Excel/TSV/JSON/Parquet/SQL", accept_multiple_files=True)
sheet_url = st.text_input("Or paste a public Google Sheets URL")

tables = []

# Google Sheet
if sheet_url:
    df = load_google_sheet(sheet_url)
    if df is not None:
        df.to_sql("google_sheet", conn, if_exists="replace", index=False)
        st.success(f"Loaded google_sheet ({len(df)} rows)")
        tables.append("google_sheet")

# Files
if uploads:
    for f in uploads:
        data = load_file(f)
        if data is None:
            continue
        if isinstance(data, str):
            if is_destructive(data):
                st.error(f"SQL dump {f.name} contains destructive operations. Blocked.")
                continue
            try:
                cursor.executescript(data)
                st.success(f"Executed SQL dump: {f.name}")
            except Exception as e:
                st.error(f"SQL dump failed: {e}")
            continue

        # Assign table name
        base = re.sub(r"\W+","_", f.name.split(".")[0]).lower()
        tab = base
        i = 1
        while cursor.execute("SELECT name FROM sqlite_master WHERE name=?", (tab,)).fetchone():
            i += 1
            tab = f"{base}_{i}"

        data.to_sql(tab, conn, if_exists="replace", index=False)
        tables.append(tab)
        st.success(f"Loaded {f.name} → {tab} ({len(data)} rows)")

# Show schema
with st.expander("View loaded tables & schema"):
    rows = cursor.execute("SELECT name FROM sqlite_master").fetchall()
    if not rows:
        st.write("No tables loaded.")
    else:
        for (t,) in rows:
            st.write(f"### {t}")
            cols = cursor.execute(f"PRAGMA table_info({t})").fetchall()
            st.dataframe(pd.DataFrame(cols, columns=["cid","name","type","notnull","default","pk"]))

# ─────────────────────────────────────────────────────────────
# BUILD SCHEMA SUMMARY FOR PROMPTS
# ─────────────────────────────────────────────────────────────
def build_schema_text():
    out = []
    rows = cursor.execute("SELECT name FROM sqlite_master").fetchall()
    for (t,) in rows:
        cols = [c[1] for c in cursor.execute(f"PRAGMA table_info({t})").fetchall()]
        out.append(f"{t}: {', '.join(cols)}")
    return "\n".join(out)

# ─────────────────────────────────────────────────────────────
# GENERIC LOCAL NL→SQL (SAFE)
# ─────────────────────────────────────────────────────────────
def local_nl_to_sql(nl):
    text = nl.lower()

    # Find any table
    t = cursor.execute("SELECT name FROM sqlite_master").fetchone()
    if not t:
        return ""
    t = t[0]

    # preview request
    if any(k in text for k in ["show", "preview", "list rows", "sample", "head"]):
        return f"SELECT * FROM {t} LIMIT 100;"

    # simple top N
    m = re.search(r"top\s*(\d+)", text)
    if m:
        n = int(m.group(1))
        cols = [c[1] for c in cursor.execute(f"PRAGMA table_info({t})").fetchall()]
        if cols:
            return f"SELECT {cols[0]}, COUNT(*) AS cnt FROM {t} GROUP BY {cols[0]} ORDER BY cnt DESC LIMIT {n};"

    # basic numeric average
    df = pd.read_sql_query(f"SELECT * FROM {t} LIMIT 500", conn)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if "average" in text or "avg" in text:
        if num_cols:
            return f"SELECT AVG({num_cols[0]}) AS avg_value FROM {t};"

    # fallback
    return f"SELECT * FROM {t} LIMIT 100;"

# ─────────────────────────────────────────────────────────────
# 2) NATURAL LANGUAGE → SQL
# ─────────────────────────────────────────────────────────────
st.header("2) Natural language → SQL")
nl = st.text_area("Ask a question (plain English)")

use_api = st.checkbox("Use OpenAI API (if available)", value=True)

if st.button("Generate SQL"):
    if not nl.strip():
        st.warning("Enter a question first.")
    else:
        sql = None

        # API path
        if use_api and client is not None:
            schema_text = build_schema_text()
            prompt = f"""
You are an expert SQL generator. Return ONLY one valid SELECT statement in a ```sql``` block.
NO DDL. NO INSERT/UPDATE/DELETE.
Database dialect: SQLite.

SCHEMA:
{schema_text}

QUESTION:
{nl}
"""
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"user","content":prompt}],
                    temperature=0
                )
                content = resp.choices[0].message["content"]
                sql = extract_sql_codeblock(content)
            except Exception as e:
                st.error(f"API failed: {e}. Falling back to local generator.")
                sql = local_nl_to_sql(nl)

        # local fallback
        else:
            sql = local_nl_to_sql(nl)

        # safety
        if is_destructive(sql):
            st.error("Generated SQL is unsafe (DDL/DML detected). Blocking.")
        else:
            st.success("Generated SQL:")
            st.code(sql, language="sql")
            st.session_state["last_sql"] = sql
st.header("3) Run SQL (execute last generated SELECT)")

col_run, col_save = st.columns([1, 1])
with col_run:
    if st.button("Run last generated SQL"):
        sql = st.session_state.get("last_sql", "")
        if not sql:
            st.error("No SQL available. Generate SQL first.")
        else:
            if is_destructive(sql):
                st.error("SQL appears destructive — refusing to run.")
            else:
                try:
                    df = pd.read_sql_query(sql, conn)
                    st.success(f"Query returned {len(df)} rows.")
                    st.dataframe(df)
                    st.download_button("Download results (CSV)", data=df.to_csv(index=False).encode("utf-8"),
                                       file_name="query_results.csv", mime="text/csv")
                    # store in history
                    if "history" not in st.session_state:
                        st.session_state["history"] = []
                    st.session_state["history"].append({
                        "ts": datetime.utcnow().isoformat(),
                        "nl": nl,
                        "sql": sql,
                        "rows": len(df)
                    })
                except Exception as e:
                    st.error(f"Failed to execute SQL: {e}")

with col_save:
    if st.button("Save last SQL to history"):
        sql = st.session_state.get("last_sql", "")
        if not sql:
            st.warning("No SQL to save.")
        else:
            if "history" not in st.session_state:
                st.session_state["history"] = []
            st.session_state["history"].append({
                "ts": datetime.utcnow().isoformat(),
                "nl": nl,
                "sql": sql,
                "rows": None
            })
            st.success("Saved to history.")

# -------------------------
# 4) Quick Dashboard (local heuristics)
# -------------------------
st.header("4) Quick Dashboard (local heuristics)")

def quick_dashboard():
    tables = [r[0] for r in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    if not tables:
        st.info("No tables loaded to build a dashboard.")
        return
    # prefer table that has at least one numeric and one date-like column
    chosen = None
    for t in tables:
        cols = cursor.execute(f"PRAGMA table_info({t})").fetchall()
        col_names = [c[1] for c in cols]
        df = pd.read_sql_query(f"SELECT * FROM {t} LIMIT 20000", conn)
        if df.empty:
            continue
        date_cols = [c for c in df.columns if "date" in c.lower() or "day" in c.lower() or "ts" in c.lower()]
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if date_cols and num_cols:
            chosen = (t, date_cols[0], num_cols[0])
            break
    # fallback to first table
    if not chosen:
        t = tables[0]
        df = pd.read_sql_query(f"SELECT * FROM {t} LIMIT 20000", conn)
        date_cols = [c for c in df.columns if "date" in c.lower() or "day" in c.lower() or "ts" in c.lower()]
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        chosen = (t, date_cols[0] if date_cols else None, num_cols[0] if num_cols else None)

    tname, dcol, ncol = chosen
    st.subheader(f"Dashboard preview for `{tname}`")
    df = pd.read_sql_query(f"SELECT * FROM {tname} LIMIT 20000", conn)

    # KPIs for top numeric columns
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    kpi_cols = num_cols[:3]
    if kpi_cols:
        kpi_cells = st.columns(len(kpi_cols))
        for i, c in enumerate(kpi_cols):
            val = df[c].sum(skipna=True)
            kpi_cells[i].metric(label=f"Total {c}", value=f"{val:,.0f}")

    # Time-series chart
    if dcol and ncol:
        try:
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
            daily = df.dropna(subset=[dcol]).groupby(pd.Grouper(key=dcol, freq="D")).agg({ncol: "sum"}).reset_index()
            fig = px.line(daily, x=dcol, y=ncol, title=f"Daily {ncol} (from {tname})")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"Failed to render time-series: {e}")

    # Top categorical distribution
    cat_cols = [c for c in df.columns if df[c].nunique(dropna=True) < max(50, len(df)//10) and not pd.api.types.is_numeric_dtype(df[c])]
    if cat_cols:
        c = cat_cols[0]
        top = df.groupby(c).size().reset_index(name="count").sort_values("count", ascending=False).head(20)
        fig2 = px.bar(top, x=c, y="count", title=f"Top {c} (from {tname})")
        st.plotly_chart(fig2, use_container_width=True)

if st.button("Generate Quick Dashboard"):
    quick_dashboard()

# -------------------------
# 5) History / Downloads
# -------------------------
st.header("5) History")
with st.expander("Show query history"):
    hist = st.session_state.get("history", [])
    if hist:
        dfh = pd.DataFrame(hist)
        st.dataframe(dfh.sort_values("ts", ascending=False).reset_index(drop=True))
        st.download_button("Download history (CSV)", data=dfh.to_csv(index=False).encode("utf-8"),
                           file_name="history.csv", mime="text/csv")
    else:
        st.write("No history yet.")

# -------------------------
# Helpful utilities & footer
# -------------------------
st.markdown("---")
st.markdown(
    """
**Notes & safety**
- The app blocks destructive SQL (INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/TRUNCATE).  
- When using the OpenAI API, set `OPENAI_API_KEY` as an environment variable (or disable API checkbox).  
- Local fallback is conservative and generic — it may not satisfy complex NL→SQL needs but guarantees no external calls.
"""
)

# end of app.py
