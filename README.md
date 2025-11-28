# DataPilot — Text-to-SQL & Quick Dashboard (Streamlit App)

This is a generic, dataset-agnostic natural language → SQL engine and dashboard generator.

### Features
- Upload CSV / Excel / TSV / JSON / Parquet / SQL dumps.
- Google Sheets URL support (auto-converts to CSV).
- OpenAI-powered NL→SQL (optional).
- Local rule-based NL→SQL fallback.
- Safe SQL execution in in-memory SQLite (SELECT-only).
- Quick dashboard mode (heuristic KPI + charts).
- Fully cloud-hosted on HuggingFace Spaces.

### How to Use
1. Upload any tabular file(s).
2. Ask questions in plain English.
3. View SQL, run SQL, download results.
4. Generate quick dashboards automatically.

### Environment Variables
To use OpenAI:
- Set `OPENAI_API_KEY` in HuggingFace Secrets.

Without API key:
- App falls back to local SQL generator.

---

**App built using Streamlit + SQLite + Plotly.**
