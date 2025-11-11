# ===============================================================
#  Quant Studio Pro — with "Amount Added / Current Value" in all tabs
#  + Combined Showdown chart (BTC Price, SMA, SMA+ML, Buy&Hold, SIP)
#  + Local GPT4All AI Agent (Jupyter-friendly)
# ===============================================================
# Save as: app.py  (or run as a single cell in Jupyter)
# Requires:
#   pip install gpt4all streamlit streamlit-jupyter plotly scikit-learn xlsxwriter pandas numpy joblib
# Make sure your model exists at: ./models/qwen2-0_5b-instruct-q4_k_m.gguf
# ===============================================================

import os, io, joblib, math, warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

# --- Streamlit in Jupyter (safe import) ---
IN_JUPYTER = False
try:
    get_ipython  # type: ignore
    IN_JUPYTER = True
except Exception:
    IN_JUPYTER = False

if IN_JUPYTER:
    from streamlit_jupyter import StreamlitPatcher
    StreamlitPatcher().jupyter()

import streamlit as st
from gpt4all import GPT4All

warnings.filterwarnings("ignore")

# ===============================================================
#  Paths & constants
# ===============================================================
DATA_DIR     = "data"
MODEL_DIR    = "models"     # sklearn scaler/logistic
LLM_DIR      = "models"     # LLM folder
LLM_FILENAME = "qwen2-0_5b-instruct-q4_k_m.gguf"  # your working file

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LLM_DIR,   exist_ok=True)

ANNUALIZATION = 252

# ===============================================================
#  Page config + theme polish
# ===============================================================
st.set_page_config(page_title="Quant Studio Pro", page_icon="📈", layout="wide")
st.markdown("""
<style>
:root { --card-bg: #0f172a10; }
section.main > div { padding-top: 0.4rem; }
.block-container { padding-top: 0.2rem; padding-bottom: 1rem; }
h1,h2,h3 { font-weight: 800; letter-spacing: -0.3px; }
div[data-testid="stMetric"] { background: var(--card-bg); border-radius: 14px; padding: 8px 12px; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { background: #11182712; padding: 8px 14px; border-radius: 12px; }
.badge { background:#22c55e22; border:1px solid #22c55e44; color:#15803d; padding:2px 8px; border-radius:999px; font-size:0.8rem; }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Quant Studio Pro ")

# ===============================================================
#  Helpers
# ===============================================================
def prefer_xlsx(csv_path: str, xlsx_path: str):
    return xlsx_path if os.path.exists(xlsx_path) else csv_path

def load_df(path: str, parse_dates=("Date",)):
    if not path or not os.path.exists(path): return None
    if path.lower().endswith((".xlsx",".xls")):
        return pd.read_excel(path, parse_dates=[c for c in parse_dates if c in ["Date"]])
    return pd.read_csv(path, parse_dates=list(parse_dates))

def safe_last(series, default=0.0):
    s = pd.Series(series).dropna()
    return float(s.iloc[-1]) if not s.empty else float(default)

def rsi(series, length=14):
    chg = series.diff()
    up  = chg.clip(lower=0).rolling(length).mean()
    dn  = -chg.clip(upper=0).rolling(length).mean()
    rs  = up/dn
    return 100 - (100/(1+rs))

def compute_features(df):
    df = df.copy().sort_values("Date").reset_index(drop=True)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"]).reset_index(drop=True)
    df["SMA10"] = df["Close"].rolling(10).mean()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA_gap"] = (df["SMA10"] - df["SMA20"]) / df["SMA20"]
    df["RSI14"] = rsi(df["Close"], 14)
    df["Ret1"]  = df["Close"].pct_change()
    df["Ret5"]  = df["Close"].pct_change(5)
    df["Ret10"] = df["Close"].pct_change(10)
    df["Vol10"] = df["Ret1"].rolling(10).std()
    df["Vol20"] = df["Ret1"].rolling(20).std()
    return df.dropna().reset_index(drop=True)

def sma_position(df, win_fast=10, win_slow=20):
    df = df.copy()
    df["SMA10"] = df["Close"].rolling(win_fast).mean()
    df["SMA20"] = df["Close"].rolling(win_slow).mean()
    above = df["SMA10"] > df["SMA20"]
    cross_up   = (above & (~above.shift(1).fillna(False))).astype(int)
    cross_down = ((~above) & (above.shift(1).fillna(False))).astype(int)
    pos = np.zeros(len(df), dtype=int)
    in_pos = 0
    for i in range(len(df)):
        if cross_up.iloc[i]==1: in_pos = 1
        if cross_down.iloc[i]==1: in_pos = 0
        pos[i] = in_pos
    df["SMA_Pos"] = pos
    return df

def vol_gate(series, window=20, threshold=0.08):
    vol = series.pct_change().rolling(window).std()
    return (vol < threshold).fillna(False).values

def sharpe(r, ann=ANNUALIZATION):
    r = pd.Series(r).dropna()
    s = r.std()
    return 0.0 if s == 0 or np.isnan(s) else (r.mean()/s)*np.sqrt(ann)

def sortino(r, ann=ANNUALIZATION):
    r = pd.Series(r).dropna()
    dn = r[r<0].std()
    return 0.0 if dn == 0 or np.isnan(dn) else (r.mean()/dn)*np.sqrt(ann)

def max_dd(eq):
    e = pd.Series(eq).astype(float)
    dd = e/e.cummax() - 1.0
    return float(dd.min())

def cagr(eq, ann=ANNUALIZATION):
    e = pd.Series(eq).astype(float)
    return (e.iloc[-1]/e.iloc[0])**(ann/len(e))-1

def backtest_long_only(df, position, fee_bps=10, daily_stop=0.03):
    fee = fee_bps/10000.0
    turn_on = (pd.Series(position).diff().fillna(0) > 0).astype(int).values
    ret_next = df["Close"].pct_change().shift(-1).fillna(0.0).values
    ret_next = np.maximum(ret_next, -daily_stop)
    net = ret_next - fee*turn_on
    strat = position * net
    equity = (1 + pd.Series(strat)).cumprod()
    stats = {
        "trades": int(turn_on.sum()),
        "exposure": float(np.mean(position)),
        "hit_rate": float(np.mean(ret_next[position==1] > 0)) if np.any(position==1) else 0.0,
        "turnover": float(np.sum(np.abs(np.diff(position))) / len(position)),
        "sharpe": float(sharpe(strat)),
        "sortino": float(sortino(strat)),
        "cagr": float(cagr(equity)),
        "maxDD": float(max_dd(equity)),
        "final": float(10000*equity.iloc[-1]),
    }
    return equity, pd.Series(strat), stats

def sip_backtest(price_df, contribution=100, freq="M"):
    d = price_df[["Date","Close"]].copy()
    inv_dates = d.set_index("Date").resample(freq).last().index
    mask = d["Date"].isin(inv_dates)
    d["Units"] = np.where(mask, contribution/d["Close"], 0.0).cumsum()
    d["ContribCum"] = np.where(mask, contribution, 0.0).cumsum()
    d["Equity"] = d["Units"]*d["Close"]
    return d[["Date","Equity","ContribCum"]]

def df_to_xlsx_bytes(df):
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="data")
    bio.seek(0)
    return bio

def ensure_models(X, y):
    """Return (scaler, lr, iso, msg). Load saved if available; else quick-train."""
    scaler_path = os.path.join(MODEL_DIR, "sma_ml_scaler.pkl")
    model_path  = os.path.join(MODEL_DIR, "sma_ml_model.pkl")

    if os.path.exists(scaler_path) and os.path.exists(model_path):
        scaler = joblib.load(scaler_path)
        lr     = joblib.load(model_path)
        split = int(len(X)*0.8)
        split = max(1, min(split, len(X)-1))
        p = lr.predict_proba(scaler.transform(X[split:]))[:,1]
        iso = IsotonicRegression(out_of_bounds="clip").fit(p, y[split:])
        return scaler, lr, iso, "Loaded saved model"

    st.warning("⚠️ No saved model found. Training a quick one inside the app…")
    split = int(len(X)*0.7)
    split = max(1, min(split, len(X)-1))
    scaler = StandardScaler().fit(X[:split])
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(scaler.transform(X[:split]), y[:split])
    p = lr.predict_proba(scaler.transform(X[split:]))[:,1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(p, y[split:])
    joblib.dump(scaler, scaler_path)
    joblib.dump(lr, model_path)
    return scaler, lr, iso, "Trained quick in-app model"

# ===============================================================
#  Local LLM (GPT4All) — your downloaded file
# ===============================================================
_LLM_SINGLETON = None
def get_local_llm():
    global _LLM_SINGLETON
    if _LLM_SINGLETON is not None:
        return _LLM_SINGLETON
    model_dir  = os.path.abspath(LLM_DIR)
    model_name = LLM_FILENAME
    _LLM_SINGLETON = GPT4All(model_name=model_name, model_path=model_dir, allow_download=False)
    return _LLM_SINGLETON

def ai_reply(context, question, temperature=0.2, max_tokens=512):
    llm = get_local_llm()
    prompt = (
        "You are a quant research copilot inside a Streamlit app. "
        "Be concise, practical, and base your answer on the context. "
        "When suggesting code, give minimal, self-contained snippets.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    )
    with llm.chat_session():
        return llm.generate(prompt, temp=temperature, max_tokens=max_tokens)

# ===============================================================
#  Sidebar — Data & Global Settings
# ===============================================================
st.sidebar.header("📥 Data & Settings")

upl = st.sidebar.file_uploader("Upload Excel (Date, Close required)", type=["xlsx","xls"])
if upl is not None:
    df_raw = pd.read_excel(upl, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    src = upl.name
else:
    fallback = prefer_xlsx("", os.path.join(DATA_DIR, "feat_daily_clean.xlsx"))
    if os.path.exists(fallback):
        df_raw = pd.read_excel(fallback, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
        src = "data/feat_daily_clean.xlsx"
    else:
        st.sidebar.error("Upload an Excel with at least 'Date' and 'Close' columns.")
        st.stop()

if "Close" not in df_raw.columns:
    st.error("Uploaded data must include a 'Close' column.")
    st.stop()

initial_capital = st.sidebar.number_input("Initial Capital ($)", min_value=100, value=10000, step=100)
fee_bps = st.sidebar.slider("Trade fee (bps)", 0, 50, 10, 1)
stop_pct = st.sidebar.slider("Daily stop (max loss %)", 0, 10, 3, 1)/100.0
trend_on = st.sidebar.toggle("Trend filter (50>200 SMA)", value=True)
vgate_on = st.sidebar.toggle("Volatility gate (20D stdev < 8%)", value=True)

# ===============================================================
#  Prep Data
# ===============================================================
df = compute_features(df_raw)
st.caption(f"Data source: **{src}**  •  rows after warm-up: **{len(df)}**")

sma_df = sma_position(df)
trend = (df["Close"].rolling(50).mean() > df["Close"].rolling(200).mean()).fillna(False).values
if not trend_on: trend = np.ones(len(df), dtype=bool)
gate = vol_gate(df["Close"], 20, 0.08) if vgate_on else np.ones(len(df), dtype=bool)
raw_pos = (sma_df["SMA_Pos"].values * trend * gate).astype(int)

# ===============================================================
#  Tabs
# ===============================================================
tabs = st.tabs([
    "🧭 Trend Pilot",
    "🤖 Smart Filter AI",
    "📦 Auto-Invest (DCA)",
    "⚔️ Strategy Showdown",
    "🧪 Lab & Diagnostics",
    "🧠 AI Agent"
])

# ===============================================================
#  Tab 1 — Trend Pilot
# ===============================================================
with tabs[0]:
    st.subheader("🧭 Trend Pilot — SMA crossover with risk overlays")

    eq_sma, ret_sma, stats_sma = backtest_long_only(df, raw_pos, fee_bps, stop_pct)
    eq_sma_dollars = initial_capital * pd.Series(eq_sma).astype(float)

    # Metrics (including Amount Added / Current Value)
    c0,c1,c2,c3,c4,c5,c6,c7 = st.columns(8)
    c0.metric("Amount Added", f"${initial_capital:,.0f}")
    c1.metric("Current Value", f"${safe_last(eq_sma_dollars):,.0f}")
    pnl = safe_last(eq_sma_dollars) - initial_capital
    c2.metric("P&L", f"${pnl:,.0f}")
    c3.metric("Return", f"{(safe_last(eq_sma)-1):.2%}")
    c4.metric("Trades", stats_sma["trades"])
    c5.metric("Exposure", f"{stats_sma['exposure']:.1%}")
    c6.metric("Sharpe", f"{stats_sma['sharpe']:.2f}")
    c7.metric("MaxDD", f"{stats_sma['maxDD']:.2%}")

    chg = pd.Series(raw_pos).diff().fillna(0).astype(int).values
    buys  = df.loc[chg== 1, ["Date","Close"]]
    sells = df.loc[chg==-1, ["Date","Close"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Close", mode="lines"))
    fig.add_trace(go.Scatter(x=buys["Date"],  y=buys["Close"],  mode="markers", name="BUY",
                             marker=dict(color="green", size=9, symbol="triangle-up")))
    fig.add_trace(go.Scatter(x=sells["Date"], y=sells["Close"], mode="markers", name="SELL",
                             marker=dict(color="red", size=9, symbol="triangle-down")))
    fig.update_layout(title="Price with SMA Signals", height=420, legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    fig_eq = go.Figure([go.Scatter(x=df["Date"], y=eq_sma_dollars, mode="lines", name="SMA Equity ($)")])
    fig_eq.update_layout(title="Equity — Trend Pilot (Dollars)", height=420, legend=dict(orientation="h"))
    st.plotly_chart(fig_eq, use_container_width=True)

# ===============================================================
#  Tab 2 — Smart Filter AI (Logistic + Isotonic)
# ===============================================================
with tabs[1]:
    st.subheader("🤖 Smart Filter AI — confirm SMA with a probability model")

    FEATURES = ["SMA_gap","RSI14","Ret1","Ret5","Ret10","Vol10","Vol20"]
    X = df[FEATURES].astype("float32").values
    y = (df["Close"].shift(-1) > df["Close"]).astype(int).values

    scaler, lr, iso, msg = ensure_models(X, y)
    st.caption(f"Model: {msg}")

    ml_thresh = st.slider("ML confidence threshold", 0.40, 0.80, 0.55, 0.01)
    if math.isnan(ml_thresh): ml_thresh = 0.55

    probs = iso.predict(lr.predict_proba(scaler.transform(X))[:,1])
    df["ML_Prob"] = probs
    ml_pos = ((raw_pos==1) & (df["ML_Prob"]>ml_thresh)).astype(int).values

    eq_ml, ret_ml, stats_ml = backtest_long_only(df, ml_pos, fee_bps, stop_pct)
    eq_ml_dollars = initial_capital * pd.Series(eq_ml).astype(float)

    # Today's Action panel + money metrics
    last_idx = df.index[-1]
    last_prob = float(df.loc[last_idx, "ML_Prob"])
    last_sma = int(raw_pos[last_idx])
    action, reason = "HOLD", "SMA not long"
    if last_sma == 1:
        if last_prob >= ml_thresh: action, reason = "BUY", f"Prob {last_prob:.2f} ≥ {ml_thresh:.2f}"
        else: reason = f"Prob {last_prob:.2f} < {ml_thresh:.2f}"
    else:
        prev_pos = int(raw_pos[last_idx-1]) if last_idx>0 else 0
        if prev_pos==1 and last_sma==0: action, reason = "SELL","SMA crossed down"

    c0,cA,cB,cC,cD,cE,cF,cG = st.columns(8)
    c0.metric("Amount Added", f"${initial_capital:,.0f}")
    cA.metric("Current Value", f"${safe_last(eq_ml_dollars):,.0f}")
    pnl_ml = safe_last(eq_ml_dollars) - initial_capital
    cB.metric("P&L", f"${pnl_ml:,.0f}")
    cC.metric("Action", action)
    cD.metric("ML_Prob", f"{last_prob:.2f}")
    cE.metric("Sharpe", f"{stats_ml['sharpe']:.2f}")
    cF.metric("Exposure", f"{stats_ml['exposure']:.1%}")
    cG.metric("MaxDD", f"{stats_ml['maxDD']:.2%}")
    st.caption(f"Why: {reason}")

    fig_ml = go.Figure([go.Scatter(x=df["Date"], y=eq_ml_dollars, mode="lines", name="SMA+ML Equity ($)")])
    fig_ml.update_layout(title="Equity — Smart Filter AI (Dollars)", height=420, legend=dict(orientation="h"))
    st.plotly_chart(fig_ml, use_container_width=True)

# ===============================================================
#  Tab 3 — Auto-Invest (DCA / SIP)
# ===============================================================
with tabs[2]:
    st.subheader("📦 Auto-Invest (Dollar-Cost Averaging / SIP)")

    freq = st.selectbox("Contribution frequency", ["D - Daily","W - Weekly","M - Monthly"], index=2)
    freq_map = {"D - Daily":"D","W - Weekly":"W","M - Monthly":"M"}
    contrib = st.number_input("Contribution amount ($)", min_value=10, value=100, step=10)

    sip_df = sip_backtest(df[["Date","Close"]], contribution=contrib, freq=freq_map[freq])
    eq_sip = pd.to_numeric(sip_df["Equity"], errors="coerce")

    fig_sip = go.Figure([go.Scatter(x=sip_df["Date"], y=eq_sip, mode="lines", name="SIP")])
    fig_sip.update_layout(title="Equity — Auto-Invest", height=420, legend=dict(orientation="h"))
    st.plotly_chart(fig_sip, use_container_width=True)

    tot_contrib = safe_last(sip_df["ContribCum"])
    final_val = safe_last(eq_sip)
    c1,c2,c3 = st.columns(3)
    c1.metric("Amount Added (Contrib)", f"${tot_contrib:,.0f}")
    c2.metric("Current Value", f"${final_val:,.0f}")
    c3.metric("P&L", f"${(final_val - tot_contrib):,.0f}")

# ===============================================================
#  Tab 4 — Strategy Showdown (all curves + BTC price)
# ===============================================================
with tabs[3]:
    st.subheader("⚔️ Strategy Showdown — Strategies (All-in-One)")

    # Recompute equity curves in $ for initial_capital
    eq_sma, _, _ = backtest_long_only(df, raw_pos, fee_bps, stop_pct)
    eq_sma_dollars = initial_capital * pd.Series(eq_sma).astype(float)

    # ML (prob at 0.55 by default for showdown)
    if "ML_Prob" not in df.columns:
        FEATURES = ["SMA_gap","RSI14","Ret1","Ret5","Ret10","Vol10","Vol20"]
        X = df[FEATURES].astype("float32").values
        y = (df["Close"].shift(-1) > df["Close"]).astype(int).values
        scaler, lr, iso, _ = ensure_models(X, y)
        df["ML_Prob"] = iso.predict(lr.predict_proba(scaler.transform(X))[:,1])

    ml_pos_show = ((raw_pos==1) & (df["ML_Prob"]>0.55)).astype(int).values
    eq_ml_show, _, _ = backtest_long_only(df, ml_pos_show, fee_bps, stop_pct)
    eq_ml_dollars = initial_capital * pd.Series(eq_ml_show).astype(float)

    # Buy & Hold in $
    eq_bh_dollars = initial_capital * (df["Close"] / df["Close"].iloc[0])

    # SIP needs alignment to daily dates — forward fill onto df["Date"]
    # Build SIP series with monthly default settings similar to Tab 3
    sip_df_show = sip_backtest(df[["Date","Close"]], contribution=500, freq="M")  # default for showdown
    sip_daily = pd.DataFrame({"Date": df["Date"]}).merge(sip_df_show[["Date","Equity"]], on="Date", how="left")
    sip_daily["Equity"] = sip_daily["Equity"].ffill()
    sip_equity = pd.to_numeric(sip_daily["Equity"], errors="coerce")

    # Combined plot with secondary axis for BTC price
    btc_norm = initial_capital * (df["Close"] / df["Close"].iloc[0])

    fig_all = go.Figure()
    fig_all.add_trace(go.Scatter(x=df["Date"], y=eq_sma_dollars, name="SMA ($)", mode="lines"))
    fig_all.add_trace(go.Scatter(x=df["Date"], y=eq_ml_dollars,  name="SMA+ML ($)", mode="lines"))
    fig_all.add_trace(go.Scatter(x=df["Date"], y=eq_bh_dollars,  name="Buy&Hold ($)", mode="lines"))
    fig_all.add_trace(go.Scatter(x=df["Date"], y=sip_equity,      name="SIP ($)", mode="lines"))

    fig_all.update_layout(
        title="Strategy Equity ($)",
        height=520,
        legend=dict(orientation="h"),
        yaxis_title="$PRICE)",
    )
    st.plotly_chart(fig_all, use_container_width=True)

    # Money metrics summary table
    summary = pd.DataFrame({
            "Strategy": ["SMA","SMA+ML","Buy&Hold","SIP"],
            "Amount Added": [initial_capital, initial_capital, initial_capital, safe_last(sip_df_show["ContribCum"])],
            "Current Value": [safe_last(eq_sma_dollars), safe_last(eq_ml_dollars), safe_last(eq_bh_dollars), safe_last(sip_df_show["Equity"])],
        })
    summary["P&L"] = summary["Current Value"] - summary["Amount Added"]
    st.dataframe(summary.style.format({"Amount Added":"${:,.0f}","Current Value":"${:,.0f}","P&L":"${:,.0f}"}),
                 use_container_width=True)

    # Download combined data
    export_df = pd.DataFrame({
            "Date": df["Date"],
            "BTC_Close": df["Close"],
            "SMA_$": eq_sma_dollars.values,
            "SMA+ML_$": eq_ml_dollars.values,
            "Buy&Hold_$": eq_bh_dollars.values,
            "SIP_$": sip_equity.values
        })
    cdl1, cdl2 = st.columns(2)
    cdl1.download_button("⬇️ Download Combined (xlsx)", data=df_to_xlsx_bytes(export_df),
                         file_name="showdown_all_series.xlsx",
                         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    cdl2.download_button("⬇️ Download Summary (csv)", data=summary.to_csv(index=False).encode(),
                         file_name="showdown_summary.csv", mime="text/csv")

# ===============================================================
#  Tab 5 — Lab & Diagnostics
# ===============================================================
with tabs[4]:
    st.subheader("🧪 Lab & Diagnostics — sensitivity • importance • rolling Sharpe • trade log")

    FEATURES = ["SMA_gap","RSI14","Ret1","Ret5","Ret10","Vol10","Vol20"]
    X = df[FEATURES].astype("float32").values
    y = (df["Close"].shift(-1) > df["Close"]).astype(int).values
    scaler, lr, iso, _ = ensure_models(X, y)
    df["ML_Prob"] = iso.predict(lr.predict_proba(scaler.transform(X))[:,1])

    thr_list = np.round(np.arange(0.45, 0.71, 0.05), 2)
    fee_list = [5, 10, 15]
    rows = []
    base_ret = df["Close"].pct_change().shift(-1).fillna(0.0).values
    base_ret = np.maximum(base_ret, -stop_pct)
    for thr in thr_list:
        for fbps in fee_list:
            pos = ((raw_pos==1) & (df["ML_Prob"]>thr)).astype(int).values
            turn = (pd.Series(pos).diff().fillna(0) > 0).astype(int).values
            net  = base_ret - (fbps/10000.0)*turn
            strat = pos*net
            eq    = (1+pd.Series(strat)).cumprod()
            rows.append({"ML_thresh":thr,"fee_bps":fbps,
                         "Sharpe":sharpe(strat),"CAGR":cagr(eq),
                         "MaxDD":max_dd(eq),"Exposure":pos.mean(),"Trades":int(turn.sum())})
    sens = pd.DataFrame(rows)
    st.write("### Sensitivity (ML threshold × fees)")
    st.dataframe(sens, use_container_width=True)

    st.write("### Feature Importance (Logistic coefficients)")
    coef = pd.Series(lr.coef_.ravel(), index=FEATURES).sort_values(key=np.abs, ascending=False)
    st.dataframe(coef.rename("coef").round(3))

    def rolling_sharpe(ret, win=126):
        out = []
        r = pd.Series(ret).reset_index(drop=True)
        for i in range(len(r)):
            a = r[max(0,i-win+1):i+1]
            out.append(sharpe(a) if len(a)>5 else np.nan)
        return pd.Series(out, index=df["Date"])

    rs_sma = rolling_sharpe(backtest_long_only(df, raw_pos, fee_bps, stop_pct)[1])
    rs_ml  = rolling_sharpe(backtest_long_only(df, ((raw_pos==1) & (df["ML_Prob"]>0.55)).astype(int).values,
                                              fee_bps, stop_pct)[1])

    fig_rs = go.Figure()
    fig_rs.add_trace(go.Scatter(x=df["Date"], y=pd.to_numeric(rs_sma, errors="coerce"), name="SMA"))
    fig_rs.add_trace(go.Scatter(x=df["Date"], y=pd.to_numeric(rs_ml,  errors="coerce"), name="SMA+ML"))
    fig_rs.update_layout(title="Rolling 6-Month Sharpe", height=360, legend=dict(orientation="h"))
    st.plotly_chart(fig_rs, use_container_width=True)

    # Optional external trade log listing
    trades_path = prefer_xlsx(os.path.join(DATA_DIR,"trades_sma_ml.xlsx"),
                              os.path.join(DATA_DIR,"trades_sma_ml_filled.xlsx"))
    tr = load_df(trades_path)
    st.write("### Trade Log (SMA+ML)")
    if tr is not None and "Date" in tr.columns:
        st.dataframe(tr, use_container_width=True, height=320)
        st.download_button("⬇️ Download Trades",
                           data=df_to_xlsx_bytes(tr),
                           file_name="trades_sma_ml.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.info("No trade log file found. Generate trades in your research notebook to populate this table.")

# ===============================================================
#  Tab 6 — Local AI Agent (GPT4All)
# ===============================================================
with tabs[5]:
    st.subheader("🧠 AI Agent")
    left, right = st.columns([2,1])

    with right:
        temp = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, 0.05)
        max_toks = st.slider("Max tokens", 128, 1024, 512, 64)
        st.markdown(f"<span class='badge'>Model</span> {LLM_FILENAME}", unsafe_allow_html=True)
        st.caption("Tip: keep temperature 0.2–0.4 for focused answers.")

        ex = st.expander("Quick prompts")
        with ex:
            st.write("- Explain the latest BUY/SELL and risk overlays.")
            st.write("- Reduce drawdown but keep CAGR > 15%.")
            st.write("- Propose new features for the ML filter and why.")
            st.write("- Provide a minimal code patch for a trailing stop.")
            st.write("- Compare SMA vs SMA+ML and when each works best.")

    with left:
        # live context
        eq_sma, ret_sma, stats_sma = backtest_long_only(df, raw_pos, fee_bps, stop_pct)
        ctx_lines = [
            f"Initial capital: ${initial_capital:,.0f}",
            f"Fee bps: {fee_bps}",
            f"Daily stop: {stop_pct:.2%}",
            f"SMA — Trades {stats_sma['trades']}, Exp {stats_sma['exposure']:.1%}, "
            f"Sharpe {stats_sma['sharpe']:.2f}, CAGR {stats_sma['cagr']:.2%}, MaxDD {stats_sma['maxDD']:.2%}",
        ]
        if "ML_Prob" in df.columns:
            ml_pos_prev = ((raw_pos==1) & (df["ML_Prob"]>0.55)).astype(int).values
            _, _, stats_ml = backtest_long_only(df, ml_pos_prev, fee_bps, stop_pct)
            ctx_lines.append(
                f"SMA+ML — Trades {stats_ml['trades']}, Exp {stats_ml['exposure']:.1%}, "
                f"Sharpe {stats_ml['sharpe']:.2f}, CAGR {stats_ml['cagr']:.2%}, MaxDD {stats_ml['maxDD']:.2%}"
            )
        ctx = "\n".join(ctx_lines)

        user_q = st.text_area("Ask the agent:", height=140,
                              placeholder="e.g., How can I reduce drawdown while keeping CAGR above 15%?")
        if st.button("Ask"):
            try:
                with st.spinner("Thinking locally…"):
                    ans = ai_reply(ctx, user_q or "Explain the performance trade-offs and next steps.", temp, max_toks)
                st.markdown(ans)
            except Exception as e:
                st.error(f"Local AI unavailable: {e}")
                st.info("Ensure your model file exists: ./models/qwen2-0_5b-instruct-q4_k_m.gguf")
