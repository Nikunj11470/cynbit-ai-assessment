"""Revio Insight Engine - Streamlit demo.

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import os
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import llm  # noqa: E402
from src.config import ASPECT_LABELS, ASPECTS  # noqa: E402
from src.pipeline import analyse_text, run  # noqa: E402
from src.replies import generate_reply, load_brand  # noqa: E402

st.set_page_config(page_title="Revio Insight Engine", layout="wide")

# Streamlit Cloud: read the API key from secrets if present
try:
    if "ANTHROPIC_API_KEY" in st.secrets and not os.getenv("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    pass

SENT_COLORS = {"positive": "#2E9E6B", "neutral": "#9AA0A6", "negative": "#D9534F"}


@st.cache_resource(show_spinner="Running the pipeline on 3,000+ reviews...")
def get_result():
    return run()


@st.cache_data
def get_brand():
    return load_brand()


res = get_result()
brand = get_brand()
reviews = res.reviews
loc_names = dict(zip(reviews["location_id"], reviews["location_name"]))

# ---------------- Sidebar ----------------
st.sidebar.title("Revio Insight Engine")
st.sidebar.caption(f"Demo client: **{brand['brand_name']}** (fictional), {len(loc_names)} outlets")
llm_ready = llm.available()
use_llm = st.sidebar.toggle("Use Claude for reply drafts", value=False, disabled=not llm_ready,
                            help="Needs ANTHROPIC_API_KEY. Without it the app runs fully offline.")
st.sidebar.markdown(f"LLM status: {'🟢 connected' if llm_ready else '⚪ offline mode'}")
sel_locs = st.sidebar.multiselect("Outlets", options=list(loc_names), format_func=loc_names.get,
                                  default=list(loc_names))
min_d, max_d = reviews["date"].min().date(), reviews["date"].max().date()
date_range = st.sidebar.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
st.sidebar.divider()
st.sidebar.caption(
    f"Pipeline stats: {res.stats.get('raw_reviews', len(reviews))} raw reviews · "
    f"{res.stats.get('duplicates_removed', 0)} cross-posted duplicates removed · "
    f"{res.stats.get('pii_masked', 0)} with PII masked · {res.stats.get('hinglish', 0)} Hinglish"
)

start, end = (date_range if isinstance(date_range, tuple) and len(date_range) == 2 else (min_d, max_d))
mask = reviews["location_id"].isin(sel_locs) & reviews["date"].dt.date.between(start, end)
view = reviews[mask]
long = res.aspects_long[res.aspects_long["review_id"].isin(view["review_id"])]

tabs = st.tabs(["📊 Overview", "🔍 Why ratings move", "🧩 Complaint themes", "🚨 Crisis alerts",
                "💬 Reply studio", "🧪 Try a review"])

# ---------------- Overview ----------------
with tabs[0]:
    st.header("Reputation overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reviews", f"{len(view):,}")
    c2.metric("Avg rating", f"{view['rating'].mean():.2f} ★")
    c3.metric("Negative share", f"{(view['pred_sentiment'] == 'negative').mean():.0%}")
    c4.metric("High-severity reviews", int((view["severity"] == "high").sum()))

    weekly = (view.set_index("date").groupby("location_name")["rating"]
              .resample("W").mean().reset_index())
    fig = px.line(weekly, x="date", y="rating", color="location_name", markers=True,
                  labels={"rating": "Avg rating (weekly)", "date": "", "location_name": "Outlet"})
    for _, inc in res.incidents[res.incidents["location_id"].isin(sel_locs)].iterrows():
        fig.add_vrect(x0=pd.Timestamp(inc["start"]), x1=pd.Timestamp(inc["end"]) + timedelta(days=1),
                      fillcolor="red", opacity=0.08, line_width=0)
    fig.update_layout(height=380, legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Red bands = crisis incidents detected by the early-warning system.")

    st.subheader("Aspect sentiment by outlet")
    if not long.empty:
        net = (long.assign(score=long["sentiment"].map({"positive": 1, "negative": -1, "neutral": 0}))
               .groupby(["location_name", "aspect"])["score"].mean().unstack().reindex(columns=ASPECTS))
        net.columns = [ASPECT_LABELS[a] for a in net.columns]
        heat = px.imshow(net, color_continuous_scale="RdYlGn", zmin=-1, zmax=1, aspect="auto", text_auto=".2f")
        heat.update_layout(height=360, coloraxis_colorbar_title="Net<br>sentiment")
        st.plotly_chart(heat, use_container_width=True)
        st.caption("Net sentiment = (positive - negative) / mentions. Green = strength, red = problem.")

# ---------------- Why ratings move ----------------
with tabs[1]:
    st.header("What is driving the rating change?")
    loc = st.selectbox("Outlet", options=sel_locs or list(loc_names), format_func=loc_names.get, key="drv_loc")
    window = st.slider("Compare the last N days with the N days before", 14, 45, 30)
    cutoff = reviews["date"].max() - pd.Timedelta(days=window)
    prev_cut = cutoff - pd.Timedelta(days=window)
    la = res.aspects_long[res.aspects_long["location_id"] == loc]
    score = la["sentiment"].map({"positive": 1, "negative": -1, "neutral": 0})
    recent = score[la["date"] > cutoff].groupby(la["aspect"]).mean()
    before = score[(la["date"] > prev_cut) & (la["date"] <= cutoff)].groupby(la["aspect"]).mean()
    drivers = pd.DataFrame({"before": before, "recent": recent}).dropna()
    drivers["change"] = drivers["recent"] - drivers["before"]
    drivers = drivers.sort_values("change")
    drivers.index = [ASPECT_LABELS[a] for a in drivers.index]

    r_now = reviews[(reviews["location_id"] == loc) & (reviews["date"] > cutoff)]["rating"].mean()
    r_before = reviews[(reviews["location_id"] == loc) & (reviews["date"] > prev_cut) & (reviews["date"] <= cutoff)]["rating"].mean()
    st.metric(f"Avg rating, last {window} days", f"{r_now:.2f} ★", f"{r_now - r_before:+.2f} vs previous {window} days")

    bar = go.Figure(go.Bar(x=drivers["change"], y=drivers.index, orientation="h",
                           marker_color=["#D9534F" if v < 0 else "#2E9E6B" for v in drivers["change"]]))
    bar.update_layout(height=380, xaxis_title="Change in net sentiment", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(bar, use_container_width=True)
    if not drivers.empty:
        worst = drivers.index[0]
        st.info(f"**Biggest negative driver at {loc_names[loc]}: {worst}** "
                f"(net sentiment {drivers['before'].iloc[0]:+.2f} → {drivers['recent'].iloc[0]:+.2f}).")

# ---------------- Themes ----------------
with tabs[2]:
    st.header("Recurring complaint themes")
    st.caption(f"Negative snippets clustered with TF-IDF → LSA → K-Means "
               f"(k picked by silhouette = {res.themes.attrs.get('silhouette', '-')}). All outlets, full period.")
    for _, t in res.themes.head(12).iterrows():
        with st.expander(f"**{t['representative'].capitalize()}** — {t['mentions']} mentions "
                         f"· {ASPECT_LABELS[t['main_aspect']]} · most at {t['top_location']}"):
            st.write("Key terms:", t["label"])
            for ex in t["examples"]:
                st.markdown(f"> {ex}")

# ---------------- Crisis ----------------
with tabs[3]:
    st.header("Crisis early-warning")
    inc = res.incidents[res.incidents["location_id"].isin(sel_locs)].copy()
    if inc.empty:
        st.success("No incidents for the selected outlets.")
    else:
        inc["outlet"] = inc["location_id"].map(loc_names)
        st.dataframe(inc[["outlet", "start", "end", "days", "peak_neg_share", "high_severity_reviews", "trigger"]],
                     use_container_width=True, hide_index=True)
        pick = st.selectbox("Inspect outlet signal", options=inc["location_id"].unique(), format_func=loc_names.get)
        sig = res.signal[res.signal["location_id"] == pick]
        f = go.Figure()
        f.add_trace(go.Scatter(x=sig["date"], y=sig["neg_share"], name="Negative share (3-day)", line=dict(color="#D9534F")))
        f.add_trace(go.Scatter(x=sig["date"], y=sig["baseline"], name="28-day baseline", line=dict(dash="dash", color="#555")))
        f.add_trace(go.Bar(x=sig["date"], y=sig["high"], name="High-severity reviews", yaxis="y2", marker_color="#F0AD4E", opacity=0.6))
        f.update_layout(height=380, yaxis=dict(title="Share", tickformat=".0%"),
                        yaxis2=dict(title="High-severity", overlaying="y", side="right"), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(f, use_container_width=True)
        hs = reviews[(reviews["location_id"] == pick) & (reviews["severity"] == "high")].nlargest(5, "created_at")
        st.subheader("Latest high-severity reviews")
        for _, r in hs.iterrows():
            st.error(f"{r['date'].date()} · {r['rating']}★ · {r['severity_category']} — {r['text_clean']}")

# ---------------- Reply studio ----------------
with tabs[4]:
    st.header("Reply studio (human-in-the-loop)")
    filt = st.radio("Show", ["Needs attention (≤2★ or high severity)", "All recent"], horizontal=True)
    pool = view if filt == "All recent" else view[(view["rating"] <= 2) | (view["severity"] == "high")]
    pool = pool.nlargest(40, "created_at")
    if pool.empty:
        st.info("No reviews match.")
    else:
        rid = st.selectbox("Review", pool["review_id"],
                           format_func=lambda r: f"{pool.set_index('review_id').loc[r, 'rating']}★ · "
                                                 f"{pool.set_index('review_id').loc[r, 'location_name']} · "
                                                 f"{pool.set_index('review_id').loc[r, 'text_clean'][:80]}...")
        rv = pool.set_index("review_id").loc[rid].to_dict() | {"review_id": rid}
        left, right = st.columns(2)
        with left:
            st.subheader("Review")
            st.write(f"**{rv['rating']}★** · {rv['location_name']} · {rv['platform']} · {rv['date'].date()}")
            st.write(rv["text_clean"])
            st.write("**Detected aspects**")
            for a in rv["aspects"]:
                st.markdown(f":{'green' if a['sentiment'] == 'positive' else 'red'}[{ASPECT_LABELS[a['aspect']]} "
                            f"({a['sentiment']})] — _“{a['evidence']}”_")
            st.write(f"**Severity:** {rv['severity']} ({rv['severity_category']}) · "
                     f"**Model sentiment:** {rv['pred_sentiment']} ({rv['sentiment_conf']:.0%})")
        with right:
            st.subheader("Draft reply")
            key = f"reply_{rid}_{use_llm}"
            if key not in st.session_state:
                with st.spinner("Drafting..."):
                    st.session_state[key] = generate_reply(rv, brand, use_llm=use_llm)
            out = st.session_state[key]
            edited = st.text_area("Edit before posting", out["reply"], height=180)
            st.caption(f"Source: {out['source']}")
            for name, ok in out["checks"].items():
                st.write(("✅ " if ok else "❌ ") + name)
            if out["route"] == "auto_post":
                st.success("Route: safe to auto-post")
            else:
                st.warning("Route: human approval required — " + "; ".join(out["reasons"]))
            if st.button("Approve & post (simulated)", type="primary"):
                st.toast("Reply queued for posting (demo — nothing is sent).")

# ---------------- Try a review ----------------
with tabs[5]:
    st.header("Try it live")
    sample = "Biryani ka swad zabardast tha but we waited over an hour and the waiter was rude. Call me on 9876543210."
    text = st.text_area("Paste any review", sample, height=110)
    loc_try = st.selectbox("Outlet", list(loc_names), format_func=loc_names.get, key="try_loc")
    stars = st.slider("Star rating", 1, 5, 2)
    if st.button("Analyse", type="primary") and text.strip():
        a = analyse_text(text, res.sentiment_model)
        c1, c2, c3 = st.columns(3)
        c1.metric("Sentiment", a["pred_sentiment"], f"{a['sentiment_conf']:.0%} confidence", delta_color="off")
        c2.metric("Severity", a["severity"], a["severity_category"], delta_color="off")
        c3.metric("Language", a["language"])
        st.write("**PII-scrubbed text:**", a["text_clean"])
        st.dataframe(pd.DataFrame(a["aspects"]), use_container_width=True, hide_index=True)
        rv = {"review_id": "live", "location_id": loc_try, "location_name": loc_names[loc_try], "rating": stars,
              "text": text, **a}
        out = generate_reply(rv, brand, use_llm=use_llm)
        st.subheader("Suggested reply")
        st.write(out["reply"])
        st.caption(f"Source: {out['source']} · Route: {out['route']}"
                   + (f" ({'; '.join(out['reasons'])})" if out["reasons"] else ""))
