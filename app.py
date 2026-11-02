"""
Tropical Cyclone 24-hour Forecast - Gradio app (interactive map, uncertainty, risk cards).
Needs in the same folder: cyclone_model_*.keras, scaler.json, demo_storms.json, app_summary.json, cyclone_utils.py
Run: pip install -r requirements.txt && python app.py
"""
import json, os, pathlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import gradio as gr
from tensorflow import keras
import cyclone_utils as cu

HERE = pathlib.Path(__file__).parent
SC = cu.load_scaler(HERE / "scaler.json")
MODELS = [keras.models.load_model(HERE / f"cyclone_model_{i}.keras") for i in range(int(SC["n_members"]))]
STORMS = json.load(open(HERE / "demo_storms.json"))      # {"name", "obs": [[iso, lat, lon, wind, pres, d2l, sst, shear, su, sv], ...]}
SUM = json.load(open(HERE / "app_summary.json"))
NAMES = [s["name"] for s in STORMS]
H, STEPS = cu.H, cu.STEPS
CONE = SC["cone67"]
nan = lambda v: np.nan if v is None else float(v)
fmt_lat = lambda v: f"{abs(v):.1f}°{'N' if v >= 0 else 'S'}"
fmt_lon = lambda v: f"{abs(v):.1f}°{'E' if v >= 0 else 'W'}"


def columns(obs, e):
    a = np.array([[nan(x) for x in r[1:]] for r in obs[:e + 1]], float)
    return dict(lat=a[:, 0], lon=a[:, 1], wind=a[:, 2], pres=a[:, 3], d2l=a[:, 4], sst=a[:, 5], shear=a[:, 6],
                steer_u=a[:, 7], steer_v=a[:, 8], month=np.array([int(r[0][5:7]) for r in obs[:e + 1]]))


def run(c):
    return cu.forecast_ensemble(MODELS, SC, c["lat"], c["lon"], c["wind"], c["d2l"], c["month"], c["pres"],
                                c["sst"], c["shear"], c["steer_u"], c["steer_v"])


# ------------------------------------------------------------------ visuals
def make_map(obs, e, fc, title):
    lat, lon = [r[1] for r in obs], [r[2] for r in obs]
    mean, mem = fc["mean"], fc["members"]
    fig = go.Figure()
    fig.add_trace(go.Scattermap(lat=lat, lon=lon, mode="lines", line=dict(width=3, color="#b0b7c3"),
                                name="Observed track", hoverinfo="skip"))
    for k in range(STEPS - 1, -1, -1):                     # uncertainty circles (67% of past errors), largest first
        clat, clon = cu.circle_latlon(mean["lat"][k], mean["lon"][k], CONE[k])
        fig.add_trace(go.Scattermap(lat=clat, lon=clon, mode="lines", fill="toself", fillcolor="rgba(229,57,53,0.09)",
                                    line=dict(width=1, color="rgba(229,57,53,0.45)"), hoverinfo="skip",
                                    name="Likely area (67% of past errors)", legendgroup="cone", showlegend=(k == STEPS - 1)))
    for m in range(mem["lat"].shape[0]):
        fig.add_trace(go.Scattermap(lat=[lat[e], *mem["lat"][m]], lon=[lon[e], *mem["lon"][m]], mode="lines",
                                    line=dict(width=1.5, color="rgba(229,57,53,0.30)"), hoverinfo="skip",
                                    name="Ensemble members", legendgroup="mem", showlegend=(m == 0)))
    hist = range(max(e - H + 1, 0), e + 1)
    fig.add_trace(go.Scattermap(lat=[lat[i] for i in hist], lon=[lon[i] for i in hist], mode="lines+markers",
                                line=dict(width=4, color="#1e88e5"), marker=dict(size=8, color="#1e88e5"),
                                name="Last 48 h (model input)", hoverinfo="text",
                                text=[f"{obs[i][0]} UTC<br>{obs[i][3]:.0f} kt" for i in hist]))
    fig.add_trace(go.Scattermap(lat=[lat[e], *mean["lat"]], lon=[lon[e], *mean["lon"]], mode="lines",
                                line=dict(width=4, color="#e53935"), name="Forecast (ensemble mean)", hoverinfo="skip"))
    cats = [cu.category(w) for w in mean["wind"]]
    fig.add_trace(go.Scattermap(lat=list(mean["lat"]), lon=list(mean["lon"]), mode="markers+text",
                                marker=dict(size=15, color=[c[1] for c in cats]), showlegend=False,
                                text=[f"+{6 * (k + 1)}h" for k in range(STEPS)], textposition="top right", hoverinfo="text",
                                hovertext=[f"<b>+{6 * (k + 1)} h</b><br>{mean['wind'][k]:.0f} kt ({cats[k][0]})<br>"
                                           f"{mean['d2l'][k]:.0f} km from land<br>±{CONE[k]:.0f} km (67%)" for k in range(STEPS)]))
    if e + STEPS < len(obs):
        fig.add_trace(go.Scattermap(lat=lat[e:e + STEPS + 1], lon=lon[e:e + STEPS + 1], mode="lines+markers",
                                    line=dict(width=3, color="#2e7d32"), marker=dict(size=9, color="#2e7d32"),
                                    name="What actually happened", hoverinfo="text",
                                    text=[f"{obs[i][0]} UTC<br>{obs[i][3]:.0f} kt" for i in range(e, e + STEPS + 1)]))
    pl = np.r_[lat[max(e - H, 0):e + 1], mean["lat"]]; po = np.r_[lon[max(e - H, 0):e + 1], mean["lon"]]
    span = max(np.ptp(pl), np.ptp(po) * np.cos(np.radians(pl.mean()))) + 4
    fig.update_layout(map=dict(style="open-street-map", center=dict(lat=float(pl.mean()), lon=float(po.mean())),
                               zoom=float(np.clip(np.log2(360 / (span * 1.5)), 2.5, 8))),
                      margin=dict(l=0, r=0, t=42, b=0), height=580, title=dict(text=title, x=0.01, font=dict(size=16)),
                      legend=dict(x=0.01, y=0.97, bgcolor="rgba(255,255,255,0.9)", font=dict(size=11)))
    return fig


def card(label, value, sub, color):
    return (f"<div style='flex:1;min-width:190px;background:#fff;border:1px solid #e3e8ef;border-left:6px solid {color};"
            f"border-radius:12px;padding:14px 16px;box-shadow:0 1px 3px rgba(0,0,0,.06)'>"
            f"<div style='font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:#6b7280'>{label}</div>"
            f"<div style='font-size:26px;font-weight:700;color:#111827;margin:2px 0'>{value}</div>"
            f"<div style='font-size:13px;color:#4b5563'>{sub}</div></div>")


def cards(fc, c):
    m = fc["mean"]; cat, col = cu.category(m["wind"][-1])
    moved = float(cu.haversine_km(c["lat"][-1], c["lon"][-1], m["lat"][-1], m["lon"][-1]))
    ri, lf = fc["ri_prob"], fc["landfall_prob"]
    ri_auc, lf_auc = SUM["events"]["ri"]["auc"], SUM["events"]["landfall"]["auc"]
    if ri_auc != ri_auc or ri_auc < 0.6:        # low skill: do not show alarm colours or imply reliability
        ri_col, ri_sub = "#9e9e9e", f"Experimental, low skill (test ROC-AUC {ri_auc:.2f}; 0.50 = chance). Do not rely on it."
    else:
        ri_col = "#e53935" if ri >= .5 else "#fb8c00" if ri >= .2 else "#43a047"
        ri_sub = f"Ensemble chance of a gain of 30 kt or more in 24 h (test ROC-AUC {ri_auc:.2f})"
    if lf is None:
        c3 = card("Land interaction", "At / near land", f"Predicted {m['d2l'][-1]:.0f} km from land in 24 h", "#6c757d")
    else:
        lab, lc = ("High", "#e53935") if lf >= .6 else ("Moderate", "#fb8c00") if lf >= .3 else ("Low", "#43a047")
        c3 = card("Landfall approach (24 h)", f"{lab} ({lf:.0%})", f"Chance of being within 100 km of land (test ROC-AUC {lf_auc:.2f}); "
                                                                  f"mean {m['d2l'][-1]:.0f} km away", lc)
    return ("<div style='display:flex;gap:12px;flex-wrap:wrap;margin:6px 0 10px'>"
            + card("Intensity in 24 h", f"{m['wind'][-1]:.0f} kt", f"{cat} (now {c['wind'][-1]:.0f} kt)", col)
            + card("Position in 24 h", f"{fmt_lat(m['lat'][-1])}, {fmt_lon(m['lon'][-1])}",
                   f"{moved:.0f} km from current position, ±{CONE[-1]:.0f} km (67%)", "#1e88e5")
            + c3 + card("Rapid intensification", f"{ri:.0%}", ri_sub, ri_col)
            + "</div>")


def table(fc, c, obs=None, e=None):
    m = fc["mean"]; rows = []
    for k in range(STEPS):
        r = {"Lead (h)": 6 * (k + 1), "Latitude": round(float(m["lat"][k]), 2), "Longitude": round(float(m["lon"][k]), 2),
             "Wind (kt)": round(float(m["wind"][k]), 0), "Category": cu.category(m["wind"][k])[0],
             "Distance to land (km)": round(float(m["d2l"][k]), 0), "Uncertainty ± (km)": round(float(CONE[k]), 0)}
        if obs is not None and e + k + 1 < len(obs):
            t = obs[e + k + 1]
            r["Actual wind (kt)"] = round(float(t[3]), 0)
            r["Track error (km)"] = round(float(cu.haversine_km(m["lat"][k], m["lon"][k], t[1], t[2])), 0)
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ callbacks
def hindcast(name, e):
    obs = STORMS[NAMES.index(name)]["obs"]
    e = int(min(max(e, H), len(obs) - 1))
    c = columns(obs, e); fc = run(c)
    return cards(fc, c), make_map(obs, e, fc, f"{name}: forecast issued {obs[e][0][:16]} UTC"), table(fc, c, obs, e)


def on_storm(name):
    n = len(STORMS[NAMES.index(name)]["obs"])
    return gr.update(minimum=H, maximum=n - 1, value=min(H + 10, n - 1))


COLS = ["lat", "lon", "wind_kt", "pressure_hpa (optional)", "dist_to_land_km", "sst_C (optional)",
        "shear_m_s (optional)", "steer_u (optional)", "steer_v (optional)"]


def manual(df, month):
    d = pd.DataFrame(df)
    if list(d.columns) != COLS:
        d = d.iloc[:, :len(COLS)]; d.columns = COLS[:d.shape[1]]
    d = d.apply(pd.to_numeric, errors="coerce")
    d = d.dropna(subset=[COLS[0], COLS[1], COLS[2], COLS[4]])
    if len(d) < H + 1:
        raise gr.Error(f"Enter at least {H + 1} complete rows (6-hourly, oldest first; lat, lon, wind and distance to land are required).")
    a = d.values.astype(float)
    c = dict(lat=a[:, 0], lon=a[:, 1], wind=a[:, 2], pres=a[:, 3], d2l=a[:, 4], sst=a[:, 5], shear=a[:, 6],
             steer_u=a[:, 7], steer_v=a[:, 8], month=np.full(len(a), int(month)))
    fc = run(c)
    obs = [[f"2000-{int(month):02d}-01 00:00", r[0], r[1], r[2]] for r in a]
    return cards(fc, c), make_map(obs, len(obs) - 1, fc, "Forecast from your observations"), table(fc, c)


def evidence():
    p = SUM["perf"]
    md = ["### How well does it work?", f"Held-out test: **{SUM['counts']['test_storms']} storms / {SUM['counts']['test_windows']:,} forecasts** "
          f"(storms never seen in training). Environmental data: **{'used' if SUM['use_env'] else 'not used'}**.", "",
          "| Model | 12 h track error (km) | 24 h track error (km) | 24 h wind error (kt) |", "|---|---|---|---|"]
    md += [f"| {r['model']} | {r['track_12h']:.0f} | {r['track_24h']:.0f} | {r['wind_24h']:.1f} |" for r in p]
    md += ["", "*Lower is better. Persistence = the storm keeps its current motion and intensity; any useful model must beat it.*", "",
           "### Does the extra information help? (same GRU, features added step by step)", "",
           "| Inputs | 24 h track error (km) | 24 h wind error (kt) |", "|---|---|---|"]
    md += [f"| {r['name']} | {r['track_24h']:.0f} | {r['wind_24h']:.1f} |" for r in SUM["ablation"]]
    ev = SUM["events"]
    md += ["", "### Risk indicators (held-out test)", "",
           f"- **Rapid intensification (≥30 kt in 24 h):** ROC-AUC {ev['ri']['auc']:.2f}; alert at ≥40% chance catches "
           f"{ev['ri']['recall']:.0%} of events with {ev['ri']['precision']:.0%} precision.",
           f"- **Landfall approach (within 100 km of land in 24 h):** ROC-AUC {ev['landfall']['auc']:.2f}; "
           f"recall {ev['landfall']['recall']:.0%}, precision {ev['landfall']['precision']:.0%}.",
           f"- **Uncertainty circles:** drawn at the 67th percentile of past 24 h errors ({CONE[-1]:.0f} km); on the test set they "
           f"contained the real position {SUM['coverage'][-1]:.0%} of the time (target about 67%).", "",
           "### Limits", "Reanalysis fields are coarse (2.5°) and daily; wind averaging periods differ between agencies; "
           "the model sees no satellite imagery. **Research and education demo, not an operational forecast.**"]
    imp = SUM["importance"]
    fig = go.Figure(go.Bar(x=[v["track"] for v in imp.values()], y=list(imp.keys()), orientation="h", marker_color="#1e88e5"))
    fig.update_layout(title="What the model relies on: 24 h track error increase when a group is shuffled (km)",
                      height=300, margin=dict(l=10, r=10, t=50, b=30), xaxis_title="km worse when shuffled")
    return "\n".join(md), fig


_o = STORMS[0]["obs"]; _e = min(H + 10, len(_o) - 1)
EXAMPLE = [[nan(x) if x is not None else None for x in r[1:]] for r in _o[_e - H:_e + 1]]
HEADER = ("<div style='background:linear-gradient(120deg,#0b3d91,#1e88e5 60%,#4fc3f7);color:#fff;border-radius:14px;"
          "padding:22px 26px;margin-bottom:8px'><div style='font-size:28px;font-weight:800'>Tropical Cyclone 24-hour Forecast</div>"
          "<div style='font-size:15px;opacity:.95;margin-top:4px'>Deep-learning ensemble using storm history, geography, "
          "sea-surface temperature, vertical wind shear and steering flow</div></div>"
          "<div style='background:#fff8e1;border:1px solid #ffe082;border-radius:10px;padding:9px 14px;font-size:13px;color:#5d4037'>"
          "Research / education demo. Not an operational forecast; never use it for safety decisions. Follow your national weather service (IMD in India).</div>")

with gr.Blocks(title="Tropical Cyclone 24-hour Forecast") as demo:
    gr.HTML(HEADER)
    with gr.Tab("Hindcast: replay real storms"):
        gr.Markdown("Pick a storm (all were **held out from training**) and a forecast time, and compare the forecast with what happened.")
        with gr.Row():
            dd = gr.Dropdown(NAMES, value=NAMES[0], label="Storm", scale=2)
            sl = gr.Slider(H, len(_o) - 1, value=_e, step=1, label="Forecast issued at observation # (6-hourly)", scale=3)
            btn = gr.Button("Run forecast", variant="primary", scale=1)
        out_cards, out_map, out_tbl = gr.HTML(), gr.Plot(), gr.Dataframe(label="Forecast table")
        dd.change(on_storm, dd, sl)
        btn.click(hindcast, [dd, sl], [out_cards, out_map, out_tbl])
        sl.release(hindcast, [dd, sl], [out_cards, out_map, out_tbl])
        demo.load(hindcast, [dd, sl], [out_cards, out_map, out_tbl])
    with gr.Tab("Your own observations"):
        gr.Markdown("Enter **9 or more** consecutive 6-hourly observations, oldest first. Lat, lon, wind (kt) and distance to land (km) "
                    "are required. Optional columns can be left blank (typical values are used); supplying SST, 200-850 hPa shear "
                    "and steering winds from GFS/ERA5 improves accuracy. Prefilled with an example.")
        df = gr.Dataframe(value=EXAMPLE, headers=COLS, datatype="number", interactive=True)
        month = gr.Dropdown(list(range(1, 13)), value=int(_o[_e][0][5:7]), label="Month")
        b2 = gr.Button("Run forecast", variant="primary")
        c2, m2, t2 = gr.HTML(), gr.Plot(), gr.Dataframe(label="Forecast table")
        b2.click(manual, [df, month], [c2, m2, t2])
    with gr.Tab("Model and evidence"):
        ev_md, ev_fig = evidence()
        gr.Markdown(ev_md); gr.Plot(value=ev_fig)

if __name__ == "__main__":
    demo.launch(share=os.environ.get("SHARE") == "1")
