"""Shared helpers for the cyclone forecasting project (used by the notebook and by app.py)."""
import json
import numpy as np

H = 8            # input window: 8 observations x 6 h = 48 h of history
STEPS = 4        # forecast leads: +6, +12, +18, +24 h
FEATS = ["lat", "sin_lon", "cos_lon", "wind_kt", "log_dist2land", "dlat", "dlon", "dwind", "sin_month", "cos_month",
         "speed_kmh", "sin_heading", "cos_heading", "pres_hpa", "has_pres",
         "sst_c", "shear_ms", "steer_u", "steer_v"]
IDX = {n: i for i, n in enumerate(FEATS)}
GROUPS = {"base": list(range(0, 10)), "motion_pressure": list(range(10, 15)), "environment": list(range(15, 19))}
GROUP_LABELS = {"base": "Position, wind, geography, season", "motion_pressure": "Storm motion and pressure",
                "environment": "Environment (SST, shear, steering)"}
ENV_DEFAULT = {"sst": 27.5, "shear": 8.0, "steer_u": -3.0, "steer_v": 1.0}   # used when a value is missing
# Outputs (16): [dlat x4, dlon x4, dwind x4, d(dist-to-land) x4] relative to the last observation
CATS = [(34, "Depression", "#6c757d"), (64, "Tropical storm", "#1e88e5"), (83, "Category 1", "#43a047"),
        (96, "Category 2", "#c0ca33"), (113, "Category 3", "#fb8c00"), (137, "Category 4", "#e53935"),
        (1e9, "Category 5", "#8e24aa")]


def category(w):
    for hi, name, col in CATS:
        if w < hi:
            return name, col


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(np.asarray(lon2, float) - np.asarray(lon1, float))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def unwrap_lon(lon):
    return np.degrees(np.unwrap(np.radians(np.asarray(lon, float))))


def pressure_from_wind(w):
    """Atkinson-Holliday wind-pressure relation (knots -> hPa), used only where pressure is missing."""
    return 1010.0 - (np.maximum(np.asarray(w, float), 1.0) / 6.7) ** (1 / 0.644)


def _fill(a, n, default):
    if a is None:
        return np.full(n, default, float)
    a = np.array(a, float)
    a[~np.isfinite(a)] = default
    return a


def storm_features(lat, lon, wind, d2l, month, pres=None, sst=None, shear=None, steer_u=None, steer_v=None):
    """Per-observation features for one storm (consecutive 6-hourly rows). Returns (T, 19) float32.
    Row 0 has zero motion by construction, so windows must start at row >= 1."""
    lat, lon, wind, d2l, month = (np.asarray(a, float) for a in (lat, lon, wind, d2l, month))
    n = len(lat)
    lon_u = unwrap_lon(lon)
    dlat, dlon, dw = (np.r_[0.0, np.diff(a)] for a in (lat, lon_u, wind))
    lr, m = np.radians(lon), 2 * np.pi * month / 12.0
    dy, dx = dlat * 111.19, dlon * 111.19 * np.cos(np.radians(lat))
    speed = np.hypot(dx, dy) / 6.0
    head = np.arctan2(dx, dy)
    sh, ch = np.where(speed > 0, np.sin(head), 0.0), np.where(speed > 0, np.cos(head), 1.0)
    if pres is None:
        has, p = np.zeros(n), pressure_from_wind(wind)
    else:
        pres = np.asarray(pres, float); ok = np.isfinite(pres)
        has, p = ok.astype(float), np.where(ok, pres, pressure_from_wind(wind))
    cols = [lat, np.sin(lr), np.cos(lr), wind, np.log1p(np.clip(d2l, 0, None)), dlat, dlon, dw, np.sin(m), np.cos(m),
            speed, sh, ch, p, has,
            _fill(sst, n, ENV_DEFAULT["sst"]), _fill(shear, n, ENV_DEFAULT["shear"]),
            _fill(steer_u, n, ENV_DEFAULT["steer_u"]), _fill(steer_v, n, ENV_DEFAULT["steer_v"])]
    return np.stack(cols, axis=1).astype("float32")


def persistence_delta(x_last):
    """Constant-motion forecast of the 16 outputs from RAW features of the last observation. (N,F) or (F,) -> (N,16)."""
    x = np.atleast_2d(np.asarray(x_last, float))
    k = np.arange(1, STEPS + 1)
    P = np.zeros((len(x), 4 * STEPS))
    P[:, :STEPS] = x[:, [IDX["dlat"]]] * k
    P[:, STEPS:2 * STEPS] = x[:, [IDX["dlon"]]] * k
    return P


def decode(y, lat0, lon0, wind0, d2l0):
    y = np.asarray(y, float)
    return {"lat": lat0 + y[:STEPS],
            "lon": ((lon0 + y[STEPS:2 * STEPS] + 180.0) % 360.0) - 180.0,
            "wind": np.clip(wind0 + y[2 * STEPS:3 * STEPS], 0, None),
            "d2l": np.clip(d2l0 + y[3 * STEPS:], 0, None)}


def circle_latlon(lat0, lon0, r_km, n=60):
    th = np.linspace(0, 2 * np.pi, n)
    return (lat0 + (r_km / 111.19) * np.sin(th),
            lon0 + (r_km / (111.19 * max(np.cos(np.radians(lat0)), 0.1))) * np.cos(th))


def load_scaler(path):
    d = json.load(open(path))
    for k in ("x_mean", "x_std", "r_mean", "r_std"):
        d[k] = np.array(d[k], dtype="float32")
    return d


def forecast_ensemble(models, sc, lat, lon, wind, d2l, month, pres=None, sst=None, shear=None, steer_u=None, steer_v=None):
    """Forecast the next 24 h from >= H+1 consecutive 6-hourly observations (oldest first)."""
    lat, lon, wind, d2l = (np.asarray(a, float) for a in (lat, lon, wind, d2l))
    if len(lat) < H + 1:
        raise ValueError(f"Need at least {H + 1} observations, got {len(lat)}")
    F = storm_features(lat, lon, wind, d2l, month, pres, sst, shear, steer_u, steer_v)
    pers = persistence_delta(F[-1])[0]
    xs = ((F[-H:][:, sc["feat_idx"]] - sc["x_mean"]) / sc["x_std"])[None].astype("float32")
    R = np.stack([np.asarray(m(xs, training=False))[0] for m in models]) * sc["r_std"] + sc["r_mean"]
    Y = pers + R
    members = [decode(y, lat[-1], lon[-1], wind[-1], d2l[-1]) for y in Y]
    stack = {k: np.stack([m[k] for m in members]) for k in members[0]}
    return {"mean": decode(Y.mean(0), lat[-1], lon[-1], wind[-1], d2l[-1]), "members": stack,
            "ri_prob": float(np.mean(stack["wind"][:, -1] - wind[-1] >= 30)),
            "landfall_prob": float(np.mean(stack["d2l"][:, -1] <= 100)) if d2l[-1] > 100 else None}
