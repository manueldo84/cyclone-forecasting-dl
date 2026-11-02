# ===== CELL A: extra test cases (robustness of the final ensemble; no retraining) =====
def ens_pred(Xs): return Pp_te + np.mean([resid(m, Xs, FI) for m in ens], axis=0)
tests = {"Normal inputs": eval_preds(ens_pred(Xte_s), Yte, mte)}

Xn = Xte.copy()                                  # 1) environment missing = what the app does when optional columns are blank
for f, key in [("sst_c", "sst"), ("shear_ms", "shear"), ("steer_u", "steer_u"), ("steer_v", "steer_v")]:
    Xn[:, :, IDX[f]] = cu.ENV_DEFAULT[key]
tests["Environment missing (typical values)"] = eval_preds(ens_pred(nx(Xn)), Yte, mte)

Xp = Xte.copy()                                  # 2) pressure not available
Xp[:, :, IDX["has_pres"]] = 0; Xp[:, :, IDX["pres_hpa"]] = cu.pressure_from_wind(Xte[:, :, IDX["wind_kt"]])
tests["Pressure missing (estimated from wind)"] = eval_preds(ens_pred(nx(Xp)), Yte, mte)

rng = np.random.default_rng(SEED)                # 3) noisy inputs (Gaussian noise in standardised units)
for sig in (0.1, 0.25, 0.5):
    tests[f"Input noise, sigma = {sig}"] = eval_preds(ens_pred(Xte_s + rng.normal(0, sig, Xte_s.shape).astype("float32")), Yte, mte)
tests["Persistence (reference)"] = results["Persistence"]
t = pd.DataFrame(tests).T[["track_km_12h", "track_km_24h", "wind_mae_24h", "d2l_mae_24h"]].round(1)
display(t); t.to_csv(OUT / "extra_test_cases.csv")

# ===== CELL B: statistical confidence, bootstrap over test STORMS (not windows) =====
w_m = np.abs(preds[ENS][:, 3 * STEPS - 1] - Yte[:, 3 * STEPS - 1]); w_p = np.abs(Pp_te[:, 3 * STEPS - 1] - Yte[:, 3 * STEPS - 1])
g_ = pd.DataFrame({"sid": mte.sid.values, "tm": d_m, "tp": d_p, "wm": w_m, "wp": w_p}).groupby("sid").sum()
n_ = len(g_); rng = np.random.default_rng(SEED); it, iw = [], []
for _ in range(2000):
    s_ = g_.values[rng.integers(0, n_, n_)].sum(0)
    it.append(1 - s_[0] / s_[1]); iw.append(1 - s_[2] / s_[3])
ci = lambda a: (np.percentile(a, 2.5) * 100, np.percentile(a, 97.5) * 100)
tot = g_.values.sum(0)
print(f"24 h track error vs persistence: {100 * (1 - tot[0] / tot[1]):.1f}% lower (95% CI {ci(it)[0]:.1f}% to {ci(it)[1]:.1f}%), {n_} test storms")
print(f"24 h wind error vs persistence:  {100 * (1 - tot[2] / tot[3]):.1f}% lower (95% CI {ci(iw)[0]:.1f}% to {ci(iw)[1]:.1f}%)")
