# Tropical Cyclone Track and Intensity Forecasting with Deep Learning

**Course:** Artificial Neural Networks and Deep Learning | **Assessment:** ESE Project Implementation and Deployment (CIA-III)
**Author:** Manu Mathew Eldo, 24215216, 5BDA

**Live demo:** https://manueldo84.github.io/cyclone-forecasting-dl/

Forecasts a tropical cyclone's **position, wind speed and distance to land at +6, +12, +18 and +24 hours** from its last 48 hours of
observations, with an uncertainty estimate and the probability of **landfall approach** and **rapid intensification**. Inputs combine storm
history (IBTrACS), geography (distance to land), season, central pressure, and three environmental fields from NOAA reanalysis:
**sea-surface temperature, vertical wind shear and steering flow**. A persistence baseline, Ridge regression, MLP, LSTM and GRU are compared
on **storms never seen in training**; the final model is a 5-member MLP ensemble that learns a correction to persistence.
The model is deployed as an interactive web app (hosted free on GitHub Pages) with a Gradio version for local use.

> Research and education project. **Not an operational forecast; do not use it for safety decisions.** Follow your national weather service.

## Key results (held-out test storms: 335 storms, 6,622 forecasts)

| 24-hour forecast | Persistence | Ensemble | Change |
|---|---|---|---|
| Track error | 189.4 km | **157.1 km** | -17.1% |
| Wind error | 12.3 kt | **8.5 kt** | -30.9% |
| Distance-to-land error | 200.6 km | 128.6 km | -36% |

- Better than persistence at every lead time and in all six basins.
- SST, shear and steering flow lowered the 24 h track error of a GRU from 174.5 km to 164.3 km (single run per setting).
- Uncertainty circles (67%) contained the true position 65% of the time at 24 h.
- Landfall approach: ROC-AUC 0.75 (moderate). **Rapid intensification: ROC-AUC 0.54, close to chance** (reported as a negative result).
- Neural-network results vary by about 1-2 km between training runs (GPU non-determinism); the numbers above are from the run whose models are deployed here.

## Try it

1. **Live web app:** open the live-demo link above. Run a forecast on the **Hindcast** tab (replays held-out storms such as Ockhi 2017 and
   Amphan 2020 and compares the forecast with what actually happened), try **Your own observations** (editable table; paste rows from a
   spreadsheet), and read the **Model and evidence** tab.
2. **Locally, no install:** download `index.html` and open it in a browser (needs internet only for the Plotly library and map data).
3. **Gradio version (Python 3.10-3.13):**

```bash
pip install -r requirements.txt
python check_setup.py      # optional: confirms all files are present and a forecast runs
python app.py              # then open http://127.0.0.1:7860
```

## Repository structure

```
index.html                Static web app: the five trained networks run in the browser (served by GitHub Pages)
app.py                    Gradio web app (interactive map, forecast cards, evidence tab)
cyclone_utils.py          Feature engineering, persistence, ensemble forecasting (shared by the notebook and app.py)
web_template.html         Template for the static web app
export_web_app.py         Bakes the trained models into index.html
cyclone_model_0..4.keras  The five trained ensemble members
scaler.json               Standardisation statistics, feature indices, uncertainty-circle radii
demo_storms.json          Held-out storms used by the Hindcast tab
app_summary.json          Numbers shown on the Evidence tab
requirements.txt          Python dependencies for app.py
check_setup.py            Verifies the folder is complete and produces a test forecast
make_zip.py               Builds the submission ZIP
DEPLOY.md                 Free deployment options (GitHub Pages, Hugging Face Static Space, local, Colab)
notebooks/                Cyclone_Forecasting.ipynb (full pipeline with outputs), extra_test_cells.py
results/                  Result tables (CSV/JSON) and figures exported from the notebook
```

The project report and presentation are part of the course submission (ZIP on Google Classroom) and are not stored in this repository.

## Reproduce the training

1. Open `notebooks/Cyclone_Forecasting.ipynb` in Google Colab (GPU runtime) and choose **Run all**.
   Allow about 40-90 minutes the first time; most of it is downloading reanalysis fields (cached afterwards).
2. The notebook downloads the data (see **Data** below), builds 8-step windows split **by storm**
   (31,464 train / 7,034 validation / 6,622 test windows), trains the models, evaluates them and exports the model, JSON and result files.
3. To rebuild the web app from new models, run `python export_web_app.py` next to the exported files (see `DEPLOY.md`).

## Method in brief

- **Inputs:** 8 time steps x 19 features (position, wind, distance to land, season, motion, pressure, SST, shear, steering u/v).
- **Targets:** changes in latitude, longitude, wind and distance to land at four lead times.
- **Residual learning:** the network predicts the correction to a constant-motion persistence forecast.
- **Model selection** on validation data (MLP chosen); **5-seed ensemble** gives the forecast and its spread.
- **Uncertainty:** circle radius = 67th percentile of validation track errors per lead time; coverage checked on test storms.

## How the web app works

- `index.html` embeds the five trained MLPs (weights), the scaler, the held-out demo storms and the evidence tables in one file; there is no server.
- The browser code re-implements the feature engineering and the network forward pass. Its forecasts were checked against the Python
  implementation on 36 test cases (largest difference below 0.001, identical event probabilities).
- The default map draws coastlines with no tile service or API key; optional tile backgrounds are available in the Map menu.
- The rapid-intensification indicator is shown as experimental and low-skill, because its test ROC-AUC is close to chance.

## Data

The datasets are public and are downloaded automatically by the notebook; they are not stored in this repository (the storm file is about 145 MB).

| Dataset | Used for | Source |
|---|---|---|
| IBTrACS v04r01, `ibtracs.since1980.list.v04r01.csv` (4,677 storms, 138,122 six-hourly observations, 1980-2026) | Position, wind, pressure, distance to land | https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ |
| ERSSTv5 monthly SST (`sst.mnmean.nc`) | SST at the storm position | NOAA PSL, `Datasets/noaa.ersst.v5/` |
| NCEP/NCAR Reanalysis 1 daily winds (`uwnd`, `vwnd`, 850/500/200 hPa) | Vertical wind shear and steering flow | NOAA PSL, `Datasets/ncep.reanalysis.dailyavgs/pressure/` |

The environment experiments use storms from 2000 onward. Data downloaded on: 30/11/2026.

## Limitations

24 h track error (157.1 km) is well above operational numerical models; reanalysis fields are coarse (2.5 degrees, daily) and shear is
box-averaged; no satellite imagery; rapid intensification is not predictable by this model; wind definitions differ between agencies;
few North Indian Ocean storms; single data split and single ablation runs.

## References and libraries

- **Data:** IBTrACS (Knapp et al., 2010, NOAA NCEI); ERSSTv5 (Huang et al., 2017); NCEP/NCAR Reanalysis 1 (Kalnay et al., 1996).
- **Key sources:** Neumann (1972, CLIPER); DeMaria and Kaplan (1994, SHIPS); Rappaport et al. (2009); Alemany et al. (2019);
  Hochreiter and Schmidhuber (1997); Cho et al. (2014). The full list is in the project report.
- **Libraries:** TensorFlow/Keras, scikit-learn, pandas, NumPy, xarray, Plotly, Gradio, Matplotlib; Google Colab for training.

## Author contribution

**Author:** Manu Mathew Eldo. The following describes who did what.

- **Project definition and direction:** Manu Mathew chose the topic and the requirements, reviewed each design decision and iterated on the
  results.
- **Running, testing and deployment:** Manu Mathew ran the full pipeline on Google Colab (data download, training, evaluation), diagnosed and
  reported problems, exported the models, generated the web app, tested it, and deployed it on GitHub Pages. 
- **Documentation and presentation:** Manu Mathew reviewed, completed and finalised the project report and slides, and prepared the submission.
