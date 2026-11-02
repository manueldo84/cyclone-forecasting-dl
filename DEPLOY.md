# Deployment options (all free)

Hugging Face now charges for **Gradio** and **Docker** Spaces; only **Static** Spaces are free. The project therefore ships two versions of
the same app, which give the same forecasts:

| Version | Files | Where it runs |
|---|---|---|
| **Static web app** (recommended for hosting) | `index.html` (generated) | Any browser; free on a Hugging Face Static Space or GitHub Pages |
| Gradio app | `app.py` + model files | Your laptop, or a temporary Colab link |

## 1. Generate `index.html` (once)

The page embeds the five trained networks, so it needs no server. Create it from the files in your Colab download.

**In Colab** (if your runtime is still alive): upload `export_web_app.py` and `web_template.html` into `/content/outputs`, then run

```python
%cd /content/outputs
!python export_web_app.py
```

and download `/content/outputs/index.html`.

**On your laptop**: put `export_web_app.py`, `web_template.html`, the five `cyclone_model_*.keras`, `scaler.json`, `demo_storms.json` and
`app_summary.json` in one folder, then run `pip install tensorflow-cpu numpy` and `python export_web_app.py`.

**Map background:** the default map ("Coastlines and countries") needs no API key and no tile server; it only needs internet to load the
Plotly library and its coastline data. Some tile services have started requiring keys (CARTO shows an "API key required" watermark) or
block pages opened from a local file (OpenStreetMap). The tile options in the Map menu are therefore optional extras. If you are
offline, choose **Plain**: the tracks and circles still show.

**Test it first:** double-click `index.html`. It needs internet only for the map tiles and the Plotly library. Run a forecast on each tab.

## 2a. Host on a Hugging Face Static Space

1. huggingface.co, **New Space**: choose a name, **SDK = Static**, template **Blank**, visibility **Public**, then **Create Space**.
2. Open the **Files** tab, then **Add file**, then **Upload files**. Upload `index.html` (it replaces the default one) and commit.
3. The Space updates within seconds. Use its **Direct URL** (Embed this Space) or the Space page link in your report and slides.

## 2b. Host on GitHub Pages (also free, and it lives in your repository)

1. Put `index.html` at the **top level** of your GitHub repository and commit.
2. Repository **Settings, then Pages**: Source = *Deploy from a branch*, Branch = `main`, folder = `/ (root)`, then **Save**.
3. After about a minute the site is at `https://<username>.github.io/<repository>/`.

## 3. Run the Gradio version locally (backup for the live demo)

```bash
pip install -r requirements.txt
python app.py          # open http://127.0.0.1:7860
```

## 4. Temporary public link from Colab (emergency backup)

Run the last cell of the notebook; it prints a `gradio.live` link valid for about a week while the runtime stays connected.

## Checks before presenting

- Open the hosted link on your phone and on your laptop; run both tabs.
- Keep the local `index.html` or the local Gradio app ready in case the network fails.
- Save screenshots for the slides and report.
