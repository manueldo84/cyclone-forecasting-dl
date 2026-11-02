"""Exports the trained ensemble to ONE self-contained web page (index.html) that runs entirely in the browser.
No server and no TensorFlow are needed to VIEW the page, so it can be hosted free on a Hugging Face *Static* Space
or on GitHub Pages.

Run in the folder that contains cyclone_model_*.keras, scaler.json, demo_storms.json, app_summary.json and web_template.html:

    python export_web_app.py            (needs: pip install tensorflow-cpu numpy)

Creates  index.html  (a few MB). Works for MLP ensembles (the model chosen in the notebook).
"""
import base64
import json
import pathlib
import sys

import numpy as np
from tensorflow import keras

HERE = pathlib.Path(__file__).parent
OUT = HERE / (sys.argv[1] if len(sys.argv) > 1 else "index.html")


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode()


paths = sorted(HERE.glob("cyclone_model_*.keras"))
if not paths:
    sys.exit("No cyclone_model_*.keras files found next to this script.")

members = []
for p in paths:
    model = keras.models.load_model(p)
    if any(isinstance(l, (keras.layers.LSTM, keras.layers.GRU)) for l in model.layers):
        sys.exit(f"{p.name} is a recurrent model. This exporter supports the MLP ensemble only "
                 f"(the notebook picks the architecture on validation data; yours should have chosen MLP).")
    layers = []
    for l in model.layers:
        if not isinstance(l, keras.layers.Dense):
            continue
        w, b = l.get_weights()
        layers.append({"w": b64(w), "b": b64(b), "n_in": int(w.shape[0]), "n_out": int(w.shape[1]),
                       "act": "relu" if l.activation.__name__ == "relu" else "linear"})
    members.append(layers)
    print(f"{p.name}: {len(layers)} dense layers, {sum(x['n_in'] * x['n_out'] for x in layers):,} weights")

data = {
    "models": members,
    "scaler": json.load(open(HERE / "scaler.json")),
    "storms": json.load(open(HERE / "demo_storms.json")),
    "summary": json.load(open(HERE / "app_summary.json")),
}
html = (HERE / "web_template.html").read_text(encoding="utf-8").replace("__DATA__", json.dumps(data, separators=(",", ":")))
OUT.write_text(html, encoding="utf-8")
print(f"Wrote {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB). Open it in a browser, or upload it to a Static Space / GitHub Pages.")
