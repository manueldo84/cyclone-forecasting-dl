"""Checks that the project folder is complete and that the app can produce a forecast.
Run:  python check_setup.py
"""
import pathlib, sys

HERE = pathlib.Path(__file__).parent
REQUIRED = ["app.py", "cyclone_utils.py", "scaler.json", "demo_storms.json", "app_summary.json", "requirements.txt", "README.md"]
missing = [f for f in REQUIRED if not (HERE / f).exists()]
models = sorted(HERE.glob("cyclone_model_*.keras"))
if not models:
    missing.append("cyclone_model_0.keras ... cyclone_model_4.keras")
if missing:
    print("MISSING FILES:", *missing, sep="\n  ")
    print("Copy them from the Colab download (cyclone_bundle.zip) into this folder.")
    sys.exit(1)
print(f"Found {len(models)} model files and all required project files.")
sys.path.insert(0, str(HERE))
import app  # loads the models and builds the interface; does not start the server
cards_html, figure, table = app.hindcast(app.NAMES[0], 14)
print(table.to_string(index=False))
print(f"\nOK: the app produced a forecast for {app.NAMES[0]}. Start it with:  python app.py")
