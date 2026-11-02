"""Builds the ZIP for Google Classroom.   Run:  python make_zip.py  [zip_name]
The ZIP is created next to this folder. Caches, git data and very large files are left out."""
import pathlib, sys, zipfile

root = pathlib.Path(__file__).resolve().parent
name = sys.argv[1] if len(sys.argv) > 1 else "Cyclone_Forecasting_Project"
skip_parts = {".git", "__pycache__", ".ipynb_checkpoints"}
out = root.parent / f"{name}.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if p.is_dir() or any(part in skip_parts for part in rel.parts) or p.suffix in {".pyc", ".zip"}:
            continue
        if p.stat().st_size > 90e6:
            print("Skipped (over 90 MB):", rel); continue
        z.write(p, pathlib.Path(name) / rel)
print(f"Created {out}  ({out.stat().st_size / 1e6:.1f} MB)")
