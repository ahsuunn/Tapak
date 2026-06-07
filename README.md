# TAPAK — Street Vendor Location Intelligence Engine

> **GEMASTIK Competition Project** · Smart City · Machine Learning · Geospatial Analysis

TAPAK is an end-to-end ML pipeline that identifies optimal street vendor (PKL) locations across Bandung, Indonesia. It ingests real population data, fetches OpenStreetMap features, engineers geospatial features, trains an XGBoost classifier, and outputs an interactive choropleth map of suitability zones.

---

## Demo Output

The pipeline produces an interactive Folium map (`output/bandung_vendor_zones.html`) color-coded by suitability:

| Zone | Color | Score |
|------|-------|-------|
| 🟢 Highly Suitable | `#2ecc71` | > 0.65 |
| 🟡 Moderately Suitable | `#f39c12` | 0.35 – 0.65 |
| 🔴 Not Suitable | `#e74c3c` | < 0.35 |

---

## Project Structure

```
Tapak/
├── src/
│   ├── tapak_pipeline.py          # Main ML pipeline (Steps 1–14)
│   └── fetch_overpass_data.js     # Browser console script to fetch OSM data
├── data/
│   ├── population_density.csv     # BPS Bandung kecamatan population data (manual)
│   └── overpass/                  # Cached Overpass API responses (auto-generated)
│       ├── overpass_4a_poi.json
│       ├── overpass_4b_civic.json
│       ├── overpass_4c_parks.json
│       └── overpass_4d_roads.json
├── output/                        # All generated artifacts
│   ├── feature_matrix.csv
│   ├── grid_scored.csv
│   ├── tapak_model.json
│   ├── confusion_matrix.png
│   ├── shap_summary_beeswarm.png
│   ├── shap_bar_importance.png
│   ├── shap_waterfall_sample.png
│   └── bandung_vendor_zones.html
├── pyproject.toml
└── README.md
```

---

## Quickstart

### 1. Clone & install dependencies

```bash
git clone https://github.com/ahsuunn/Tapak.git
cd Tapak
uv sync
```

> Requires [uv](https://docs.astral.sh/uv/). Alternatively use `pip install -r requirements.txt`.

### 2. Fetch OSM data (one-time)

The pipeline queries OpenStreetMap via the Overpass API. Because the Overpass public endpoint can be rate-limited, we provide a browser-based fetcher that bypasses those restrictions.

**Option A — Browser console (recommended)**

1. Open your browser and go to any page (e.g. `https://overpass-api.de`)
2. Press **F12** → **Console** tab
3. Paste the full contents of [`src/fetch_overpass_data.js`](src/fetch_overpass_data.js) and press **Enter**
4. Four JSON files will auto-download. Move them to `data/overpass/`:

```
data/overpass/overpass_4a_poi.json      ← Commercial & Transport POIs
data/overpass/overpass_4b_civic.json    ← Health & Civic features
data/overpass/overpass_4c_parks.json    ← Parks & open space
data/overpass/overpass_4d_roads.json    ← Road network
```

**Option B — Automatic (pipeline handles it)**

If no cache files are found, the pipeline tries:
1. Overpass API (3 mirrors with retry)
2. osmnx fallback (uses its own robust HTTP stack)

### 3. Run the pipeline

```bash
uv run .\src\tapak_pipeline.py
```

On a successful run you will see output like:

```
==================================================
STEP 9: Train XGBoost Model
==================================================
              precision    recall  f1-score
  Unsuitable       0.99      0.93      0.96
    Suitable       0.62      0.96      0.75
    accuracy                           0.93

ROC-AUC: 0.9737
```

Open `output/bandung_vendor_zones.html` in your browser to explore the interactive map.

---

## Pipeline Steps

| Step | Description |
|------|-------------|
| **1** | Configuration block (`CONFIG` dict) |
| **2** | Create 500m × 500m spatial grid over Bandung |
| **3** | Load & normalize BPS population density data |
| **4** | Fetch OSM data (POIs, roads, parks) via Overpass API |
| **5** | Compute real features: `poi_score`, `traffic_score` |
| **6** | Fabricate synthetic features: `signal_score`, `construction_index`, `vendor_hotspot_score` |
| **7** | Rule-based labeling (Perda PKL Bandung regulations) |
| **8** | Assemble feature matrix → `output/feature_matrix.csv` |
| **9** | Train XGBoost classifier with early stopping |
| **10** | SHAP explainability plots |
| **11** | Score all grid cells → `output/grid_scored.csv` |
| **12** | Generate Folium interactive map |
| **13** | Automated sanity check against known PKL hotspots |
| **14** | Output file summary |

---

## Features Used

| Feature | Source | Description |
|---------|--------|-------------|
| `pop_density_norm` | BPS (real) | Normalized population density per kecamatan |
| `poi_score` | OSM (real) | Log-normalized count of commercial POIs per cell |
| `traffic_score` | OSM (real) | Weighted road length per cell (UTM projected) |
| `signal_score` | Synthetic | ITU-R M.2135 urban coverage model proxy |
| `construction_index` | Synthetic | Clark's Law urban growth corridor proximity |
| `vendor_hotspot_score` | Synthetic | Gaussian decay from known PKL hotspots (Perda Bandung) |

---

## Model Performance (Sample Run)

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.9737** |
| Accuracy | 93% |
| Suitable Recall | 96% |
| Top Feature (SHAP) | `poi_score` |

---

## Configuration

All tunable parameters live in the `CONFIG` dict at the top of `tapak_pipeline.py`:

```python
CONFIG = {
    "bbox": [107.55, -7.0, 107.70, -6.85],   # Bandung bounding box
    "cell_size": 0.0045,                        # ~500m grid resolution
    "test_size": 0.2,
    "random_state": 42,
    "synthetic_seed": 2024,
    ...
}
```

---

## Dependencies

Managed via `uv` / `pyproject.toml`:

| Package | Purpose |
|---------|---------|
| `osmnx` | OSM boundary & road network fetching |
| `geopandas` + `shapely` | Spatial grid & geometry operations |
| `xgboost` | Gradient boosted classifier |
| `shap` | Model explainability |
| `folium` | Interactive HTML map |
| `scikit-learn` | Train/test split, metrics |
| `pandas` + `numpy` | Data wrangling |
| `matplotlib` + `seaborn` | Static plots |
| `requests` | Overpass API HTTP calls |
| `jupytext` | Convert pipeline to Jupyter notebook |

Install all:
```bash
uv sync
# or
pip install osmnx geopandas shapely pandas numpy scikit-learn xgboost shap folium matplotlib seaborn requests jupytext
```

---

## Convert to Jupyter Notebook

The pipeline uses Jupytext cell markers (`# %%`) throughout:

```bash
uv run jupytext --to notebook src/tapak_pipeline.py
```

This produces `src/tapak_pipeline.ipynb` which can be opened in Jupyter or VS Code.

---

## License

MIT
