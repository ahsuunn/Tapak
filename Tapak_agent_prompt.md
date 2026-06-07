# TAPAK Location Intelligence Engine — Agent Prompt

## Pass this entire document to your coding agent (Claude Code, Cursor, etc.)

---

## CONTEXT

You are building the ML pipeline for TAPAK, a smart city system to identify optimal street vendor (PKL) locations in Bandung, Indonesia. This is for a university competition (GEMASTIK). The pipeline goes from raw inputs to a trained XGBoost model that outputs a suitability score per 500m grid cell across Bandung.

The developer has ONE manually collected file:

- `population_density.csv` — BPS data with columns:
  - `Kecamatan` (sub-district name)
  - `Jumlah Penduduk (Ribu)` (population in thousands)
  - `Laju Pertumbuhan Penduduk per Tahun` (annual growth rate)
  - `Persentase Penduduk` (population percentage)
  - `Kepadatan Penduduk per km persegi (Km2)` (population density per km²)
  - `Rasio Jenis Kelamin Penduduk` (sex ratio)

All other data is sourced from OpenStreetMap via the Overpass API (called programmatically) or fabricated using literature-grounded synthetic methods described below.

---

## YOUR TASK

Build a single, well-commented Python script called `tapak_pipeline.py` that does ALL of the following steps in order. Also produce a `requirements.txt`. The script must be written so it can be trivially converted to a Jupytext notebook (each logical section separated by `# %% [markdown]` and `# %%` cell markers as comments).

---

## STEP 0 — Environment Setup

Install and import all required libraries. At the top of the script, include a commented block:

```
# pip install osmnx geopandas shapely pandas numpy scikit-learn xgboost shap folium matplotlib seaborn requests jupytext
```

Required imports:

- `osmnx`, `geopandas`, `shapely`, `pandas`, `numpy`
- `xgboost`, `sklearn` (train_test_split, metrics)
- `shap`, `folium`, `matplotlib`, `seaborn`
- `requests`, `json`, `time`, `os`, `warnings`

---

## STEP 1 — Configuration Block

At the top, define a single CONFIG dict so the developer can tweak everything in one place:

```python
CONFIG = {
    # Bandung bounding box [west, south, east, north]
    "bbox": [107.55, -7.0, 107.70, -6.85],

    # Grid cell size in degrees (~500m)
    "cell_size": 0.0045,

    # Overpass API endpoint
    "overpass_url": "https://overpass-api.de/api/interpreter",

    # Overpass query timeout (seconds)
    "overpass_timeout": 60,

    # Path to manually collected CSV
    "population_csv": "population_density.csv",

    # Output paths
    "output_dir": "output/",
    "model_path": "output/tapak_model.json",
    "grid_output": "output/grid_scored.csv",
    "map_output": "output/bandung_vendor_zones.html",

    # Model
    "test_size": 0.2,
    "random_state": 42,

    # Synthetic data random seed (for reproducibility)
    "synthetic_seed": 2024,

    # Road traffic weights (road class -> relative traffic intensity)
    "road_weights": {
        "primary": 6.0,
        "primary_link": 5.0,
        "secondary": 4.0,
        "secondary_link": 3.5,
        "tertiary": 2.5,
        "tertiary_link": 2.0,
        "residential": 1.0,
        "service": 0.5,
        "unclassified": 0.8,
        "living_street": 0.6
    }
}
```

Create the output directory if it doesn't exist.

---

## STEP 2 — Create the Spatial Grid

Create a regular 500m × 500m grid of rectangular cells covering the Bandung bounding box.

Each cell is a Shapely `box`. Assign a unique `cell_id` (integer). The result is a GeoDataFrame in `EPSG:4326` with columns: `cell_id`, `geometry`.

Also compute and store the centroid of each cell as `centroid_lat` and `centroid_lon` columns (for later spatial joins and distance calculations).

Print: total number of cells created.

---

## STEP 3 — Load & Process Population Density (Real Data)

Load `population_density.csv`. Handle the following:

- Strip whitespace from the `Kecamatan` column
- The density column `Kepadatan Penduduk per km persegi (Km2)` may have dots as thousand separators and commas as decimal separators (Indonesian number format) — clean and convert to float
- Rename columns to clean snake_case for easier handling

To assign population density to grid cells, use a spatial lookup:

- Load a GeoJSON or shapefile of Bandung kecamatan boundaries using `osmnx.geocode_to_gdf` or by downloading from OSM with `ox.features_from_place("Bandung, Indonesia", tags={"boundary": "administrative", "admin_level": "7"})`
- If OSM boundary download fails, fall back to: spatially join grid cell centroids to kecamatan polygons using a Voronoi approximation based on known kecamatan centroid coordinates (hardcode the 30 kecamatan centroids as a fallback lookup table)
- Spatial join: for each grid cell centroid, find which kecamatan it falls in, then assign that kecamatan's density value
- Normalize the density to 0–1 range (min-max scaling) → column: `pop_density_norm`

Print: summary stats of pop_density_norm.

---

## STEP 4 — Query OSM via Overpass API

Write a robust `query_overpass(query_string)` helper function that:

- POSTs to the Overpass API
- Handles HTTP errors and timeouts with retry logic (3 retries, exponential backoff)
- Returns the parsed JSON response
- Adds a 2-second sleep between queries to avoid rate limiting

Run the following **separate queries** (do not combine into one — Overpass handles them more reliably separately). After each query, parse the result into a GeoDataFrame with columns `['geometry', 'amenity_type', 'lat', 'lon']`.

### Query 4A — Commercial & Transport POIs

```
[out:json][timeout:60][bbox:-7.0,107.55,-6.85,107.70];
(
  node[amenity=school];
  way[amenity=school];
  node[amenity=university];
  way[amenity=university];
  node[amenity=marketplace];
  way[amenity=marketplace];
  node[amenity=bus_station];
  way[amenity=bus_station];
  node[amenity=food_court];
  way[amenity=food_court];
  node[shop=supermarket];
  way[shop=supermarket];
  node[shop=mall];
  way[shop=mall];
);
out center;
```

### Query 4B — Health & Civic (for negative label logic)

```
[out:json][timeout:60][bbox:-7.0,107.55,-6.85,107.70];
(
  node[amenity=hospital];
  way[amenity=hospital];
  node[amenity=place_of_worship];
  way[amenity=place_of_worship];
  node[amenity=government];
  way[amenity=government];
);
out center;
```

### Query 4C — Parks & Open Space (for negative label logic)

```
[out:json][timeout:60][bbox:-7.0,107.55,-6.85,107.70];
(
  way[leisure=park];
  relation[leisure=park];
  way[landuse=grass];
  way[landuse=recreation_ground];
);
out center;
```

### Query 4D — Road Network

```
[out:json][timeout:60][bbox:-7.0,107.55,-6.85,107.70];
(
  way[highway=primary];
  way[highway=primary_link];
  way[highway=secondary];
  way[highway=secondary_link];
  way[highway=tertiary];
  way[highway=tertiary_link];
  way[highway=residential];
  way[highway=service];
  way[highway=unclassified];
);
out geom;
```

For ways in 4D, reconstruct LineString geometries from the node coordinates in `out geom` response. Each way element has a `geometry` array of `{lat, lon}` objects — convert these to Shapely LineStrings.

After all queries, print count of features retrieved per category.

**Overpass fallback:** If any query fails after 3 retries, print a warning and continue with an empty GeoDataFrame for that category. The pipeline must not crash on API failures.

---

## STEP 5 — Compute Real Features per Grid Cell

### Feature 5A: POI Cluster Score

Using the commercial POIs from Query 4A:

- For each grid cell, count how many POI points fall within it (spatial join)
- Apply log1p transform to reduce outlier effect: `log1p(count)`
- Normalize 0–1 → column: `poi_score`

### Feature 5B: Traffic Flow Score

Using roads from Query 4D:

- Project both roads and grid to `EPSG:32748` (UTM zone 48S — correct for Bandung) for accurate length calculation
- For each grid cell, sum `road_length_in_cell × road_weight` for all road segments intersecting the cell
- Normalize 0–1 → column: `traffic_score`

Handle the case where a road crosses multiple cells by clipping the road geometry to each cell before measuring length.

---

## STEP 6 — Fabricate Synthetic Features

These features have no public data source. Generate them using population-proxy models with added realistic noise. Use `CONFIG["synthetic_seed"]` for reproducibility.

### Feature 6A: Cellular Signal Density

Rationale: Signal infrastructure deployment follows population density logarithmically (ITU-R M.2135 urban coverage model).

```python
def compute_signal_density(pop_density_norm, poi_score, rng):
    # Base: log-transform of population density
    base = np.log1p(pop_density_norm * 50000) / np.log1p(50000)

    # Modifier: commercial areas have denser small-cell deployment
    commercial_boost = poi_score * 0.15

    # Realistic noise (sigma=0.04 from field measurement variance literature)
    noise = rng.normal(0, 0.04, size=len(pop_density_norm))

    signal = base + commercial_boost + noise
    return np.clip(signal, 0, 1)
```

Output column: `signal_score`

### Feature 6B: Construction Intensity Index

Rationale: Development activity concentrates along urban growth corridors and commercial edges. Bandung's main corridors: Gedebage (east), Summarecon (north), Dago upper (north), Buah Batu (south).

```python
def compute_construction_index(grid, rng):
    # Known Bandung development corridor centroids [lon, lat]
    corridors = [
        (107.693, -6.938),  # Gedebage Teknopolis
        (107.640, -6.855),  # Summarecon Bandung
        (107.617, -6.877),  # Dago upper
        (107.645, -6.953),  # Buah Batu
        (107.577, -6.921),  # Pasteur / Sukajadi
    ]

    # For each cell centroid, compute min distance to any corridor (in degrees, approx)
    def min_dist_to_corridors(lon, lat):
        return min(((lon-cx)**2 + (lat-cy)**2)**0.5 for cx, cy in corridors)

    distances = grid.apply(
        lambda row: min_dist_to_corridors(row['centroid_lon'], row['centroid_lat']), axis=1
    )

    # Inverse distance, normalized + noise
    base = 1 / (1 + distances * 100)
    noise = rng.exponential(0.05, size=len(grid))
    index = base + noise

    # Normalize
    index = (index - index.min()) / (index.max() - index.min())
    return np.clip(index, 0, 1)
```

Output column: `construction_index`

### Feature 6C: Existing Vendor Hotspot Density

Rationale: Vendors cluster around established commercial corridors. Known Bandung PKL hotspot areas: Jl. Cihampelas, Jl. Dago, Jl. Braga, Alun-alun, Pasar Baru, Jl. Riau (Diponegoro), UNPAD/ITB vicinity, Jl. Soekarno-Hatta.

```python
def compute_vendor_hotspot_density(grid, rng):
    # Known PKL concentration centroids [lon, lat, relative_intensity]
    hotspots = [
        (107.608, -6.900, 1.0),   # Jl. Cihampelas
        (107.616, -6.888, 0.9),   # Jl. Dago / ITB area
        (107.607, -6.917, 0.9),   # Alun-alun Bandung
        (107.610, -6.915, 0.85),  # Pasar Baru
        (107.618, -6.908, 0.8),   # Jl. Braga
        (107.630, -6.902, 0.75),  # Jl. Riau / Diponegoro
        (107.652, -6.917, 0.7),   # Jl. Soekarno-Hatta east
        (107.571, -6.918, 0.65),  # Jl. Pasteur
        (107.638, -6.893, 0.6),   # UNPAD Dipatiukur
        (107.596, -6.926, 0.6),   # Jl. Moh. Toha / Pasar Caringin
    ]

    def hotspot_score(lon, lat):
        # Gaussian decay from each hotspot center
        score = 0
        for hx, hy, intensity in hotspots:
            dist_sq = (lon - hx)**2 + (lat - hy)**2
            score += intensity * np.exp(-dist_sq / (2 * 0.003**2))  # sigma ~300m
        return score

    scores = grid.apply(
        lambda row: hotspot_score(row['centroid_lon'], row['centroid_lat']), axis=1
    )

    noise = rng.exponential(0.05, size=len(grid))
    scores = scores + noise
    scores = (scores - scores.min()) / (scores.max() - scores.min())
    return np.clip(scores, 0, 1)
```

Output column: `vendor_hotspot_score`

Print: mean and std of each fabricated feature to verify distributions look reasonable.

---

## STEP 7 — Rule-Based Labeling

Construct binary labels using knowledge-driven rules derived from Perda Kota Bandung PKL regulations and urban suitability principles. This is the ground truth proxy.

### Positive label conditions (label = 1, "suitable zone"):

A cell is labeled suitable if it meets ALL of:

- `poi_score > 0.25` (has meaningful commercial/civic infrastructure nearby)
- `traffic_score > 0.15` (accessible by road, but not a major highway)
- `pop_density_norm > 0.3` (sufficient residential catchment)
- NOT within 80m of a hospital or place of worship (use Query 4B results — buffer and check)
- NOT primarily covered by park/open space (use Query 4C results)
- NOT on a primary highway (traffic_score very high but no POI diversity — exclude cells where `traffic_score > 0.85` AND `poi_score < 0.2`)

### Negative label conditions (label = 0, "unsuitable"):

Everything that doesn't meet the positive conditions.

Add a `label_confidence` column (0–1) that reflects how strongly the cell passes or fails the thresholds — this will be useful for SHAP visualization.

Print: count of label=1 and label=0 cells. If label=1 is less than 15% of total cells, lower the poi_score threshold by 0.05 and recalculate. Print the final class distribution.

---

## STEP 8 — Assemble Feature Matrix

Combine all features into a single DataFrame:

```
cell_id | centroid_lat | centroid_lon | pop_density_norm | poi_score |
traffic_score | signal_score | construction_index | vendor_hotspot_score | label
```

Drop any rows with NaN values. Print shape of final feature matrix.

Save this as `output/feature_matrix.csv` for inspection.

---

## STEP 9 — Train XGBoost Model

Split data 80/20 stratified on label.

Train XGBoost with these starting hyperparameters (reasonable for this dataset size):

```python
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=...,   # set to (count label=0 / count label=1) for class imbalance
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=CONFIG['random_state']
)
```

Fit with early stopping on validation set (eval_set, early_stopping_rounds=20).

After training, print and save:

- Classification report (precision, recall, F1)
- ROC-AUC score
- Confusion matrix (as seaborn heatmap → save to `output/confusion_matrix.png`)

Save the trained model to `CONFIG['model_path']`.

---

## STEP 10 — SHAP Explainability

Generate SHAP values for the test set.

Produce and save these three plots to `output/`:

1. `shap_summary_beeswarm.png` — beeswarm plot of all features (for the paper, BAB 3.2.5)
2. `shap_bar_importance.png` — mean absolute SHAP bar chart (feature importance ranking)
3. `shap_waterfall_sample.png` — waterfall plot for a single high-scoring cell (pick the cell with the highest predicted suitability score)

Print the top 3 most important features by mean |SHAP value|.

---

## STEP 11 — Score All Grid Cells & Output

Run `predict_proba` on the full grid (all cells, not just test set) to get suitability scores.

Add to the grid DataFrame:

- `suitability_score` (0–1, probability of label=1)
- `zone_class`: `"Highly Suitable"` (>0.65), `"Moderately Suitable"` (0.35–0.65), `"Not Suitable"` (<0.35)

Save `output/grid_scored.csv` with all columns.

---

## STEP 12 — Folium Interactive Map

Create an interactive Folium choropleth map.

Color scheme:

- Highly Suitable: `#2ecc71` (green)
- Moderately Suitable: `#f39c12` (orange)
- Not Suitable: `#e74c3c` (red, semi-transparent)

For each grid cell:

- Draw as a rectangle using its bounding box coordinates
- Fill with the appropriate color, `fillOpacity=0.45`, `weight=0.3`
- Add a popup showing: `cell_id`, `suitability_score` (2 decimal places), `zone_class`, and top contributing feature

Also add:

- A tile layer (use OpenStreetMap tiles)
- A legend in the bottom-right corner (HTML-based, embedded in the map)
- Bandung city center marker at `[-6.9175, 107.6191]`

Save to `CONFIG['map_output']`. Print: "Map saved to output/bandung_vendor_zones.html — open in browser to view."

---

## STEP 13 — Sanity Check

After generating the map, run an automated sanity check. Print a report to console:

```
=== TAPAK SANITY CHECK ===
Known PKL hotspot validation:
- Jl. Cihampelas area (lon=107.608, lat=-6.900): [score] → [PASS/FAIL — expect >0.5]
- Jl. Dago / ITB area (lon=107.616, lat=-6.888): [score] → [PASS/FAIL]
- Alun-alun Bandung (lon=107.607, lat=-6.917): [score] → [PASS/FAIL]
- Pasar Baru (lon=107.610, lat=-6.915): [score] → [PASS/FAIL]

For each location: find the grid cell whose centroid is closest to the coordinate,
report its suitability_score and zone_class.

Overall: X/4 known hotspots correctly classified as Suitable or Highly Suitable.
If score < 2/4: print WARNING — model may need threshold adjustment.
```

---

## STEP 14 — Jupytext Headers

Throughout the script, use these comment markers so it can be converted to a Jupyter notebook with `jupytext --to notebook tapak_pipeline.py`:

```python
# %% [markdown]
# ## Step N — Title

# %%
# actual code here
```

Each of the 14 steps above should be its own cell block.

---

## OUTPUT CHECKLIST

When done, the `output/` directory must contain:

- `feature_matrix.csv` — full feature matrix with labels
- `grid_scored.csv` — all grid cells with suitability scores
- `tapak_model.json` — trained XGBoost model
- `confusion_matrix.png`
- `shap_summary_beeswarm.png`
- `shap_bar_importance.png`
- `shap_waterfall_sample.png`
- `bandung_vendor_zones.html` — interactive map

---

## IMPORTANT CONSTRAINTS

1. The script must run end-to-end with only `population_density.csv` as input. Everything else is fetched or fabricated.
2. Every fabricated feature must have a one-line comment citing its real-world basis (ITU-R, Clark's Law, Perda Bandung, etc.)
3. No step should crash the entire pipeline — wrap Overpass queries and file operations in try/except with meaningful fallback behavior.
4. All random operations use seeded RNGs (`np.random.default_rng(CONFIG['synthetic_seed'])`) for full reproducibility.
5. Print progress updates at the start of each step: `print(f"\n{'='*50}\nSTEP N: Title\n{'='*50}")`.
6. The final print statement should be a summary table of all output files with their sizes.
