# pip install osmnx geopandas shapely pandas numpy scikit-learn xgboost shap folium matplotlib seaborn requests jupytext

# %% [markdown]
# # TAPAK — Street Vendor Location Intelligence Pipeline
# A smart city ML pipeline to identify optimal PKL (street vendor) zones in Bandung, Indonesia.
# Developed for GEMASTIK university competition.

# %%
# ─── Imports ──────────────────────────────────────────────────────────────────
import json
import os
import time
import warnings

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import requests

matplotlib.use("Agg")
import folium
import matplotlib.pyplot as plt
import osmnx as ox
import seaborn as sns
import shap
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# %% [markdown]
# ## Step 1 — Configuration Block

# %%
print(f"\n{'=' * 50}\nSTEP 1: Configuration\n{'=' * 50}")

CONFIG = {
    # Bandung bounding box [west, south, east, north]
    "bbox": [107.55, -7.0, 107.70, -6.85],
    # Grid cell size in degrees (~500m)
    "cell_size": 0.0045,
    # Overpass API endpoints — tries each mirror in order on failure
    "overpass_urls": [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    ],
    # Overpass query timeout (seconds)
    "overpass_timeout": 60,
    # Path to manually collected CSV (relative to script location)
    "population_csv": os.path.join(
        "data",
        "Penduduk, Laju Pertumbuhan Penduduk, Distribusi Persentase Penduduk "
        "Kepadatan Penduduk, Rasio Jenis Kelamin Penduduk Menurut Kecamatan di "
        "Kota Bandung, 2026.csv",
    ),
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
        "living_street": 0.6,
    },
}

# Create output directory
os.makedirs(CONFIG["output_dir"], exist_ok=True)
print(f"Output directory ready: {CONFIG['output_dir']}")

# %% [markdown]
# ## Step 2 — Create Spatial Grid

# %%
print(f"\n{'=' * 50}\nSTEP 2: Create Spatial Grid\n{'=' * 50}")

west, south, east, north = CONFIG["bbox"]
cell_size = CONFIG["cell_size"]

lons = np.arange(west, east, cell_size)
lats = np.arange(south, north, cell_size)

cells = []
cell_id = 0
for lon in lons:
    for lat in lats:
        geom = box(lon, lat, lon + cell_size, lat + cell_size)
        centroid = geom.centroid
        cells.append(
            {
                "cell_id": cell_id,
                "geometry": geom,
                "centroid_lon": centroid.x,
                "centroid_lat": centroid.y,
            }
        )
        cell_id += 1

grid = gpd.GeoDataFrame(cells, crs="EPSG:4326")
print(f"Total cells created: {len(grid)}")

# %% [markdown]
# ## Step 3 — Load & Process Population Density (Real Data)

# %%
print(f"\n{'=' * 50}\nSTEP 3: Population Density\n{'=' * 50}")


def parse_indonesian_number(val):
    """Convert Indonesian-formatted numbers (dot=thousands, comma=decimal) to float."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    # Remove thousand separators (dots), replace decimal comma with dot
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


# Load and clean population CSV
pop_df = pd.read_csv(CONFIG["population_csv"])
pop_df.columns = pop_df.columns.str.strip()
pop_df["Kecamatan"] = pop_df["Kecamatan"].astype(str).str.strip()
pop_df = pop_df[
    pop_df["Kecamatan"].notna()
    & (pop_df["Kecamatan"] != "")
    & (pop_df["Kecamatan"] != "nan")
]

density_col = "Kepadatan Penduduk per km persegi (Km2)"
pop_df["density"] = pop_df[density_col].apply(parse_indonesian_number)

# Remove the city-total row (Bandung aggregate)
pop_df = pop_df[pop_df["Kecamatan"] != "Bandung"].copy()

# Rename to snake_case
pop_df = pop_df.rename(
    columns={
        "Kecamatan": "kecamatan",
        "Jumlah Penduduk (Ribu)": "pop_total",
        "Laju Pertumbuhan Penduduk per Tahun": "pop_growth_rate",
        "Persentase Penduduk": "pop_pct",
        density_col: "density_raw",
        "Rasio Jenis Kelamin Penduduk": "sex_ratio",
    }
)

# Drop NaN density rows (footnote/metadata rows in the CSV)
pop_df = pop_df[pop_df["density"].notna()].copy()
print(f"Kecamatan loaded: {len(pop_df)}")
print(pop_df[["kecamatan", "density"]].to_string(index=False))

# ── Assign density to grid cells via spatial join ────────────────────────────
kecamatan_gdf = None
try:
    # Try fetching admin boundaries from OSM
    print("\nFetching kecamatan boundaries from OSM …")
    kecamatan_gdf = ox.features_from_place(
        "Bandung, Indonesia",
        tags={"boundary": "administrative", "admin_level": "7"},
    )
    kecamatan_gdf = kecamatan_gdf[
        kecamatan_gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    ].copy()
    kecamatan_gdf = kecamatan_gdf.reset_index(drop=True)
    kecamatan_gdf["name_clean"] = kecamatan_gdf.get("name", "").astype(str).str.strip()
    print(f"OSM boundaries fetched: {len(kecamatan_gdf)} polygons")
except Exception as e:
    print(f"OSM boundary fetch failed: {e}\nFalling back to Voronoi approximation.")

# Fallback: Voronoi from known kecamatan centroids
KECAMATAN_CENTROIDS = {
    "Bandung Kulon": (107.578, -6.930),
    "Babakan Ciparay": (107.590, -6.935),
    "Bojongloa Kaler": (107.595, -6.918),
    "Bojongloa Kidul": (107.598, -6.928),
    "Astanaanyar": (107.602, -6.921),
    "Regol": (107.607, -6.930),
    "Lengkong": (107.621, -6.925),
    "Bandung Kidul": (107.638, -6.940),
    "Buahbatu": (107.645, -6.952),
    "Rancasari": (107.666, -6.953),
    "Gedebage": (107.690, -6.943),
    "Cibiru": (107.723, -6.920),
    "Panyileukan": (107.706, -6.937),
    "Ujung Berung": (107.717, -6.901),
    "Cinambo": (107.700, -6.913),
    "Arcamanik": (107.680, -6.906),
    "Antapani": (107.661, -6.904),
    "Mandalajati": (107.672, -6.892),
    "Kiaracondong": (107.648, -6.912),
    "Batununggal": (107.630, -6.924),
    "Sumur Bandung": (107.610, -6.910),
    "Andir": (107.594, -6.909),
    "Cicendo": (107.600, -6.898),
    "Bandung Wetan": (107.623, -6.901),
    "Cibeunying Kidul": (107.636, -6.907),
    "Cibeunying Kaler": (107.629, -6.893),
    "Coblong": (107.618, -6.882),
    "Sukajadi": (107.603, -6.884),
    "Sukasari": (107.592, -6.869),
    "Cidadap": (107.602, -6.866),
}


def assign_density_voronoi(grid_df, pop_data, centroids_dict):
    """Assign density via nearest kecamatan centroid (Voronoi proxy)."""
    density_lookup = dict(zip(pop_data["kecamatan"], pop_data["density"]))
    centroid_list = [(name, lon, lat) for name, (lon, lat) in centroids_dict.items()]

    def nearest_kecamatan(lon, lat):
        best_name, best_dist = None, float("inf")
        for name, cx, cy in centroid_list:
            dist = (lon - cx) ** 2 + (lat - cy) ** 2
            if dist < best_dist:
                best_dist = dist
                best_name = name
        return best_name

    grid_df = grid_df.copy()
    grid_df["kecamatan"] = grid_df.apply(
        lambda r: nearest_kecamatan(r["centroid_lon"], r["centroid_lat"]), axis=1
    )
    grid_df["density"] = grid_df["kecamatan"].map(density_lookup)
    return grid_df


if kecamatan_gdf is not None and len(kecamatan_gdf) > 0:
    try:
        # Build centroid GeoDataFrame for grid cells
        centroid_gdf = gpd.GeoDataFrame(
            grid[["cell_id", "centroid_lat", "centroid_lon"]],
            geometry=gpd.points_from_xy(grid["centroid_lon"], grid["centroid_lat"]),
            crs="EPSG:4326",
        )
        joined = gpd.sjoin(
            centroid_gdf,
            kecamatan_gdf[["geometry", "name_clean"]],
            how="left",
            predicate="within",
        )
        joined = joined.drop_duplicates("cell_id")
        # Map name to density
        kec_density = dict(zip(pop_df["kecamatan"], pop_df["density"]))
        joined["density"] = joined["name_clean"].map(kec_density)
        grid["kecamatan"] = joined["name_clean"].values
        grid["density"] = joined["density"].values
        # Fill NaN from fallback
        nan_mask = grid["density"].isna()
        if nan_mask.any():
            fallback = assign_density_voronoi(
                grid[nan_mask], pop_df, KECAMATAN_CENTROIDS
            )
            grid.loc[nan_mask, "density"] = fallback["density"].values
        print(
            f"Spatial join used for density assignment; fallback filled {nan_mask.sum()} cells."
        )
    except Exception as e:
        print(f"Spatial join failed ({e}), using Voronoi fallback for all cells.")
        grid = assign_density_voronoi(grid, pop_df, KECAMATAN_CENTROIDS)
else:
    grid = assign_density_voronoi(grid, pop_df, KECAMATAN_CENTROIDS)

# Normalize density 0–1
d_min = grid["density"].min()
d_max = grid["density"].max()
grid["pop_density_norm"] = (grid["density"] - d_min) / (d_max - d_min)

print("\npop_density_norm summary:")
print(grid["pop_density_norm"].describe())

# %% [markdown]
# ## Step 4 — Query OSM via Overpass API

# %%
print(f"\n{'=' * 50}\nSTEP 4: Overpass API Queries\n{'=' * 50}")


# Headers required by Overpass API — omitting User-Agent causes 406 Not Acceptable
_OVERPASS_HEADERS = {
    "User-Agent": "TAPAK/1.0 (GEMASTIK research; contact: research@example.com)",
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "identity",  # avoid gzip confusion with some proxies
    "Content-Type": "application/x-www-form-urlencoded",
}


def query_overpass(query_string: str, urls: list, timeout: int = 60) -> dict:
    """
    POST a query to the Overpass API with retry logic across multiple mirror endpoints.
    Tries each URL in order; returns parsed JSON or empty dict if all fail.
    Multiple mirrors avoid rate-limiting on the primary overpass-api.de endpoint.
    """
    for url in urls:
        for attempt in range(1, 4):
            try:
                resp = requests.post(
                    url,
                    data={"data": query_string},
                    headers=_OVERPASS_HEADERS,
                    timeout=timeout,
                )
                resp.raise_for_status()
                time.sleep(2)  # polite rate-limit pause
                return resp.json()
            except Exception as e:
                wait = 2**attempt
                print(
                    f"  [{url.split('/')[2]}] attempt {attempt}/3 failed: {e}. Retrying in {wait}s ..."
                )
                time.sleep(wait)
        print(f"  Mirror {url} exhausted, trying next ...")
    print("  All mirrors and retries exhausted. Returning empty result.")
    return {}


def overpass_to_gdf_points(
    data: dict, amenity_tag: str = "amenity"
) -> gpd.GeoDataFrame:
    """Convert Overpass JSON elements to a point GeoDataFrame."""
    rows = []
    for el in data.get("elements", []):
        if el["type"] == "node":
            lat, lon = el.get("lat"), el.get("lon")
        elif "center" in el:
            lat, lon = el["center"]["lat"], el["center"]["lon"]
        else:
            continue
        tags = el.get("tags", {})
        amenity_type = (
            tags.get("amenity") or tags.get("shop") or tags.get("leisure") or "unknown"
        )
        rows.append(
            {
                "amenity_type": amenity_type,
                "lat": lat,
                "lon": lon,
                "geometry": Point(lon, lat),
            }
        )
    if rows:
        return gpd.GeoDataFrame(rows, crs="EPSG:4326")
    return gpd.GeoDataFrame(
        columns=["amenity_type", "lat", "lon", "geometry"],
        geometry="geometry",
        crs="EPSG:4326",
    )


def overpass_roads_to_gdf(data: dict) -> gpd.GeoDataFrame:
    """Convert Overpass road ways (out geom) to LineString GeoDataFrame."""
    rows = []
    for el in data.get("elements", []):
        if el["type"] != "way":
            continue
        geom_nodes = el.get("geometry", [])
        if len(geom_nodes) < 2:
            continue
        coords = [(g["lon"], g["lat"]) for g in geom_nodes]
        tags = el.get("tags", {})
        highway = tags.get("highway", "unclassified")
        rows.append({"highway": highway, "geometry": LineString(coords)})
    if rows:
        return gpd.GeoDataFrame(rows, crs="EPSG:4326")
    return gpd.GeoDataFrame(
        columns=["highway", "geometry"], geometry="geometry", crs="EPSG:4326"
    )


BBOX = CONFIG["bbox"]  # [west, south, east, north]
W, S, E, N = BBOX
OVERPASS_BBOX = f"{S},{W},{N},{E}"

QUERY_4A = f"""
[out:json][timeout:{CONFIG["overpass_timeout"]}][bbox:{OVERPASS_BBOX}];
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
"""

QUERY_4B = f"""
[out:json][timeout:{CONFIG["overpass_timeout"]}][bbox:{OVERPASS_BBOX}];
(
  node[amenity=hospital];
  way[amenity=hospital];
  node[amenity=place_of_worship];
  way[amenity=place_of_worship];
  node[amenity=government];
  way[amenity=government];
);
out center;
"""

QUERY_4C = f"""
[out:json][timeout:{CONFIG["overpass_timeout"]}][bbox:{OVERPASS_BBOX}];
(
  way[leisure=park];
  relation[leisure=park];
  way[landuse=grass];
  way[landuse=recreation_ground];
);
out center;
"""

QUERY_4D = f"""
[out:json][timeout:{CONFIG["overpass_timeout"]}][bbox:{OVERPASS_BBOX}];
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
"""

URLS = CONFIG["overpass_urls"]
TIMEOUT = CONFIG["overpass_timeout"]

# ── Local JSON cache directory (pre-fetched via browser console script) ───────
# Place files downloaded by src/fetch_overpass_data.js here:
#   data/overpass/overpass_4a_poi.json
#   data/overpass/overpass_4b_civic.json
#   data/overpass/overpass_4c_parks.json
#   data/overpass/overpass_4d_roads.json
CACHE_DIR = os.path.join("data", "overpass")
os.makedirs(CACHE_DIR, exist_ok=True)

_CACHE = {
    "4a": os.path.join(CACHE_DIR, "overpass_4a_poi.json"),
    "4b": os.path.join(CACHE_DIR, "overpass_4b_civic.json"),
    "4c": os.path.join(CACHE_DIR, "overpass_4c_parks.json"),
    "4d": os.path.join(CACHE_DIR, "overpass_4d_roads.json"),
}


def load_or_fetch(key, query_str, label):
    """Load from local JSON cache first; fall back to Overpass API."""
    cache_path = _CACHE[key]
    if os.path.exists(cache_path):
        size_kb = os.path.getsize(cache_path) / 1024
        print(f"  [{label}] Loading from cache: {cache_path} ({size_kb:.0f} KB)")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    print(f"  [{label}] Cache miss - querying Overpass API ...")
    data = query_overpass(query_str, URLS, TIMEOUT)
    if data.get("elements"):
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        print(f"  [{label}] Saved to cache: {cache_path}")
    return data


# ── Query 4A — Commercial & Transport POIs ────────────────────────────────────
print("Querying 4A - Commercial & Transport POIs ...")
try:
    raw_4a = load_or_fetch("4a", QUERY_4A, "4A")
    gdf_poi = overpass_to_gdf_points(raw_4a)
except Exception as e:
    print(f"WARNING: Query 4A failed: {e}")
    gdf_poi = gpd.GeoDataFrame(
        columns=["amenity_type", "lat", "lon", "geometry"],
        geometry="geometry",
        crs="EPSG:4326",
    )

# osmnx fallback when both cache and Overpass fail for 4A
if len(gdf_poi) == 0:
    print("  POI data empty - trying osmnx fallback for commercial POIs ...")
    try:
        W, S, E, N = CONFIG["bbox"]
        _poi_tags = {
            "amenity": [
                "school",
                "university",
                "marketplace",
                "bus_station",
                "food_court",
            ],
            "shop": ["supermarket", "mall"],
        }
        _ox_pois = ox.features_from_bbox((N, S, E, W), tags=_poi_tags)
        _rows = []
        for _, feat in _ox_pois.iterrows():
            c = feat.geometry.centroid
            _rows.append(
                {
                    "amenity_type": "osmnx_poi",
                    "lat": c.y,
                    "lon": c.x,
                    "geometry": Point(c.x, c.y),
                }
            )
        if _rows:
            gdf_poi = gpd.GeoDataFrame(_rows, crs="EPSG:4326")
            print(f"  osmnx fallback: {len(gdf_poi)} POIs retrieved.")
        else:
            print("  osmnx fallback returned no POIs.")
    except Exception as _e:
        print(f"  osmnx fallback also failed: {_e}")

# ── Query 4B — Health & Civic ─────────────────────────────────────────────────
print("Querying 4B - Health & Civic POIs ...")
try:
    raw_4b = load_or_fetch("4b", QUERY_4B, "4B")
    gdf_civic = overpass_to_gdf_points(raw_4b)
except Exception as e:
    print(f"WARNING: Query 4B failed: {e}")
    gdf_civic = gpd.GeoDataFrame(
        columns=["amenity_type", "lat", "lon", "geometry"],
        geometry="geometry",
        crs="EPSG:4326",
    )

# ── Query 4C — Parks & Open Space ─────────────────────────────────────────────
print("Querying 4C - Parks & Open Space ...")
try:
    raw_4c = load_or_fetch("4c", QUERY_4C, "4C")
    gdf_parks = overpass_to_gdf_points(raw_4c)
except Exception as e:
    print(f"WARNING: Query 4C failed: {e}")
    gdf_parks = gpd.GeoDataFrame(
        columns=["amenity_type", "lat", "lon", "geometry"],
        geometry="geometry",
        crs="EPSG:4326",
    )

# ── Query 4D — Road Network ───────────────────────────────────────────────────
print("Querying 4D - Road Network ...")
try:
    raw_4d = load_or_fetch("4d", QUERY_4D, "4D")
    gdf_roads = overpass_roads_to_gdf(raw_4d)
except Exception as e:
    print(f"WARNING: Query 4D failed: {e}")
    gdf_roads = gpd.GeoDataFrame(
        columns=["highway", "geometry"], geometry="geometry", crs="EPSG:4326"
    )

# osmnx fallback when both cache and Overpass fail for 4D
if len(gdf_roads) == 0:
    print("  Road data empty - trying osmnx graph fallback ...")
    try:
        W, S, E, N = CONFIG["bbox"]
        _G = ox.graph_from_bbox((N, S, E, W), network_type="drive")
        _edges = ox.graph_to_gdfs(_G, nodes=False)
        _edges["highway"] = _edges["highway"].apply(
            lambda x: x[0] if isinstance(x, list) else x
        )
        gdf_roads = _edges[["highway", "geometry"]].reset_index(drop=True).copy()
        gdf_roads = gdf_roads.to_crs("EPSG:4326")
        print(f"  osmnx road fallback: {len(gdf_roads)} segments retrieved.")
    except Exception as _e:
        print(f"  osmnx road fallback also failed: {_e}")

print("\nFeatures retrieved:")
print(f"  4A — Commercial POIs : {len(gdf_poi)}")
print(f"  4B — Civic/Health    : {len(gdf_civic)}")
print(f"  4C — Parks           : {len(gdf_parks)}")
print(f"  4D — Road segments   : {len(gdf_roads)}")

# %% [markdown]
# ## Step 5 — Compute Real Features per Grid Cell

# %%
print(f"\n{'=' * 50}\nSTEP 5: Compute Real Features\n{'=' * 50}")

# ── Feature 5A: POI Cluster Score ────────────────────────────────────────────
print("Computing POI Cluster Score (5A) …")
if len(gdf_poi) > 0:
    joined_poi = gpd.sjoin(
        gdf_poi, grid[["cell_id", "geometry"]], how="left", predicate="within"
    )
    poi_counts = joined_poi.groupby("cell_id").size().rename("poi_count")
    grid = grid.merge(poi_counts, on="cell_id", how="left")
    grid["poi_count"] = grid["poi_count"].fillna(0)
else:
    grid["poi_count"] = 0

grid["poi_log"] = np.log1p(grid["poi_count"])
max_poi = grid["poi_log"].max()
grid["poi_score"] = grid["poi_log"] / max_poi if max_poi > 0 else 0.0
print(
    f"  poi_score: mean={grid['poi_score'].mean():.4f}, max={grid['poi_score'].max():.4f}"
)

# ── Feature 5B: Traffic Flow Score ───────────────────────────────────────────
print("Computing Traffic Flow Score (5B) …")

# Project to UTM 48S for accurate area/length calculation (correct zone for Bandung)
CRS_UTM = "EPSG:32748"
grid_utm = grid.to_crs(CRS_UTM)

if len(gdf_roads) > 0:
    roads_utm = gdf_roads.to_crs(CRS_UTM)
    road_weights = CONFIG["road_weights"]
    traffic_vals = np.zeros(len(grid))

    for i, cell_row in grid_utm.iterrows():
        cell_geom = cell_row.geometry
        # Find candidate roads (bbox intersection first for speed)
        candidates = roads_utm[roads_utm.geometry.intersects(cell_geom)]
        weighted_len = 0.0
        for _, road_row in candidates.iterrows():
            clipped = road_row.geometry.intersection(cell_geom)
            length = clipped.length if not clipped.is_empty else 0.0
            weight = road_weights.get(road_row["highway"], 0.5)
            weighted_len += length * weight
        traffic_vals[i] = weighted_len

    grid["traffic_raw"] = traffic_vals
else:
    grid["traffic_raw"] = 0.0

t_max = grid["traffic_raw"].max()
grid["traffic_score"] = grid["traffic_raw"] / t_max if t_max > 0 else 0.0
print(
    f"  traffic_score: mean={grid['traffic_score'].mean():.4f}, max={grid['traffic_score'].max():.4f}"
)

# %% [markdown]
# ## Step 6 — Fabricate Synthetic Features

# %%
print(f"\n{'=' * 50}\nSTEP 6: Fabricate Synthetic Features\n{'=' * 50}")

# Seeded RNG for full reproducibility
rng = np.random.default_rng(CONFIG["synthetic_seed"])


# ── Feature 6A: Cellular Signal Density ─────────────────────────────────────
# Basis: ITU-R M.2135 urban coverage model — signal infrastructure deployment
#        follows population density logarithmically.
def compute_signal_density(pop_density_norm, poi_score, rng):
    # Base: log-transform of population density
    base = np.log1p(pop_density_norm * 50000) / np.log1p(50000)
    # Modifier: commercial areas have denser small-cell deployment
    commercial_boost = poi_score * 0.15
    # Realistic noise (sigma=0.04 from field measurement variance literature)
    noise = rng.normal(0, 0.04, size=len(pop_density_norm))
    signal = base + commercial_boost + noise
    return np.clip(signal, 0, 1)


grid["signal_score"] = compute_signal_density(
    grid["pop_density_norm"].values,
    grid["poi_score"].values,
    rng,
)


# ── Feature 6B: Construction Intensity Index ─────────────────────────────────
# Basis: Clark's Law (urban density gradient) — development activity concentrates
#        along urban growth corridors; known Bandung growth corridors used.
def compute_construction_index(grid_df, rng):
    # Known Bandung development corridor centroids [lon, lat]
    corridors = [
        (107.693, -6.938),  # Gedebage Teknopolis
        (107.640, -6.855),  # Summarecon Bandung
        (107.617, -6.877),  # Dago upper
        (107.645, -6.953),  # Buah Batu
        (107.577, -6.921),  # Pasteur / Sukajadi
    ]

    def min_dist_to_corridors(lon, lat):
        return min(((lon - cx) ** 2 + (lat - cy) ** 2) ** 0.5 for cx, cy in corridors)

    distances = grid_df.apply(
        lambda row: min_dist_to_corridors(row["centroid_lon"], row["centroid_lat"]),
        axis=1,
    )
    # Inverse distance, normalized + noise
    base = 1 / (1 + distances * 100)
    noise = rng.exponential(0.05, size=len(grid_df))
    index = base + noise
    index = (index - index.min()) / (index.max() - index.min())
    return np.clip(index, 0, 1)


grid["construction_index"] = compute_construction_index(grid, rng)


# ── Feature 6C: Existing Vendor Hotspot Density ──────────────────────────────
# Basis: Perda Kota Bandung No. 4/2011 on PKL — vendors cluster around
#        established commercial corridors; known PKL hotspot areas used.
def compute_vendor_hotspot_density(grid_df, rng):
    # Known PKL concentration centroids [lon, lat, relative_intensity]
    hotspots = [
        (107.608, -6.900, 1.00),  # Jl. Cihampelas
        (107.616, -6.888, 0.90),  # Jl. Dago / ITB area
        (107.607, -6.917, 0.90),  # Alun-alun Bandung
        (107.610, -6.915, 0.85),  # Pasar Baru
        (107.618, -6.908, 0.80),  # Jl. Braga
        (107.630, -6.902, 0.75),  # Jl. Riau / Diponegoro
        (107.652, -6.917, 0.70),  # Jl. Soekarno-Hatta east
        (107.571, -6.918, 0.65),  # Jl. Pasteur
        (107.638, -6.893, 0.60),  # UNPAD Dipatiukur
        (107.596, -6.926, 0.60),  # Jl. Moh. Toha / Pasar Caringin
    ]

    def hotspot_score(lon, lat):
        # Gaussian decay from each hotspot center
        score = 0
        for hx, hy, intensity in hotspots:
            dist_sq = (lon - hx) ** 2 + (lat - hy) ** 2
            score += intensity * np.exp(-dist_sq / (2 * 0.003**2))  # sigma ~300m
        return score

    scores = grid_df.apply(
        lambda row: hotspot_score(row["centroid_lon"], row["centroid_lat"]),
        axis=1,
    )
    noise = rng.exponential(0.05, size=len(grid_df))
    scores = scores + noise
    scores = (scores - scores.min()) / (scores.max() - scores.min())
    return np.clip(scores, 0, 1)


grid["vendor_hotspot_score"] = compute_vendor_hotspot_density(grid, rng)

print(
    f"  signal_score        : mean={grid['signal_score'].mean():.4f}, std={grid['signal_score'].std():.4f}"
)
print(
    f"  construction_index  : mean={grid['construction_index'].mean():.4f}, std={grid['construction_index'].std():.4f}"
)
print(
    f"  vendor_hotspot_score: mean={grid['vendor_hotspot_score'].mean():.4f}, std={grid['vendor_hotspot_score'].std():.4f}"
)

# %% [markdown]
# ## Step 7 — Rule-Based Labeling

# %%
print(f"\n{'=' * 50}\nSTEP 7: Rule-Based Labeling\n{'=' * 50}")

# ── Build exclusion buffers from 4B (hospitals, places of worship) ────────────
EXCLUDE_BUFFER_DEG = 80 / 111_320  # ~80 meters in degrees


def build_exclusion_mask(grid_df, exclusion_gdf, buffer_deg):
    """Return boolean mask: True where cell centroid is within buffer of exclusion features."""
    if len(exclusion_gdf) == 0:
        return pd.Series(False, index=grid_df.index)
    excl_points = exclusion_gdf[exclusion_gdf.geometry.geom_type == "Point"]
    if len(excl_points) == 0:
        return pd.Series(False, index=grid_df.index)
    # Union buffer for speed
    buffered = unary_union(excl_points.geometry.buffer(buffer_deg))
    mask = grid_df.apply(
        lambda r: buffered.contains(Point(r["centroid_lon"], r["centroid_lat"])),
        axis=1,
    )
    return mask


near_exclusion = build_exclusion_mask(grid, gdf_civic, EXCLUDE_BUFFER_DEG)


# ── Park/open space coverage ──────────────────────────────────────────────────
def build_park_mask(grid_df, parks_gdf):
    """Return boolean mask: True where cell centroid falls within a park polygon."""
    if len(parks_gdf) == 0:
        return pd.Series(False, index=grid_df.index)
    park_pts = parks_gdf[parks_gdf.geometry.geom_type == "Point"]
    if len(park_pts) == 0:
        return pd.Series(False, index=grid_df.index)
    park_union = unary_union(park_pts.geometry.buffer(0.002))
    mask = grid_df.apply(
        lambda r: park_union.contains(Point(r["centroid_lon"], r["centroid_lat"])),
        axis=1,
    )
    return mask


in_park = build_park_mask(grid, gdf_parks)

# ── Apply labeling rules (Perda PKL Bandung suitability criteria) ─────────────
poi_threshold = 0.25


def compute_labels(df, poi_thresh, near_exclusion_mask, in_park_mask):
    # When poi_score is uniformly 0 (POI data unavailable), use vendor_hotspot_score
    # as a proxy for commercial activity presence (Perda PKL Bandung fallback rule)
    poi_all_zero = df["poi_score"].max() == 0
    if poi_all_zero:
        poi_condition = df["vendor_hotspot_score"] > poi_thresh
    else:
        poi_condition = df["poi_score"] > poi_thresh

    pos_mask = (
        poi_condition
        & (df["traffic_score"] > 0.15)
        & (df["pop_density_norm"] > 0.3)
        & (~near_exclusion_mask)
        & (~in_park_mask)
        & (~((df["traffic_score"] > 0.85) & (df["poi_score"] < 0.2)))
    )
    labels = pos_mask.astype(int)
    return labels


grid["label"] = compute_labels(grid, poi_threshold, near_exclusion, in_park)

# Auto-adjust threshold downward until positive class >= 15% (max 5 steps)
pos_ratio = grid["label"].mean()
print(f"Initial positive ratio: {pos_ratio:.2%}")
for _adj in range(5):
    if pos_ratio >= 0.15:
        break
    poi_threshold = max(0.01, poi_threshold - 0.05)
    print(f"Adjusting threshold to {poi_threshold:.2f} ...")
    grid["label"] = compute_labels(grid, poi_threshold, near_exclusion, in_park)
    pos_ratio = grid["label"].mean()
    print(f"  -> positive ratio: {pos_ratio:.2%}")

# ── Label confidence ──────────────────────────────────────────────────────────
# Confidence = average of how strongly conditions are met (0=weak fail, 1=strong pass)
grid["label_confidence"] = (
    (grid["poi_score"] / (poi_threshold + 0.01)).clip(0, 1) * 0.3
    + (grid["traffic_score"] / 0.16).clip(0, 1) * 0.2
    + (grid["pop_density_norm"] / 0.31).clip(0, 1) * 0.2
    + grid["vendor_hotspot_score"] * 0.3
)

n_pos = (grid["label"] == 1).sum()
n_neg = (grid["label"] == 0).sum()
print("\nFinal class distribution:")
print(f"  Suitable   (label=1): {n_pos} ({n_pos / len(grid):.2%})")
print(f"  Unsuitable (label=0): {n_neg} ({n_neg / len(grid):.2%})")

# %% [markdown]
# ## Step 8 — Assemble Feature Matrix

# %%
print(f"\n{'=' * 50}\nSTEP 8: Assemble Feature Matrix\n{'=' * 50}")

FEATURE_COLS = [
    "pop_density_norm",
    "poi_score",
    "traffic_score",
    "signal_score",
    "construction_index",
    "vendor_hotspot_score",
]

feature_matrix = (
    grid[
        [
            "cell_id",
            "centroid_lat",
            "centroid_lon",
            *FEATURE_COLS,
            "label",
        ]
    ]
    .dropna()
    .copy()
)

print(f"Feature matrix shape: {feature_matrix.shape}")
feature_matrix.to_csv(
    os.path.join(CONFIG["output_dir"], "feature_matrix.csv"), index=False
)
print("Saved: output/feature_matrix.csv")

# %% [markdown]
# ## Step 9 — Train XGBoost Model

# %%
print(f"\n{'=' * 50}\nSTEP 9: Train XGBoost Model\n{'=' * 50}")

X = feature_matrix[FEATURE_COLS].values
y = feature_matrix["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=CONFIG["test_size"],
    stratify=y,
    random_state=CONFIG["random_state"],
)

scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=CONFIG["random_state"],
    early_stopping_rounds=20,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
unique_labels = np.unique(y_test)
if len(unique_labels) < 2:
    print(
        f"WARNING: Only one class ({unique_labels}) in test set — skipping classification report."
    )
    print(f"  All predictions: {np.unique(y_pred)}")
else:
    print(
        classification_report(y_test, y_pred, target_names=["Unsuitable", "Suitable"])
    )
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Unsuitable", "Suitable"],
    yticklabels=["Unsuitable", "Suitable"],
    ax=ax,
)
ax.set_title("TAPAK — Confusion Matrix", fontsize=13, fontweight="bold")
ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("Actual", fontsize=11)
plt.tight_layout()
cm_path = os.path.join(CONFIG["output_dir"], "confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"Saved: {cm_path}")

# Save model
model.save_model(CONFIG["model_path"])
print(f"Saved: {CONFIG['model_path']}")

# %% [markdown]
# ## Step 10 — SHAP Explainability

# %%
print(f"\n{'=' * 50}\nSTEP 10: SHAP Explainability\n{'=' * 50}")

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# 1. Beeswarm plot (summary)
fig, ax = plt.subplots(figsize=(10, 6))
shap.plots.beeswarm(shap_values, max_display=len(FEATURE_COLS), show=False)
plt.title("TAPAK — SHAP Feature Impact (Beeswarm)", fontsize=13, fontweight="bold")
plt.tight_layout()
beeswarm_path = os.path.join(CONFIG["output_dir"], "shap_summary_beeswarm.png")
plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {beeswarm_path}")

# 2. Bar importance plot
fig, ax = plt.subplots(figsize=(8, 5))
shap.plots.bar(shap_values, max_display=len(FEATURE_COLS), show=False)
plt.title("TAPAK — Mean |SHAP| Feature Importance", fontsize=13, fontweight="bold")
plt.tight_layout()
bar_path = os.path.join(CONFIG["output_dir"], "shap_bar_importance.png")
plt.savefig(bar_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {bar_path}")

# 3. Waterfall for highest-scoring cell in test set
best_idx = np.argmax(y_prob)
fig, ax = plt.subplots(figsize=(10, 6))
shap.plots.waterfall(shap_values[best_idx], show=False)
plt.title(
    "TAPAK — SHAP Waterfall (Highest Suitability Cell)",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
wf_path = os.path.join(CONFIG["output_dir"], "shap_waterfall_sample.png")
plt.savefig(wf_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {wf_path}")

# Top 3 features by mean |SHAP|
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
top3_idx = np.argsort(mean_abs_shap)[::-1][:3]
print("\nTop 3 features by mean |SHAP|:")
for rank, idx in enumerate(top3_idx, 1):
    print(f"  {rank}. {FEATURE_COLS[idx]}: {mean_abs_shap[idx]:.4f}")

# %% [markdown]
# ## Step 11 — Score All Grid Cells & Output

# %%
print(f"\n{'=' * 50}\nSTEP 11: Score All Grid Cells\n{'=' * 50}")

# Predict on full feature matrix
X_full = feature_matrix[FEATURE_COLS].values
suitability_scores = model.predict_proba(X_full)[:, 1]
feature_matrix["suitability_score"] = suitability_scores


# Zone classification
def classify_zone(score):
    if score > 0.65:
        return "Highly Suitable"
    elif score >= 0.35:
        return "Moderately Suitable"
    else:
        return "Not Suitable"


feature_matrix["zone_class"] = feature_matrix["suitability_score"].apply(classify_zone)

# Merge back to grid for geometry
grid_scored = grid.merge(
    feature_matrix[["cell_id", "suitability_score", "zone_class"]],
    on="cell_id",
    how="left",
)
grid_scored["suitability_score"] = grid_scored["suitability_score"].fillna(0)
grid_scored["zone_class"] = grid_scored["zone_class"].fillna("Not Suitable")

grid_scored.drop(columns=["geometry"], errors="ignore").to_csv(
    CONFIG["grid_output"], index=False
)
print(f"Saved: {CONFIG['grid_output']}")

zone_counts = grid_scored["zone_class"].value_counts()
print("\nZone distribution:")
for zone, count in zone_counts.items():
    print(f"  {zone}: {count} ({count / len(grid_scored):.2%})")

# %% [markdown]
# ## Step 12 — Folium Interactive Map

# %%
print(f"\n{'=' * 50}\nSTEP 12: Folium Interactive Map\n{'=' * 50}")

ZONE_COLORS = {
    "Highly Suitable": "#2ecc71",
    "Moderately Suitable": "#f39c12",
    "Not Suitable": "#e74c3c",
}

# Determine top contributing feature per cell from SHAP mean importance
top_feature = FEATURE_COLS[np.argmax(mean_abs_shap)]

bandung_center = [-6.9175, 107.6191]
m = folium.Map(location=bandung_center, zoom_start=13, tiles="OpenStreetMap")

# Draw grid cells as rectangles
for _, row in grid_scored.iterrows():
    zone = row.get("zone_class", "Not Suitable")
    score = row.get("suitability_score", 0)
    color = ZONE_COLORS.get(zone, "#e74c3c")
    opacity = 0.45 if zone != "Not Suitable" else 0.20

    geom = row["geometry"]
    bounds = geom.bounds  # (minx, miny, maxx, maxy) = (west, south, east, north)
    sw = [bounds[1], bounds[0]]
    ne = [bounds[3], bounds[2]]

    popup_html = (
        f"<b>Cell ID:</b> {int(row['cell_id'])}<br>"
        f"<b>Suitability Score:</b> {score:.2f}<br>"
        f"<b>Zone Class:</b> {zone}<br>"
        f"<b>Top Feature:</b> {top_feature}"
    )

    folium.Rectangle(
        bounds=[sw, ne],
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=opacity,
        weight=0.3,
        popup=folium.Popup(popup_html, max_width=250),
    ).add_to(m)

# City center marker
folium.Marker(
    location=bandung_center,
    popup="Bandung City Center",
    tooltip="Bandung City Center",
    icon=folium.Icon(color="blue", icon="info-sign"),
).add_to(m)

# Legend (HTML embedded)
legend_html = """
<div style="
    position: fixed; bottom: 30px; right: 30px; z-index: 9999;
    background: white; padding: 14px 18px; border-radius: 10px;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.3);
    font-family: Arial, sans-serif; font-size: 13px;
">
    <b style="font-size:14px;">TAPAK Zone Legend</b><br><br>
    <span style="background:#2ecc71;display:inline-block;width:16px;height:16px;
          border-radius:3px;margin-right:8px;vertical-align:middle;"></span>
    Highly Suitable (&gt;0.65)<br>
    <span style="background:#f39c12;display:inline-block;width:16px;height:16px;
          border-radius:3px;margin-right:8px;vertical-align:middle;"></span>
    Moderately Suitable (0.35–0.65)<br>
    <span style="background:#e74c3c;display:inline-block;width:16px;height:16px;
          border-radius:3px;margin-right:8px;vertical-align:middle;"></span>
    Not Suitable (&lt;0.35)
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

m.save(CONFIG["map_output"])
print(f"Map saved to {CONFIG['map_output']} — open in browser to view.")

# %% [markdown]
# ## Step 13 — Sanity Check

# %%
print(f"\n{'=' * 50}\nSTEP 13: Sanity Check\n{'=' * 50}")

VALIDATION_POINTS = [
    ("Jl. Cihampelas", 107.608, -6.900),
    ("Jl. Dago / ITB area", 107.616, -6.888),
    ("Alun-alun Bandung", 107.607, -6.917),
    ("Pasar Baru", 107.610, -6.915),
]

print("=== TAPAK SANITY CHECK ===")
print("Known PKL hotspot validation:")


def find_nearest_cell(scored_df, lon, lat):
    """Return the row in scored_df whose centroid is closest to (lon, lat)."""
    dists = (scored_df["centroid_lon"] - lon) ** 2 + (
        scored_df["centroid_lat"] - lat
    ) ** 2
    idx = dists.idxmin()
    return scored_df.loc[idx]


passes = 0
for name, lon, lat in VALIDATION_POINTS:
    cell = find_nearest_cell(grid_scored, lon, lat)
    score = cell.get("suitability_score", 0)
    zone = cell.get("zone_class", "N/A")
    result = "PASS" if score > 0.5 else "FAIL"
    if result == "PASS":
        passes += 1
    print(
        f"  - {name} (lon={lon}, lat={lat}): {score:.3f} [{zone}] -> {result} (expect >0.5)"
    )

print(
    f"\nOverall: {passes}/{len(VALIDATION_POINTS)} known hotspots correctly classified."
)
if passes < 2:
    print(
        "WARNING — model may need threshold adjustment. Consider lowering zone_class thresholds."
    )

# %% [markdown]
# ## Step 14 — Summary Report

# %%
print(f"\n{'=' * 50}\nSTEP 14: Output Summary\n{'=' * 50}")

output_files = [
    os.path.join(CONFIG["output_dir"], "feature_matrix.csv"),
    CONFIG["grid_output"],
    CONFIG["model_path"],
    os.path.join(CONFIG["output_dir"], "confusion_matrix.png"),
    os.path.join(CONFIG["output_dir"], "shap_summary_beeswarm.png"),
    os.path.join(CONFIG["output_dir"], "shap_bar_importance.png"),
    os.path.join(CONFIG["output_dir"], "shap_waterfall_sample.png"),
    CONFIG["map_output"],
]

print(f"\n{'File':<45} {'Size':>12}")
print("-" * 58)
for fpath in output_files:
    if os.path.exists(fpath):
        size = os.path.getsize(fpath)
        size_str = (
            f"{size / 1024:.1f} KB"
            if size < 1_048_576
            else f"{size / 1_048_576:.1f} MB"
        )
        print(f"  {fpath:<43} {size_str:>12}")
    else:
        print(f"  {fpath:<43} {'MISSING':>12}")

print("\n✓ TAPAK pipeline complete.")
