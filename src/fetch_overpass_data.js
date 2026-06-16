// ============================================================
// TAPAK — Overpass API Fetcher (v2 — expanded indicators)
// Paste this entire script into your browser DevTools console
// (F12 → Console tab) while on ANY page (e.g. https://overpass-api.de)
//
// It will run queries and automatically download the results
// as JSON files into your Downloads folder.
//
// EXISTING (already cached — uncomment if re-fetch needed):
//   • overpass_4a_poi.json        — Schools, universities, markets, bus stations, malls
//   • overpass_4b_civic.json      — Hospitals, worship, government
//   • overpass_4c_parks.json      — Parks, grass, recreation
//   • overpass_4d_roads.json      — Full road network
//
// NEW queries for expanded indicator system:
//   • overpass_4e_catering.json   — Restaurants, cafes, fast food (CaFD)
//   • overpass_4f_shops.json      — All shops (CoFD)
//   • overpass_4g_offices.json    — Office buildings (BFD)
//   • overpass_4h_buildings.json  — Building footprints (FA)
//   • overpass_4i_recreation.json — Recreational facilities (RFD)
// ============================================================

const OVERPASS_URL = "https://overpass-api.de/api/interpreter";
const BBOX = "-7.0,107.55,-6.85,107.70"; // south,west,north,east for Overpass

// ── Helper: POST one query ──────────────────────────────────
async function runQuery(name, queryStr) {
  console.log(`[TAPAK] Fetching ${name}...`);
  const resp = await fetch(OVERPASS_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: "data=" + encodeURIComponent(queryStr),
  });
  if (!resp.ok) throw new Error(`HTTP ${resp.status} ${resp.statusText}`);
  const data = await resp.json();
  console.log(`[TAPAK] ${name}: ${data.elements.length} elements`);
  return data;
}

// ── Helper: trigger browser download ───────────────────────
function downloadJSON(filename, data) {
  const blob = new Blob([JSON.stringify(data)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
  console.log(`[TAPAK] Downloaded: ${filename}`);
}

// ════════════════════════════════════════════════════════════
// EXISTING QUERIES (already cached — uncomment if needed)
// ════════════════════════════════════════════════════════════

// ── Query 4A — Commercial & Transport POIs ──────────────────
const QUERY_4A = `
[out:json][timeout:60][bbox:${BBOX}];
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
`;

// ── Query 4B — Health & Civic ───────────────────────────────
const QUERY_4B = `
[out:json][timeout:60][bbox:${BBOX}];
(
  node[amenity=hospital];
  way[amenity=hospital];
  node[amenity=place_of_worship];
  way[amenity=place_of_worship];
  node[amenity=government];
  way[amenity=government];
);
out center;
`;

// ── Query 4C — Parks & Open Space ──────────────────────────
const QUERY_4C = `
[out:json][timeout:60][bbox:${BBOX}];
(
  way[leisure=park];
  relation[leisure=park];
  way[landuse=grass];
  way[landuse=recreation_ground];
);
out center;
`;

// ── Query 4D — Road Network ─────────────────────────────────
const QUERY_4D = `
[out:json][timeout:90][bbox:${BBOX}];
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
`;

// ════════════════════════════════════════════════════════════
// NEW QUERIES — Expanded Indicator System
// (Matching Zhou et al. 2025 Table 2 variable set)
// ════════════════════════════════════════════════════════════

// ── Query 4E — Catering Facilities (CaFD) ───────────────────
// Restaurants, cafes, fast food — catering facility density
// Paper finding: CaFD is NEGATIVELY correlated with vendor clustering
// (formal dining crowds out informal food vendors)
const QUERY_4E = `
[out:json][timeout:60][bbox:${BBOX}];
(
  node[amenity=restaurant];
  way[amenity=restaurant];
  node[amenity=cafe];
  way[amenity=cafe];
  node[amenity=fast_food];
  way[amenity=fast_food];
  node[amenity=bar];
  way[amenity=bar];
  node[amenity=pub];
  way[amenity=pub];
);
out center;
`;

// ── Query 4F — All Shops / Commercial Facilities (CoFD) ─────
// Broader commercial facility density — all retail types
const QUERY_4F = `
[out:json][timeout:90][bbox:${BBOX}];
(
  node[shop];
  way[shop];
);
out center;
`;

// ── Query 4G — Offices / Business Facilities (BFD) ──────────
// Office buildings, coworking spaces
const QUERY_4G = `
[out:json][timeout:60][bbox:${BBOX}];
(
  node[office];
  way[office];
  node[amenity=coworking_space];
  way[amenity=coworking_space];
);
out center;
`;

// ── Query 4H — Building Footprints (FA) ─────────────────────
// Building outlines with floor/level data for Floor Area computation
// Note: this query may be large; timeout set to 120s
const QUERY_4H = `
[out:json][timeout:120][bbox:${BBOX}];
(
  way[building];
  relation[building];
);
out center body;
`;

// ── Query 4I — Recreational Facilities (RFD) ────────────────
// Sports, fitness, entertainment venues — top importance factor in Wuhan paper
const QUERY_4I = `
[out:json][timeout:60][bbox:${BBOX}];
(
  node[leisure=sports_centre];
  way[leisure=sports_centre];
  node[leisure=fitness_centre];
  way[leisure=fitness_centre];
  node[leisure=stadium];
  way[leisure=stadium];
  node[leisure=swimming_pool];
  way[leisure=swimming_pool];
  node[leisure=playground];
  way[leisure=playground];
  node[amenity=cinema];
  way[amenity=cinema];
  node[amenity=theatre];
  way[amenity=theatre];
  node[amenity=community_centre];
  way[amenity=community_centre];
  node[tourism=museum];
  way[tourism=museum];
  node[tourism=attraction];
  way[tourism=attraction];
);
out center;
`;

// ════════════════════════════════════════════════════════════
// RUN QUERIES — Uncomment the ones you need
// ════════════════════════════════════════════════════════════
(async () => {
  try {
    // ── Existing queries (already cached — uncomment only if re-fetch needed)
    // const data4a = await runQuery("4A - Commercial POIs", QUERY_4A);
    // downloadJSON("overpass_4a_poi.json", data4a);
    // await new Promise(r => setTimeout(r, 2000));

    // const data4b = await runQuery("4B - Civic/Health", QUERY_4B);
    // downloadJSON("overpass_4b_civic.json", data4b);
    // await new Promise(r => setTimeout(r, 2000));

    // const data4c = await runQuery("4C - Parks", QUERY_4C);
    // downloadJSON("overpass_4c_parks.json", data4c);
    // await new Promise(r => setTimeout(r, 2000));

    // const data4d = await runQuery("4D - Roads", QUERY_4D);
    // downloadJSON("overpass_4d_roads.json", data4d);
    // await new Promise(r => setTimeout(r, 2000));

    // ── NEW queries — run these
    // const data4e = await runQuery("4E - Catering Facilities", QUERY_4E);
    // downloadJSON("overpass_4e_catering.json", data4e);
    // await new Promise(r => setTimeout(r, 3000)); // longer pause for rate limiting

    // const data4f = await runQuery("4F - Shops/Commercial", QUERY_4F);
    // downloadJSON("overpass_4f_shops.json", data4f);
    // await new Promise(r => setTimeout(r, 3000));

    // const data4g = await runQuery("4G - Offices", QUERY_4G);
    // downloadJSON("overpass_4g_offices.json", data4g);
    // await new Promise(r => setTimeout(r, 3000));

    // const data4h = await runQuery("4H - Buildings", QUERY_4H);
    // downloadJSON("overpass_4h_buildings.json", data4h);
    // await new Promise(r => setTimeout(r, 3000));

    const data4i = await runQuery("4I - Recreation", QUERY_4I);
    downloadJSON("overpass_4i_recreation.json", data4i);

    console.log("[TAPAK] All NEW queries complete. Move the JSON files into:");
    console.log("        c:\\Users\\thema\\Informatics\\Language\\Python\\Tapak\\data\\overpass\\");
    console.log("");
    console.log("[TAPAK] Expected files:");
    console.log("  overpass_4e_catering.json   → CaFD (Catering Facility Density)");
    console.log("  overpass_4f_shops.json      → CoFD (Commercial Facility Density)");
    console.log("  overpass_4g_offices.json    → BFD  (Business Office Facility Density)");
    console.log("  overpass_4h_buildings.json  → FA   (Floor Area proxy)");
    console.log("  overpass_4i_recreation.json → RFD  (Recreational Facility Density)");
  } catch (err) {
    console.error("[TAPAK] Error:", err);
    console.error("[TAPAK] If rate-limited, wait 30s and re-run the script.");
  }
})();
