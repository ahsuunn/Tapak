// ============================================================
// TAPAK — Overpass API Fetcher
// Paste this entire script into your browser DevTools console
// (F12 → Console tab) while on ANY page (e.g. https://overpass-api.de)
//
// It will run all 4 queries and automatically download the results
// as JSON files into your Downloads folder:
//   • overpass_4a_poi.json
//   • overpass_4b_civic.json
//   • overpass_4c_parks.json
//   • overpass_4d_roads.json
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

// ── Run all queries sequentially ────────────────────────────
(async () => {
  try {
    // const data4a = await runQuery("4A - Commercial POIs", QUERY_4A);
    // downloadJSON("overpass_4a_poi.json", data4a);

    // await new Promise(r => setTimeout(r, 2000)); // polite pause

    // const data4b = await runQuery("4B - Civic/Health", QUERY_4B);
    // downloadJSON("overpass_4b_civic.json", data4b);

    // await new Promise(r => setTimeout(r, 2000));

    const data4c = await runQuery("4C - Parks", QUERY_4C);
    downloadJSON("overpass_4c_parks.json", data4c);

    await new Promise(r => setTimeout(r, 2000));

    const data4d = await runQuery("4D - Roads", QUERY_4D);
    downloadJSON("overpass_4d_roads.json", data4d);

    console.log("[TAPAK] All 4 queries complete. Move the 4 JSON files into:");
    console.log("        c:\\Users\\thema\\Informatics\\Language\\Python\\Tapak\\data\\overpass\\");
  } catch (err) {
    console.error("[TAPAK] Error:", err);
  }
})();
