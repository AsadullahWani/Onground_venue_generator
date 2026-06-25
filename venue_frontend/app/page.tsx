
"use client";

import { Righteous } from "next/font/google";
import { useMemo, useState } from "react";

type Venue = {
  place_id: string;       // API returns place_id, not id
  name: string;
  sport_type: string;     // raw Geoapify category e.g. "sport.pitch"
  ai_sport_type: string;  // AI-inferred e.g. "Cricket" or "Football, Cricket"
  ai_confidence: string;  // "high" | "medium" | "low" | "skipped"
  formatted_address: string;
  website: string;        // API field is `website`, not website_uri
  ai_summary: string;     // always returned (empty string if not generated)
};

const PAGE_SIZE = 9;

export default function VenueFinder() {
  const [city, setCity] = useState("");
  const [sport, setSport] = useState("");
  const [loading, setLoading] = useState(false);

  const [venues, setVenues] = useState<Venue[]>([]);
  const [selectedSport, setSelectedSport] = useState("All Sports");
  const [page, setPage] = useState(1);
  const [maxResults, setMaxResults] = useState(100);

  async function searchVenues() {
    if (!city) return;

    setLoading(true);

    try {
      const response = await fetch(
        "http://localhost:8000/venues/search",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            city: city,
            // API expects list[str] — wrap the single input value in an array.
            // If the user left the field blank, omit it so the API uses all sports.
            ...(sport ? { sports: [sport] } : {}),
            include_ai_summary: true,
            ai_categorize: true,
            // If a specific sport was typed, also filter AI results to match it.
            ...(sport ? { ai_filter_sports: [sport] } : {}),
            max_results: maxResults,
          }),
        }
      );

      const data = await response.json();
// console.log("API response:", data);
// console.log("Array?", Array.isArray(data));
      setVenues(data.venues);
     
      setPage(1);
    } catch (error) {
      console.error(error);
    }

    setLoading(false);
  }
console.log("venues:", venues);
console.log("isArray:", Array.isArray(venues));
  const sports = useMemo(() => {
    // Use ai_sport_type for filtering — it's human-readable ("Cricket", "Football")
    // vs raw Geoapify category ("sport.pitch"). Split on "," for multi-sport venues.
    const unique = [
      ...new Set(
        venues.flatMap(v =>
          v.ai_sport_type
            ? v.ai_sport_type.split(",").map(s => s.trim())
            : [v.sport_type]
        )
      ),
    ].filter(Boolean).sort();
    return ["All Sports", ...unique];
  }, [venues]);

  const filteredVenues = useMemo(() => {
    if (selectedSport === "All Sports") return venues;
    return venues.filter(v => {
      // Match against ai_sport_type (may be comma-separated for multi-sport venues)
      const vSports = v.ai_sport_type
        ? v.ai_sport_type.split(",").map(s => s.trim())
        : [v.sport_type];
      return vSports.includes(selectedSport);
    });
  }, [venues, selectedSport]);

  const totalPages = Math.ceil(
    filteredVenues.length / PAGE_SIZE
  );

  const currentVenues = filteredVenues.slice(
    (page - 1) * PAGE_SIZE,
    page * PAGE_SIZE
  );

  function exportCSV() {
    if (!filteredVenues.length) return;
    const rows = filteredVenues.map(v => ({
      name: v.name,
      sport: v.ai_sport_type || v.sport_type,
      address: v.formatted_address,
      ai_confidence: v.ai_confidence,
      website: v.website,
    }));
    const csv = [
      Object.keys(rows[0]).join(","),
      ...rows.map(r =>
        Object.values(r).map(val => `"${String(val ?? "").replace(/"/g, '""')}"`).join(",")
      ),
    ].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${city}-venues.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <main className="min-h-screen bg-[#0B1120] text-white">

      {/* HERO */}

      <section className="max-w-7xl mx-auto px-6 pt-24 pb-16 text-center">

        <div className="uppercase tracking-[0.25em] text-emerald-400 text-xs font-semibold">
          Powered by AI
        </div>

        <h1 className="mt-6 text-6xl md:text-7xl font-black">
          Find Amazing
          <span className="text-emerald-400">
            {" "}Sports Venues
          </span>
        </h1>

        <p className="max-w-2xl mx-auto mt-6 text-slate-400">
          Discover stadiums, courts, pitches and
          sports facilities with AI powered
          recommendations.
        </p>

        {/* SEARCH */}

        <div className="mt-12 bg-slate-900 border border-slate-800 rounded-3xl p-5 grid lg:grid-cols-3 gap-5">

          <input
            value={city}
            onChange={(e) =>
              setCity(e.target.value)
            }
            placeholder="City"
            className="bg-slate-800 rounded-xl p-4 outline-none"
          />

          {/* <input
            value={sport}
            onChange={(e) =>
              setSport(e.target.value)
            }
            placeholder="Sport"
            className="bg-slate-800 rounded-xl p-4 outline-none"
          /> */}

          <button
            onClick={searchVenues}
            className="bg-emerald-500 hover:bg-emerald-400 text-black rounded-xl font-bold"
          >
            Search Venues
          </button>
            <select
                value={maxResults}
                onChange={(e) => setMaxResults(Number(e.target.value))}
                className="bg-slate-800 rounded-xl p-4 outline-none"
              >
                <option value={25}>25</option>
                <option value={50}>50</option>
                <option value={100}>100</option>
                <option value={200}>200</option>
                <option value={500}>500</option>
              </select>
        
        </div>
      </section>
      {/* CONTENT */}

      <section className="max-w-7xl mx-auto px-6 pb-24">

        <div className="grid lg:grid-cols-[280px_1fr] gap-8">

          {/* SIDEBAR */}

          <aside className="bg-slate-900 border border-slate-800 rounded-3xl p-6 h-fit">

            <h3 className="font-bold text-xl">
              Filters
            </h3>

            <div className="mt-6">
              <label className="text-slate-400 text-sm">
                Sport
              </label>

              <select
                value={selectedSport}
                onChange={(e) =>
                  setSelectedSport(e.target.value)
                }
                className="mt-2 w-full bg-slate-800 rounded-xl p-3"
              >
                {sports.map(s => (
                  <option key={s}>{s}</option>
                ))}
              </select>
            </div>

            <button
              onClick={exportCSV}
              className="w-full mt-6 border border-slate-700 rounded-xl p-3"
            >
              Export CSV
            </button>
          </aside>

          {/* RESULTS */}

          <div>

            <div className="flex justify-between items-center mb-8">

              <h2 className="text-3xl font-bold">
                {filteredVenues.length} Venues Found
              </h2>

              <span className="text-slate-400">
                Page {page} of {Math.max(1,totalPages)}
              </span>

            </div>

            {loading && (
              <div className="text-center py-20">
                Loading venues...
              </div>
            )}

            <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-6">

              {currentVenues.map((venue) => (
                <div
                  key={venue.place_id}
                  className="bg-slate-900 border border-slate-800 rounded-3xl overflow-hidden"
                >

                  <div className="h-52 bg-gradient-to-br from-slate-700 to-slate-900" />

                  <div className="p-5">

                    <div className="flex items-center gap-2 flex-wrap">
                      {/* Show AI-inferred sport(s) — may be comma-separated */}
                      {(venue.ai_sport_type || venue.sport_type)
                        .split(",")
                        .map(s => s.trim())
                        .filter(Boolean)
                        .map(s => (
                          <span
                            key={s}
                            className="inline-flex bg-emerald-500/10 text-emerald-400 rounded-full px-3 py-1 text-xs"
                          >
                            {s}
                          </span>
                        ))}
                      {venue.ai_confidence && venue.ai_confidence !== "skipped" && (
                        <span className="text-xs text-slate-500">
                          {venue.ai_confidence} confidence
                        </span>
                      )}
                    </div>

                    <h3 className="mt-4 text-xl font-bold">
                      {venue.name}
                    </h3>

                    <div className="mt-2 text-slate-400 text-sm">
                      📍 {venue.formatted_address || "No address available"}
                    </div>

                    {venue.ai_summary && (
                      <div className="mt-4 bg-blue-500/10 border border-blue-500/20 rounded-xl p-3 text-sm text-slate-300">
                        ✨ {venue.ai_summary}
                      </div>
                    )}

                    <div className="flex gap-2 mt-5">
                      {venue.website && (
                        <a
                          href={venue.website}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="flex-1 bg-emerald-500 text-black text-center rounded-lg py-2 text-sm font-semibold"
                        >
                          Website
                        </a>
                      )}
                    </div>

                  </div>

                </div>
              ))}

            </div>

            {/* PAGINATION */}

            {totalPages > 1 && (
              <div className="flex justify-center gap-2 mt-12">

                <button
                  disabled={page === 1}
                  onClick={() =>
                    setPage(page - 1)
                  }
                  className="px-4 py-2 border border-slate-700 rounded-lg"
                >
                  Prev
                </button>

                <button
                  disabled={page === totalPages}
                  onClick={() =>
                    setPage(page + 1)
                  }
                  className="px-4 py-2 border border-slate-700 rounded-lg"
                >
                  Next
                </button>

              </div>
            )}

          </div>

        </div>

      </section>

    </main>
  );
}

