"""
SD6105 Data Visualisation - Tutorial 8 Solutions
Topic: Geospatial Data Visualization

8.1  Choosing a geospatial visualization for point / region data
8.2  Choosing a map projection family (azimuthal / cylindrical / conic)

Mostly conceptual. The script draws one supporting figure: how the three
projection families distort equal-size circles ("Tissot indicatrices") at
different latitudes - computed with pure numpy, no GIS library needed.

Run from inside this folder:  python tutorial8_solutions.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("charts", exist_ok=True)

# =============================================================================
# 8.1  WHICH GEOSPATIAL VISUALIZATION?
# =============================================================================
"""
(a) Annual sales of each McDonald's outlet in Singapore, 2026
    Data type: POINT data (each outlet has lat/long) + one quantitative weight.
    -> PROPORTIONAL SYMBOL MAP (symbol / bubble map): one circle per outlet at
       its location, circle AREA proportional to annual sales (scale the radius
       by sqrt(sales) so perceived area is truthful), semi-transparent fill to
       handle overlap in dense areas (Orchard, CBD); optional colour for a 2nd
       variable such as YoY growth.
    Justification: keeps the exact location of each store (choropleth would
    aggregate it away), and size encodes magnitude so high-revenue outlets
    stand out. Singapore is small, so overplotting is manageable; if it is
    not, add zoom / tooltips, or switch to a hexbin/density map for a
    "where is revenue concentrated" overview.

(b) Average temperature of each EU country
    Data type: REGION data, one value per country, continuous & intensive
    (an average, not a count) -> CHOROPLETH MAP with a SEQUENTIAL colour
    scheme (e.g. ColorBrewer YlOrRd), 5-7 classed bins, clear legend in degC.
    Justification: the value naturally belongs to the whole area of a country
    and temperature is an intensive variable, so shading area by value is
    appropriate (no "big countries look more important" problem as with
    totals). Neighbouring-country patterns (north-south gradient) become
    obvious. (If the deviation from the EU average is the message, use a
    DIVERGING scheme centred on the mean.)
    Note: an isopleth / contour map would be better if we had station-level
    temperature measurements rather than country averages.

(c) Ageing population (age > 65) of each U.S. state
    Data type: REGION data. Two options depending on the measure:
    * Proportion of residents aged 65+ (normalised, intensive)
      -> CHOROPLETH map, sequential palette. Shows where the population is
         oldest (e.g. Maine, Florida) independent of state size.
    * Absolute NUMBER of people aged 65+ (a count)
      -> a choropleth would mislead (large but sparsely populated states such as
         Montana dominate visually), so use a CARTOGRAM (e.g. Dorling or tile
         grid cartogram, area/size proportional to the count) or a proportional
         symbol map with circles at state centroids.
    Recommended: choropleth of the 65+ share + tooltip / labels with counts,
    or a tile-grid map so that small north-eastern states remain visible.

# =============================================================================
# 8.2  WHICH MAP PROJECTION?
# =============================================================================
Rule of thumb (latitude rule, lecture):
    low latitudes / tropics      -> CYLINDRICAL (tangent to the equator)
    middle latitudes             -> CONIC (cone touches at standard parallels)
    polar regions                -> AZIMUTHAL (plane touches at the pole)
Distortion is smallest where the developable surface touches the globe.

| Location      | Latitude span     | Projection family | Example              | Why                                                       |
|---------------|-------------------|-------------------|----------------------|-----------------------------------------------------------|
| (a) Malaysia  | ~1 N - 7 N        | Cylindrical       | Mercator / equal-area| Straddles the equator; a cylinder tangent at the equator  |
|               |                   |                   | cylindrical          | has almost no distortion there; E-W extent (Peninsular to |
|               |                   |                   |                      | Borneo) fits a rectangular map.                           |
| (b) China     | ~18 N - 54 N      | Conic             | Albers equal-area    | Mid-latitude country with large E-W extent; two standard  |
|               |                   |                   | conic (std. 25/47 N) | parallels keep area/shape distortion low across the whole |
|               |                   |                   |                      | country (Mercator would inflate the north-east).          |
| (c) India     | ~8 N - 37 N       | Cylindrical       | Mercator (or a conic | Mostly tropical / low latitude -> cylindrical per latitude|
|               |                   | (conic also OK)   | e.g. Lambert for     | rule. Because it extends to ~37 N with large N-S extent,  |
|               |                   |                   | national mapping)    | a conic (Lambert conformal) is also commonly used.        |
| (d) USA       | ~25 N - 49 N      | Conic             | Albers equal-area    | Classic mid-latitude, wide E-W country; Albers (std.      |
|               | (contiguous)      |                   | (std. 29.5/45.5 N)   | 29.5/45.5 N) is the standard for US thematic maps.        |
| (e) North Pole| 90 N              | Azimuthal         | Azimuthal equidistant| A plane tangent at the pole has zero distortion at the    |
|               |                   |                   | / polar stereographic| pole and distortion grows radially; cylinders/cones       |
|               |                   |                   |                      | cannot even show the pole as a point (Mercator -> inf).   |
"""

# =============================================================================
# FIGURE: distortion of equal circles under the three projection families
# =============================================================================
R = 1.0


def mercator(lon, lat):
    lam, phi = np.radians(lon), np.radians(np.clip(lat, -85, 85))
    return R * lam, R * np.log(np.tan(np.pi / 4 + phi / 2))


def albers(lon, lat, lat1=29.5, lat2=45.5, lat0=37.5, lon0=-96):
    p1, p2, p0 = np.radians([lat1, lat2, lat0])
    n = (np.sin(p1) + np.sin(p2)) / 2
    C = np.cos(p1) ** 2 + 2 * n * np.sin(p1)
    rho0 = R * np.sqrt(C - 2 * n * np.sin(p0)) / n
    phi, lam = np.radians(lat), np.radians(lon - lon0)
    rho = R * np.sqrt(C - 2 * n * np.sin(phi)) / n
    theta = n * lam
    return rho * np.sin(theta), rho0 - rho * np.cos(theta)


def azimuthal_equidistant_north(lon, lat):
    rho = R * np.radians(90 - lat)
    lam = np.radians(lon)
    return rho * np.sin(lam), -rho * np.cos(lam)


def small_circle(lon0, lat0, radius_deg=4.5, n=60):
    """Points on the sphere at a fixed angular distance from (lon0, lat0)."""
    phi0, lam0, d = np.radians(lat0), np.radians(lon0), np.radians(radius_deg)
    az = np.linspace(0, 2 * np.pi, n)
    phi = np.arcsin(np.sin(phi0) * np.cos(d) + np.cos(phi0) * np.sin(d) * np.cos(az))
    lam = lam0 + np.arctan2(np.sin(az) * np.sin(d) * np.cos(phi0),
                            np.cos(d) - np.sin(phi0) * np.sin(phi))
    return np.degrees(lam), np.degrees(phi)


def draw(ax, proj, lons, lats, centres, title, subtitle, touch=None):
    for la in lats:
        lo = np.linspace(lons[0], lons[-1], 200)
        ax.plot(*proj(lo, np.full_like(lo, la)), color="#bbbbbb", lw=0.6)
    for lo in lons:
        la = np.linspace(lats[0], lats[-1], 200)
        ax.plot(*proj(np.full_like(la, lo), la), color="#bbbbbb", lw=0.6)
    if touch is not None:
        for la in touch:
            lo = np.linspace(lons[0], lons[-1], 200)
            ax.plot(*proj(lo, np.full_like(lo, la)), color="#2166ac", lw=1.6)
    for c in centres:
        x, y = proj(*small_circle(*c))
        ax.fill(x, y, color="#d6604d", alpha=0.75)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, weight="bold", fontsize=11)
    ax.text(0.5, -0.04, subtitle, transform=ax.transAxes, ha="center", va="top", fontsize=9)


fig, axes = plt.subplots(1, 3, figsize=(15, 5.8))
draw(axes[0], mercator, np.arange(-180, 181, 30), np.arange(-80, 81, 20),
     [(lo, la) for lo in (-120, -30, 60, 150) for la in (0, 30, 55, 72)],
     "Cylindrical (Mercator)",
     "Touches at the equator (blue): circles stay true near\n"
     "the equator, inflate strongly towards the poles\n-> Malaysia, India (tropics)",
     touch=[0])
draw(axes[1], albers, np.arange(-130, -59, 10), np.arange(20, 61, 10),
     [(lo, la) for lo in (-120, -95, -70) for la in (25, 37.5, 50)],
     "Conic (Albers equal-area)",
     "Touches at two standard parallels 29.5N/45.5N (blue):\n"
     "low distortion across the mid-latitude band\n-> USA, China",
     touch=[29.5, 45.5])
draw(axes[2], azimuthal_equidistant_north, np.arange(-180, 181, 30), np.arange(30, 91, 15),
     [(lo, la) for lo in (0, 90, 180, 270) for la in (78, 55, 35)],
     "Azimuthal (equidistant, polar)",
     "Plane touches at the North Pole: true at the centre,\n"
     "distortion grows with distance from the pole\n-> Arctic / North Pole")
fig.suptitle("8.2  Equal-size circles on the globe under the three projection families",
             weight="bold", fontsize=13)
fig.tight_layout()
fig.savefig("charts/q2_projection_distortion.png", bbox_inches="tight", dpi=120)
plt.close(fig)

print("Tutorial 8 chart saved to ./charts/")
