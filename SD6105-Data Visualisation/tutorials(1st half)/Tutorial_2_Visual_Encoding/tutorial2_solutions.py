"""
SD6105 Data Visualisation - Tutorial 2 Solutions
Topic: Visual Encoding
No dataset required — conceptual + sketch-based exercises
"""

# =============================================================================
# SECTION 2.1: VISUAL MARKS AND VARIABLES
# =============================================================================

"""
Q1 - Visual Marks for Items (Bertin):
Visual marks are the basic geometric elements in a graphic that depict items.
The three visual marks proposed by Jacques Bertin (1967) are:

1. POINT (Zero-dimensional)
   - Represents a single item at a specific location
   - No direction or length
   - Example: Dot on scatter plot, city on map

2. LINE (One-dimensional)
   - Has length and direction but no area
   - Used to represent paths, trajectories, relationships, or continuous change
   - Example: Line on line chart, road on map

3. AREA (Two-dimensional)
   - Has both length and width, forming a closed shape
   - Represents extent, region, or magnitude through 2D space
   - Example: Bar in bar chart, region on choropleth map

Key differences:
- Points: position only (0D) — best for individual items in space
- Lines: position + direction (1D) — best for connections, trends, flows
- Areas: position + extent (2D) — best for showing magnitude proportionally

Q2 - Perceptual Channels (Visual Variables) for Line Mark:
Three perceptual channels proposed by Bertin:

1. VALUE (Intensity/Brightness)
   - Changes the lightness/darkness of the line
   - E.g., a dark line vs. light grey line
   - Effect on line: line appears more/less prominent

2. COLOUR (Hue)
   - Changes the colour (hue) of the line
   - E.g., red line vs. blue line
   - Effect on line: encodes nominal categories (e.g., different countries)

3. SIZE (Width/Thickness)
   - Changes the width/thickness of the line
   - E.g., thick line vs. thin line
   - Effect on line: encodes quantitative values (e.g., volume of traffic flow)

(Other channels also applicable: shape [dashed/dotted], orientation, texture)

Q3 - Telecommunication Traffic Flow Map Analysis:

Visual Marks Employed:
- LINES: Used to represent telecommunication traffic routes between cities/countries
- POINTS: Used to mark geographic node locations (cities/hubs)
- AREAS: Implicit — geographic map regions form background areas

Perceptual Channels Employed:
1. SIZE (line width/thickness):
   - Encodes the VOLUME of traffic between nodes (thicker = more traffic)
   - This is quantitative data (Ratio scale)

2. COLOUR HUE:
   - Different hues encode different regions or traffic route categories
   - Nominal scale data (which region/carrier the route belongs to)

3. POSITION:
   - Geographic position of nodes encodes the spatial location of each hub
   - Ratio scale (lat/long coordinates)

4. LENGTH:
   - Line length implicitly shows geographic distance between hubs
   - Interval/Ratio scale

Potential Limitations:
- CLUTTER: Many overlapping lines in dense hub areas make individual routes hard to trace
- SIZE OVERPLOTTING: Very small traffic volumes disappear when widths get too narrow
- COLOUR DISCRIMINATION: Too many hues makes the legend large and color matching difficult
  (exceeds the ~12 hue limit for effective categorical distinction)
- OVERLAP AMBIGUITY: Thick lines crossing each other create visual confusion about
  whether connections are direct or via intermediate hubs
"""

# =============================================================================
# SECTION 2.2: VISUAL ENCODING DESIGN
# =============================================================================

"""
Q1 - Chart Designs for Three Data Tables:

(a) Lines Chart — Living Members from Class of '45
    Data: Names (Nominal), Gender (Nominal), Age (Ratio)

    DESIGN: Horizontal bar chart (using LINES as marks)
    - Each person = one horizontal line (bar)
    - Bar LENGTH encodes Age (Ratio → Length channel, highly effective)
    - BAR COLOUR encodes Gender (Nominal → Hue channel)
      - Blue bars for Male; Red/Pink bars for Female
    - POSITION (Y-axis) shows each individual (Nominal)
    - Sorted by age descending for easy comparison

    Assumptions: Ages are diverse enough to warrant bar chart;
    all members are from 1945 graduating class (similar baseline age range ~97-100)

(b) Points Chart — Orientation T-shirts: Comparing Halls
    Data: No., Gender (Nominal), Weight (Ratio), Height (Ratio), Hall (Nominal), T-shirt Size (Ordinal)

    DESIGN: Scatter plot (using POINTS as marks)
    - X-axis: Weight (Ratio → Position, most effective for quantitative)
    - Y-axis: Height (Ratio → Position)
    - COLOUR HUE: Hall A/B/C/D (Nominal → Hue, 4 colours easily distinguished)
    - SHAPE: Gender (Nominal → Shape — circle for female, triangle for male)
    - SIZE: T-shirt Size (Ordinal → Size channel, encodes 5 sizes XS→XL)

    This creates a 5-dimensional scatter plot using position (2), colour, shape, size.

    Assumptions: Weight and Height treated as continuous Ratio data; T-shirt size
    treated as Ordinal (XS < S < M < L < XL — 5 ordered levels mapped to dot sizes)

    Visual Quality Problem: Overplotting may occur if many students have similar
    weight/height. Solution: Add jitter (random noise) to separate overlapping dots,
    or use transparency (alpha=0.6).

(c) Areas Chart — COVID-19 Infection in 2020
    Data: Dates (Interval/Temporal), Total Cases (Ratio), Warded (Ratio), ICU (Ratio)

    DESIGN: Stacked Area Chart (using AREAS as marks)
    - X-axis: Dates (Interval → Position, temporal axis)
    - Y-axis: Number of cases (Ratio)
    - AREA 1 (bottom): ICU cases — dark red (most severe, draw attention)
    - AREA 2 (middle): Warded cases — medium orange
    - AREA 3 (top): Total − Warded (community/mild cases) — light yellow/green

    Stacked areas show composition over time:
    - Top line = Total Cases
    - Area proportions reveal severity distribution changes

    Assumptions: 'Warded' includes ICU; we can compute mild = Total − Warded.
    Dates treated as continuous temporal Interval scale.

    Visual Quality Problem: Intermediate areas (Warded) are plotted against
    varying baselines, making exact values hard to read. Solution: Add reference
    lines or annotations for key peaks, or use multiple line charts for precise comparison.

Q2 - Potential Visual Quality Problems:
- OVERPLOTTING (scatter plots): Many data points overlap, obscuring density
  Solution: Transparency (alpha), jitter, 2D density heatmap, or hexbin plot
- VARYING BASELINE (stacked area): Only topmost and bottommost categories
  have stable baselines; intermediate layers difficult to compare
  Solution: Use 100% normalised stacking for composition, or separate line charts
- NOMINAL COLOUR LIMITS: More than ~12 categories creates indistinguishable colors
  Solution: Combine small categories into "Other", use shape/texture as redundancy
"""

# =============================================================================
# SECTION 2.3: DECONSTRUCTION - MIDDLE EAST FRIENDSHIP CHART
# =============================================================================

"""
Exercise 1 - Middle East Friendship/Enemy Chart (Syria Civil War)

(a) Data Table Structure:
The underlying data is a RELATIONAL dataset with entities and relationships.

Node Table (entities):
| Player_ID | Player_Name | Type (Country/Group) |
|-----------|-------------|----------------------|
| 1         | Syria       | Country              |
| 2         | USA         | Country              |
| 3         | ISIS        | Non-state group      |
| 4         | Russia      | Country              |
...

Relationship Table (links):
| Player_A_ID | Player_B_ID | Relationship_Type  |
|-------------|-------------|-------------------|
| 1           | 4           | Friend            |
| 1           | 2           | Enemy             |
| 2           | 3           | Enemy             |
| 1           | 3           | Enemy             |
...

(b) Visual Marks and Variables Used:
Marks:
- AREAS (circles): Each player = one circle node
- LINES (connections): Each relationship = one connecting line

Channels:
- COLOUR HUE of circles: Red = "Enemies" relationship, Green = "Friends", Yellow smiley = "Complicated"
- SIZE of circles: Possibly encodes importance/relevance of player (larger = more connections)
- POSITION: Layout algorithm (force-directed) places highly connected nodes centrally
- COLOUR of connections: Matches relationship type colour

(c) Redundant Coding:
YES — redundant coding used:
- Relationship type encoded by BOTH circle colour (red/green/yellow) AND smiley face icons
- This is effective redundancy: colour provides pre-attentive pop-out; icons provide
  explicit categorical labels accessible to colour-blind viewers
- However, the colour of the connecting LINES vs. the colour of circles may create
  confusion if they don't consistently encode the same attribute

(d) Colour Appropriateness:
Generally appropriate choices:
- Red = Enemies: Culturally universal (danger, conflict, opposition) — very appropriate
- Green = Friends: Positive, allied — appropriate
- Yellow = Complicated: Neutral/ambiguous colour — appropriate
However, limitations:
- Red/Green combination problematic for red-green colour blind viewers (~8% of males)
  Better: Use red/blue with different shapes, OR add text labels as backup
- Too many overlapping green/red circles creates visual noise in dense regions
"""

# =============================================================================
# SECTION 2.4: CRITICAL EYE — BBC CHARTS
# =============================================================================

"""
BBC News: "Five Charts about the Fortunes of the Chinese Family" (Oct 2017)

(a) Marriages down, divorces up — Line Chart
EFFECTIVE: Dual-line chart with diverging trends clearly shows the marriage/divorce
crossover trend. Simple, clean design. Direct labeling helps avoid legend lookup.
IMPROVEMENT: Y-axes should start at 0 to avoid exaggerating the rate of change.
Adding a reference annotation for the crossover point would strengthen the message.

(b) China's "army of leftover men" — Bar/Infographic
CRITIQUE: The use of human silhouettes (pictograph) is engaging but can distort
proportions if silhouettes are different sizes or not accurately scaled.
The "114 men per 100 women" ratio is clear but the visual comparison via
mismatched silhouette counts may mislead about the magnitude.
IMPROVEMENT: A simple diverging bar or grouped bar chart comparing age-group
sex ratios would be more precise and less prone to misinterpretation.

(c) Chinese millennials — Comparative Bar Chart
EFFECTIVE: Simple two-bar comparison (China vs USA homeownership) is clear.
CRITIQUE: Only shows two data points — lacks context about other countries.
The colour choice (if red for China, matching national flag) is culturally
appropriate but may bias perception if red is used for "positive."
IMPROVEMENT: Add 3-4 comparable countries for context.

(d) Mobile cellular subscriptions — Line Chart
EFFECTIVE: Shows dramatic rise to near-parity (close to 100 per 100 people).
CRITIQUE: The scale matters here — starting from 0 or from 80 would create
very different visual impressions of the rate of growth.
IMPROVEMENT: Highlight the inflection point where mobile exceeded one-per-person.

(e) Chinese students abroad — Bar/Area Chart
EFFECTIVE: Clearly shows exponential growth trend.
CRITIQUE: If using area/height encoding, ensure baseline starts at zero.
IMPROVEMENT: Adding comparative data (students from India, South Korea) would
give context to whether China's growth is unique or a global trend.
"""

# =============================================================================
# PYTHON CODE: Demonstrating Visual Encoding Concepts
# =============================================================================

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# --- Demonstration: Bertin's 7 Visual Variables on Line Marks ---
fig, axes = plt.subplots(3, 3, figsize=(12, 9))
fig.suptitle("Bertin's Visual Variables Applied to Line Marks", fontsize=14, fontweight='bold')

x = np.linspace(0, 1, 50)
y = np.ones(50) * 0.5

channels = [
    ('1. Position', lambda ax: [
        ax.plot(x, y + 0.15, color='black', lw=2),
        ax.plot(x, y - 0.15, color='black', lw=2),
        ax.set_title('Position (Y)', fontsize=10)]),
    ('2. Size (Width)', lambda ax: [
        ax.plot(x, y + 0.1, color='black', lw=1),
        ax.plot(x, y - 0.1, color='black', lw=5),
        ax.set_title('Size (Line Width)', fontsize=10)]),
    ('3. Value (Intensity)', lambda ax: [
        ax.plot(x, y + 0.1, color='black', lw=2),
        ax.plot(x, y - 0.1, color='lightgray', lw=2),
        ax.set_title('Value (Lightness)', fontsize=10)]),
    ('4. Colour (Hue)', lambda ax: [
        ax.plot(x, y + 0.1, color='steelblue', lw=2),
        ax.plot(x, y - 0.1, color='darkorange', lw=2),
        ax.set_title('Colour (Hue)', fontsize=10)]),
    ('5. Shape (Pattern)', lambda ax: [
        ax.plot(x, y + 0.1, color='black', lw=2, linestyle='-'),
        ax.plot(x, y - 0.1, color='black', lw=2, linestyle='--'),
        ax.set_title('Shape (Dash Pattern)', fontsize=10)]),
    ('6. Orientation', lambda ax: [
        ax.plot([0.2, 0.8], [0.3, 0.7], color='black', lw=2),
        ax.plot([0.2, 0.8], [0.7, 0.3], color='black', lw=2),
        ax.set_title('Orientation (Slope)', fontsize=10)]),
    ('7. Texture', lambda ax: [
        ax.plot(x, y + 0.1, color='black', lw=2, linestyle=(0, (1, 1))),
        ax.plot(x, y - 0.1, color='black', lw=2, linestyle=(0, (5, 5))),
        ax.set_title('Texture (Dot Density)', fontsize=10)]),
]

for i, (name, draw_fn) in enumerate(channels):
    ax = axes[i // 3][i % 3]
    draw_fn(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

axes[2][1].set_visible(False)
axes[2][2].set_visible(False)

plt.tight_layout()
output_path = "/Users/sh01840ml/Desktop/NTUDS/Core_Courses/SD6105_Data_Visualisation/completed_tutorials/Tutorial_2_Visual_Encoding/charts/"
plt.savefig(output_path + "bertin_visual_variables_demo.png", dpi=150, bbox_inches='tight')
plt.show()

# --- Demonstration: Mackinlay's Effectiveness Ranking (Bar Chart) ---
fig, axes = plt.subplots(1, 3, figsize=(15, 8))

channels_list = ['Position', 'Length', 'Angle', 'Slope', 'Area', 'Volume',
                 'Density', 'Color Sat.', 'Color Hue', 'Texture',
                 'Connection', 'Containment', 'Shape']

quant_rank  = list(range(1, 14))
ordinal_rank = [1, 8, 9, 10, 11, 12, 2, 3, 4, 5, 6, 7, 13]
nominal_rank = [1, 9, 10, 11, 12, 13, 6, 7, 2, 3, 4, 5, 8]

titles = ['Quantitative', 'Ordinal', 'Nominal']
ranks_list = [quant_rank, ordinal_rank, nominal_rank]
colors = ['#2196F3', '#FF9800', '#4CAF50']

for ax, title, ranks, color in zip(axes, titles, ranks_list, colors):
    sorted_idx = np.argsort(ranks)
    sorted_channels = [channels_list[i] for i in sorted_idx]
    sorted_effectiveness = [14 - r for r in sorted(ranks)]  # higher bar = more effective

    bars = ax.barh(sorted_channels, sorted_effectiveness, color=color, alpha=0.8, edgecolor='white')
    ax.set_xlabel('Effectiveness (higher = better)', fontsize=10)
    ax.set_title(f'{title} Data\nMackinlay\'s Ranking', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 14)
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    for bar, rank in zip(bars, sorted(ranks)):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'#{rank}', va='center', fontsize=8)

plt.suptitle("Mackinlay's Channel Effectiveness Rankings by Data Type", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(output_path + "mackinlay_effectiveness_ranking.png", dpi=150, bbox_inches='tight')
plt.show()

print("Tutorial 2 complete — conceptual answers and demo charts generated.")
