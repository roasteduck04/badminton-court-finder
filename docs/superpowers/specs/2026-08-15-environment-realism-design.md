# Environment Realism Improvements — Design Spec

**Date**: 2026-08-15
**Goal**: Make the Blender synthetic data pipeline produce images that closely resemble real badminton venue photographs, by adding procedural floor textures, wall/ceiling details, articulated player models, improved seating, and courtside clutter. All improvements are fully procedural (no external texture files).

**Priority**: Realism first — render time is secondary.

## Overview

Ten categories of improvement across 3 new files and 3 enhanced existing files. Every element is randomized per render to maximize training data diversity.

**New modules:**
- `src/tools/blender_mannequin.py` — Articulated player figures with pose presets
- `src/tools/blender_venue_details.py` — Wall banners, ceiling trusses, light housings, seating upgrades, clutter, line judge chairs

**Enhanced modules:**
- `src/tools/blender_environment.py` — Procedural floor textures, floor markings
- `src/tools/blender_court.py` — Net post improvements, court surface texture
- `src/tools/blender_occluders.py` — Wire in mannequin builder
- `src/tools/blender_render.py` — Call venue details in render loop

## 1. Procedural Floor Textures & Markings

**File**: `blender_environment.py` (enhance), `blender_court.py` (enhance)

### Venue Floor Textures

Replace flat `Principled BSDF` single-color materials with procedural node graphs:

**Concrete floor:**
- Base color + `Noise Texture` (scale 50-100) mixed via `MixRGB` for surface grain
- `Musgrave Texture` (scale 200-400) multiplied into color for darkened wear patches
- Roughness driven by same noise at different scale (0.7-0.95 range)

**Rubber floor:**
- `Voronoi Texture` (scale 30-60, cell mode) for rubber mat tile patterns
- Subtle color variation between tiles via `ColorRamp`
- High roughness (0.85-0.95) with slight variation

**Wood floor:**
- `Wave Texture` (bands, scale 20-40) for wood grain direction
- `Noise Texture` mixed in for knot patterns
- Roughness variation (0.5-0.7) following grain

### Floor Markings (venue floor, not court surface)

- **Safety zones**: Colored tape outlines (thin plane rectangles, ~5cm wide) 1-2m outside each court's boundaries. Random color from (yellow, red, blue). ~60% chance per render.
- **Court number**: Flat colored rectangle (0.8 × 0.4m) with a contrasting stripe, placed at one baseline of the main court. ~40% chance.
- **Warm-up areas**: Simple colored rectangles (3 × 2m) in 1-2 corners of the venue, away from courts. ~30% chance.

### Court Surface Texture (blender_court.py)

- Add `Noise Texture` (scale 100-200, low factor ~0.05) mixed into base color for subtle surface variation
- Tape residue: 0-5 small random rectangles (0.1-0.3m) at Z=0.0005, slightly different roughness (0.6 vs surface 0.7), ~40% chance
- Large-scale color gradient: `Gradient Texture` with very low influence (~0.03) for uneven wear

## 2. Wall & Ceiling Details

**File**: New `blender_venue_details.py`

### Wall Details

**Sponsor banners:**
- 2-6 rectangular planes per wall, placed at 1-3m height
- Sizes: 1-3m wide × 0.5-1.5m tall
- Procedural pattern: `Wave Texture` + `ColorRamp` (2-3 color stops from a bright saturated palette) to simulate text/logo blocks
- Random rotation between landscape/portrait
- Not placed on all 4 walls — pick 1-3 walls randomly

**Exit signs:**
- Small rectangles (0.3 × 0.15m) near ceiling on 1-2 walls
- Green or red emissive material (emission strength 2-5)
- Position: centered on wall, 0.5m below ceiling

**Windows:**
- 0-2 walls get window patches (large light rectangles, 2-4m wide × 1-2m tall)
- Material: lighter wall color with slight emissive glow (strength 0.5-1.5) simulating daylight
- Position: upper portion of wall (top third)
- ~40% of venues have windows, ~60% don't (underground/enclosed halls)

**Scoreboard upgrade:**
- Add frame border: thin cubes forming a rectangle around the existing scoreboard plane
- Emissive face material (strength 1-3) for backlit appearance
- Stand improvement: two thin cylinders as legs instead of floating

### Ceiling Details

**Exposed trusses:**
- 3-6 horizontal beams spanning venue width, evenly spaced along length
- Built from cubes (0.15 × venue_width × 0.15m)
- Cross-bracing: diagonal thin cubes connecting adjacent beams (~0.05m thick)
- Metallic grey material (roughness 0.4, metallic 0.7)
- ~50% chance of appearing

**Ventilation ducts:**
- 1-3 rectangular box shapes (0.4 × 0.4m cross-section, 3-8m long)
- Running parallel to venue length, offset from center
- Metallic material, positioned 0.5m below ceiling
- ~40% chance

**Hanging banners:**
- 0-3 vertical rectangular planes (1-2m wide × 2-4m tall)
- Hanging from ceiling via thin cylinder "cables"
- Bright procedural colors (same banner material as wall banners)
- Positioned at venue edges, not directly over courts
- ~30% chance

**Light housings:**
- Rectangular box (1.5 × 0.8 × 0.3m) centered on each area light position
- White/light grey material with slight emissive edge strips
- Automatically positioned by reading light locations from the Lighting collection
- Always present (real venues always have visible fixtures)

## 3. Articulated Mannequin Players

**File**: New `blender_mannequin.py`

### Body Part Geometry

All parts are cylinders or spheres. Hierarchy is parent-child so rotating a joint moves all children:

| Part | Shape | Dimensions | Parent |
|------|-------|-----------|--------|
| Torso | Cylinder | r=0.18, h=0.50 | — (root) |
| Head | UV Sphere | r=0.11 | Torso |
| Upper Arm L/R | Cylinder | r=0.05, h=0.28 | Torso |
| Forearm L/R | Cylinder | r=0.04, h=0.25 | Upper Arm |
| Upper Leg L/R | Cylinder | r=0.07, h=0.40 | Torso |
| Lower Leg L/R | Cylinder | r=0.05, h=0.38 | Upper Leg |
| Shoes L/R | Cube | 0.12 × 0.08 × 0.05 | Lower Leg |
| Racket (optional) | Plane | 0.25 × 0.10 | Forearm R |
| Hair (optional) | Sphere/Cylinder | r=0.09, h=0.06 | Head |

### Pose Presets

Each preset defines joint angles (degrees) for shoulder, elbow, hip, knee. All angles get ±5-10° random jitter.

**Standing idle:**
- Arms: shoulder 5° forward, elbow 10° bent, hanging at sides
- Legs: straight (hip 0°, knee 0°)

**Ready stance:**
- Arms: shoulder 40° forward, elbow 90° bent, hands up
- Legs: hip 20° bent, knee 30° bent (crouching)
- Torso: slight forward lean (5°)

**Lunging (forehand/backhand variants):**
- Lead leg: hip 70° forward, knee 90° bent
- Trail leg: hip -10°, knee 5°
- Racket arm: shoulder 60° forward + 30° out, elbow 120°
- Off arm: shoulder 30° back, elbow 45°
- Torso: 15° lean toward lead leg
- Mirror left/right randomly for forehand vs backhand

**Serving:**
- Racket arm: shoulder 160° up, elbow 90° (racket overhead)
- Off arm: shoulder 70° forward, elbow 10° (tossing)
- Legs: hip 5° stagger, knee 10° slight bend
- Torso: 10° lean back

**Walking:**
- Arms: alternating shoulder ±15°, elbows 20°
- Legs: alternating hip ±20°, knee 10° on forward leg
- Pick phase randomly (left or right foot forward)

### Materials

**Clothing colors** (expanded palette):
- Shirts: navy, red, green, grey, white, yellow, blue, orange, pink, teal, black
- Shorts: black, navy, white, grey (independent from shirt)
- Upper body and lower body get separate materials

**Skin tones** (6-tone palette):
- (0.95, 0.82, 0.70), (0.80, 0.62, 0.47), (0.65, 0.48, 0.35),
  (0.50, 0.35, 0.25), (0.38, 0.26, 0.18), (0.28, 0.18, 0.12)
- Applied to head, forearms (visible skin)

**Shoes**: White or dark, small cubes at foot position

**Hair**: ~60% chance. Dark sphere or short cylinder on head top. Color from dark brown/black palette.

**Racket**: ~30% of players. Flat plane in dominant hand forearm. Dark frame color with lighter string area.

### Placement

- Same Gaussian court-position bias as current
- Random Y-axis rotation biased toward net (±30° from net-facing direction)
- 0-4 players per render (unchanged from current config)

### Interface

`build_mannequin(collection, index, config)` → dict with type, position, pose metadata

## 4. Seating & Spectator Upgrades

**File**: `blender_venue_details.py`

### Tiered Bleacher Seating (replaces current flat benches)

- 1-5 stepped rows, each row +0.4m height and +0.6m depth from previous
- Built from cubes, seat-colored material randomized per venue (blue, red, grey, orange, green)
- ~40% chance of individual seat backs (small cubes at 0.5m spacing along each row) vs plain bench rows
- Placement: behind one or both baselines (3-5m from court edge), optionally along one sideline (~20% chance)

### Seated Spectator Figures

- Simplified: torso cylinder + head sphere only (no limbs — they're seated, distant)
- 0-30% seat occupancy, randomized per row
- Varied clothing colors from expanded palette
- Only spawned when seating is present

### Courtside Furniture

**Player chairs:**
- 1-2 simple chairs (cube seat 0.4m + cube back 0.6m tall + 4 cylinder legs) at one sideline near the net
- Towel: flat colored rectangle draped over chair back, ~50% chance
- ~60% chance of appearing

**Drink station:**
- Small table: flat cube (0.6 × 0.4 × 0.7m)
- 1-3 tiny cylinders on top (water bottles, r=0.03, h=0.2)
- Positioned next to player chairs
- ~40% chance

**Equipment bags:**
- Rectangular cubes (0.5 × 0.3 × 0.3m) on floor near seating
- Dark colored (navy, black, grey)
- 0-3 per venue, ~50% chance

## 5. Net Post Improvements

**File**: `blender_court.py` (enhance)

**Post base plates:**
- Flat cylinder at ground level per post (r=0.15m, h=0.02m)
- Metallic material matching post
- Always present when net is enabled

**Net tension cable:**
- Thin cylinder (r=0.005m) connecting post tops across the net
- Same metallic material as posts
- Always present when net is enabled

**Post padding:**
- Cylinder wrapping lower 1m of each post (r=0.08m, h=1.0m)
- Bright colored material (yellow, blue, red — randomized)
- Higher roughness (0.9) for foam look
- ~30% chance

## 6. Light Housings

**File**: `blender_venue_details.py`

- Rectangular box (1.5 × 0.8 × 0.3m) at each area light position
- Material: white/light grey base, emissive edge strips (thin planes on bottom face, emission 1-2)
- Reads positions from Lighting collection objects at build time
- Always present — every real venue has visible fixtures
- Adds realism to overhead and low-angle shots where ceiling is visible

## 7. Line Judge / Service Judge Chairs

**File**: `blender_venue_details.py`

- Small folding chair: cube seat (0.4m) + thin cube back (0.4 × 0.02 × 0.3m) + 4 thin cylinder legs
- Positioned at 2-4 court corners, 1-2m outside baseline/sideline intersection
- ~25% chance of appearing (competition setting — correlates with competition lighting preset)
- Optional seated figure (same simplified spectator mannequin) on each chair, ~50% chance when chair present
- Material: metallic grey frame, dark fabric seat

## 8. Courtside Clutter

**File**: `blender_venue_details.py`

Each item has an independent spawn chance per render:

| Item | Geometry | Size | Chance | Placement |
|------|----------|------|--------|-----------|
| Mop bucket | Cylinder + thin cylinder handle | r=0.15, h=0.3 | 15% | Near wall, random position |
| Rolled-up net | Horizontal cylinder | r=0.15, L=2.0 | 10% | On floor near wall |
| Stacked chairs | 2-3 offset cubes | 0.4m each | 15% | Against wall |
| Training cones | Cones | r=0.1, h=0.2 | 20% | 0-4 near court edges |

All clutter placed at least 3m from the main court to avoid interfering with court visibility, but within camera view for background realism.

## Integration

### Render Loop Changes (`blender_render.py`)

Current order:
1. `build_court()` → 2. `build_environment()` → 3. `place_camera()` → 4. `setup_lighting()` → 5. `add_occluders()` → 6. Render

New order:
1. `build_court()` (enhanced with floor texture, net post improvements)
2. `build_environment()` (enhanced with procedural floor textures, floor markings)
3. `place_camera()`
4. `setup_lighting()`
5. `build_venue_details()` (NEW — needs lighting positions, venue bounds from environment)
6. `add_occluders()` (now uses mannequin builder)
7. Render

`build_venue_details()` runs after lighting so it can read light positions for housing placement. It receives venue bounds from `build_environment()` return value — specifically `venue_cx`, `venue_cy`, `venue_d`, `venue_w`, `venue_h` — which are added to the `build_environment()` return dict. The render loop passes these as:
```python
env_meta = build_environment()
# ... place_camera, setup_lighting ...
details_meta = build_venue_details({
    "venue_bounds": env_meta["venue_bounds"],  # {cx, cy, d, w, h}
})
```

### Metadata Changes

Environment metadata gains new fields:
```json
{
  "environment": {
    "adjacent_courts": 3,
    "venue_size": "medium",
    "has_dividers": true,
    "has_spectators": true,
    "has_scoreboard": true,
    "floor_type": "rubber",
    "has_floor_markings": true,
    "has_trusses": true,
    "has_ducts": false,
    "has_banners": true,
    "has_windows": false,
    "num_wall_banners": 4,
    "has_light_housings": true,
    "has_line_judges": false,
    "seating_type": "bleacher_with_seats",
    "spectator_count": 12,
    "clutter_items": ["mop_bucket", "training_cones"]
  }
}
```

Player occluder metadata gains pose info:
```json
{
  "type": "player",
  "position": [5.2, 3.1, 0],
  "pose": "lunging",
  "has_racket": true,
  "facing_angle": 1.2
}
```

### Module Interface Summary

| Module | Function | Inputs | Returns |
|--------|----------|--------|---------|
| `blender_mannequin.py` | `build_mannequin(collection, index, config)` | Collection, player index, optional config | Metadata dict (position, pose, has_racket) |
| `blender_venue_details.py` | `build_venue_details(config)` | Config with venue bounds, lighting info | Metadata dict (all detail flags) |
| `blender_environment.py` | `build_environment(config)` (enhanced) | Same as current | Same + floor_type, has_floor_markings, venue_bounds dict (venue_cx, venue_cy, venue_d, venue_w, venue_h) |
| `blender_court.py` | `build_court(config)` (enhanced) | Same as current | Same (net posts enhanced internally) |
| `blender_occluders.py` | `add_occluders(config)` (enhanced) | Same as current | Same + pose metadata per player |

## Constraints

- All materials fully procedural — no external image textures
- Blender 5.2+ compatibility — use `users_collection` pattern, `BLENDER_EEVEE` engine name
- No changes to keypoint positions or visibility computation
- No changes to CVN annotation format (converter untouched)
- Metadata additions are backward-compatible (new fields only)
- All randomization seeded via the existing `random.seed(SEED)` in render loop
