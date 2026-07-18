# HairMatch — Technical Architecture (V1)

**Version:** 1.0
**Status:** Draft for review
**Companion to:** 01_PRD_V1.md

---

## 1. Overview

This document describes the system architecture for the V1 scope defined in the PRD: static-photo upload → quality check → face shape classification → recommendation → explanation → barber sheet. No live camera, no 3D rendering, no additional detection models beyond face shape.

## 2. High-Level Pipeline

```
Photo
  │
  ▼
Quality Assessment
  │
  ▼
Face Detection
  │
  ▼
Face Landmark Extraction
  │
  ▼
Face Shape Classification
  │
  ▼
User Inputs (texture, maintenance)
  │
  ▼
Recommendation Engine
  │   ├─ Feature Extraction
  │   ├─ Compatibility Scoring
  │   ├─ Ranking
  │   ├─ Explanation Engine
  │   └─ Barber Sheet Generator
  ▼
Response to Client
```

## 3. Component Breakdown

### 3.1 Quality Assessment
- Input: raw uploaded image
- Checks: blur (e.g. Laplacian variance threshold), lighting/exposure (brightness histogram), face yaw angle (from landmarks), occlusion, multiple-face detection
- Output: pass/fail + specific failure reason if applicable
- Runs before any classification work, to avoid wasting downstream compute on unusable photos and to avoid silently wrong predictions

### 3.2 Face Detection & Landmark Extraction
- Existing component (already built) — detects face and extracts landmarks used both for quality checks (yaw angle) and as input to the face shape classifier

### 3.3 Face Shape Classification
- Existing/in-progress component — trained classifier (SVM/RF/MLP, exported to ONNX) consuming landmark-derived features
- Output: predicted class + confidence score

### 3.4 Recommendation Engine (split architecture)

Rather than one monolithic block, the engine is split into five stages so each can be built, tested, and modified independently:

**Feature Extraction**
Combines classifier output (face shape + confidence) with user-provided inputs (hair texture, maintenance preference) into a single feature set for scoring.

**Compatibility Scoring**
Evaluates the feature set against every entry in the hairstyle catalog, producing a weighted compatibility score per style. Weights and scoring logic live in this stage only — catalog entries stay declarative data.

**Ranking**
Sorts scored entries, returns the ranked list (not just the top result).

**Explanation Engine**
Takes a ranked entry and produces a plain-language explanation. Defined behind a stable interface (input: style + user features + score; output: explanation text) so the underlying implementation is swappable. V1 implementation: template + LLM polishing. Later implementations (fine-tuned model, pure rule-based, different LLM vendor) can be swapped in without changing any calling code.

**Barber Sheet Generator**
Takes the selected style and produces the structured instruction sheet (sides guard length, top length, texture/cut method, finish, maintenance interval). V1 implementation: template-based, populated from catalog fields for the selected style.

### 3.5 Hairstyle Catalog
- Stored as structured data (JSON), not hardcoded in application logic
- Schema per entry: name, compatible face shapes, compatible hair textures, maintenance level, length, barber sheet defaults (guard sizes, texture, finish)
- Adding or editing styles requires only a data change, not a code change

## 4. Data Flow Example

```
User uploads photo
  → Quality check passes
  → Face shape: Oval (confidence 0.91)
  → User selects: texture = Wavy, maintenance = Low
  → Feature Extraction combines these
  → Compatibility Scoring evaluates catalog (30 entries) → scores
  → Ranking returns top N: French Crop (0.96), Textured Quiff (0.91), Side Part (0.84)
  → User selects French Crop
  → Explanation Engine generates: "We recommend a French Crop because..."
  → Barber Sheet Generator returns structured cut instructions
```

## 5. Technology Choices (V1)

- **CV/ML inference:** ONNX Runtime for the face shape classifier (already the plan)
- **Recommendation engine:** plain application logic, no ML model required for scoring — a weighted function over structured data
- **Explanation Engine:** LLM API call (provider left as an implementation detail behind the interface, not locked into the architecture)
- **Catalog storage:** JSON file or simple database table — full database not required for V1 unless user accounts are in scope (see Security & Access doc)

## 6. What's Explicitly Out of This Architecture

- No live video/camera pipeline
- No 3D mesh generation or rendering
- No additional CV models (hair density, hairline, beard, skin tone) — the pipeline has no stage for these in V1
- No model retraining pipeline documented here — that belongs to the classifier's own training workflow, already underway separately

## 7. Open Questions

- Where does the hairstyle catalog live — flat JSON file shipped with the app, or a lightweight database table? Depends on whether the catalog will be edited outside of code deploys.
- Is user account/history storage in scope for V1, or is this a stateless single-session flow? (Affects whether a database is needed at all for V1.)
