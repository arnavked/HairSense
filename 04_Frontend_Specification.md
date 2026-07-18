# HairMatch — Frontend Specification (V1)

**Version:** 1.0
**Status:** Draft for review
**Companion to:** 01_PRD_V1.md, 02_Technical_Architecture.md

---

## 1. Overview

Frontend scope matches the V1 user journey defined in the PRD: upload → quality check feedback → results → recommendations → barber sheet. No live camera, no try-on overlay, no chat, no account UI (pending the open question in the Security & Access doc).

## 2. Screen Flow

```
Landing
   │
   ▼
Upload
   │
   ▼
Quality Check Feedback (only shown on failure)
   │
   ▼
Questions (texture, maintenance)
   │
   ▼
Results Dashboard (ranked recommendations)
   │
   ▼
Style Detail (explanation)
   │
   ▼
Barber Sheet
```

## 3. Screen-by-Screen Spec

### 3.1 Landing
- Purpose: explain what the product does in one line, single CTA to start
- Elements: headline, short description, "Upload a photo" button
- No account/login elements (per current scope)

### 3.2 Upload
- Elements: drag-and-drop or file picker, photo preview before submission, submit button
- Client-side pre-checks before submission: file type (JPEG/PNG only), file size limit — fail fast before hitting the backend quality check
- Loading state while quality check + classification run

### 3.3 Quality Check Feedback (conditional)
- Only shown if the backend quality check fails
- Must show the *specific* reason (too dark, too blurry, face angle too steep, multiple faces detected, etc.), not a generic error
- Clear path back to re-upload

### 3.4 Questions
- Two short questions, each a simple selector (not free text): hair texture (straight/wavy/curly/coily), maintenance preference (low/medium/high)
- Should feel like 10 seconds of effort, not a form

### 3.5 Results Dashboard
- Shows face shape result with confidence (e.g. "Oval — 91% confidence")
- Shows ranked list of recommended styles, each with its compatibility score (e.g. French Crop 96%, Textured Quiff 91%, Side Part 84%)
- Each entry tappable to go to Style Detail
- Ranked list, not just a single "best" result — per FR-12 in the PRD

### 3.6 Style Detail
- Shows the plain-language explanation for why this style was recommended (from the Explanation Engine)
- Shows the compatibility score breakdown if useful (optional, not required for V1)
- CTA: "Get barber instructions"

### 3.7 Barber Sheet
- Structured, scannable layout: sides guard/length, top length, texture/cut method, finish, maintenance interval
- Designed to be shown directly to a barber (readable at a glance, not paragraph text)
- Option to go back and compare a different style from the ranked list

## 4. Component Notes

- Confidence scores and compatibility scores should be presented consistently across the Results Dashboard and Style Detail (same visual treatment — e.g. percentage + simple bar or badge)
- Error states (quality check failure, network failure) need explicit designs, not just a generic toast
- No animations/interactions beyond standard loading and transition states are required for V1 — this is a functional-first spec, not a polish pass

## 5. Explicitly Out of Scope for V1 Frontend

- No live camera capture UI
- No try-on overlay/preview screen
- No AI chat interface
- No account/login/settings screens
- No "save/compare looks" screen

## 6. Open Questions

- Should the Results Dashboard show all catalog matches or a capped top-N (e.g. top 5)? Affects both UI density and how much the Recommendation Engine needs to return.
- Any existing design system/component library to align with, or is this being designed from scratch for V1?
