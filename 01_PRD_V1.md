# HairMatch — Product Requirements Document (V1)

**Version:** 1.0
**Status:** Draft for review
**Scope:** V1 buildable product (not the expanded "AI Hair Consultant" vision — see Future Roadmap)

---

## 1. Executive Summary

HairMatch is an AI-assisted hairstyle recommendation tool. A user uploads a photo, the system classifies their face shape, combines it with a few self-reported inputs (hair texture, maintenance preference), and returns a ranked, explained list of hairstyle recommendations — including a barber-ready instruction sheet for the top pick.

V1 is intentionally scoped to what can be built and validated with the data and team available now: a static-photo pipeline with a trained face-shape classifier, a transparent scoring engine, and LLM-assisted explanations. It deliberately excludes live camera/AR, 3D try-on, and any detection task (hair density, hairline, beard, skin tone) that would require datasets that don't currently exist.

## 2. Problem Statement

People struggle to know which hairstyles suit their face shape and hair type, and often can't communicate their desired style clearly to a barber. Existing apps (FaceShape AI, YouCam, Hiface) answer a single narrow question — "what shape is my face" — and stop there, without explaining *why* a style works or helping the user act on the recommendation (e.g. talking to a barber).

## 3. Product Goals

- Accurately classify face shape from a single photo
- Recommend hairstyles ranked by compatibility, not just a single "best" answer
- Explain *why* each recommendation fits, in plain language
- Produce a barber-ready instruction sheet the user can act on immediately
- Keep the system data-driven and extensible (adding new hairstyles shouldn't require code changes)

## 4. Non-Goals (V1)

- No live camera / AR filter-style overlay
- No 3D hair mesh or neural rendering try-on
- No hair density, hairline, beard, or skin tone detection (no usable labeled dataset available)
- No growth timeline simulation
- No celebrity look-alike matching
- No conversational AI chat for re-ranking recommendations

These are listed in the Future Roadmap (Section 10) as possible V2+ additions, not specced here.

## 5. Users

Single primary persona for V1: someone about to get a haircut who wants a data-backed recommendation and a way to describe it to their barber. No separate personas for barbers/salons in V1 — that's a future B2B extension, not in scope.

## 6. User Journey (V1)

1. User uploads a photo
2. System runs a quality check (blur, lighting, angle, occlusion, multiple faces) — rejects or warns on bad photos before proceeding
3. System detects the face and classifies face shape, with a confidence score
4. User answers 2-3 quick questions: hair texture (straight/wavy/curly/coily), maintenance tolerance (low/medium/high)
5. Recommendation engine scores the hairstyle catalog against face shape + inputs
6. User sees a ranked list of styles with compatibility scores and a plain-language explanation for each
7. User selects a style and gets a barber instruction sheet (sides length, top length, texture, finish, maintenance interval)

## 7. Functional Requirements

### 7.1 Image Intake & Quality Check
- FR-1: System accepts a single uploaded photo (JPEG/PNG)
- FR-2: System runs a quality check prior to classification: blur detection, lighting/exposure check, face yaw angle, occlusion check, multiple-face detection
- FR-3: If quality check fails, system returns a specific, actionable error (e.g. "photo too dark," "face at too steep an angle") rather than a silent bad prediction

### 7.2 Face Detection & Shape Classification
- FR-4: System detects the face and extracts landmarks
- FR-5: System classifies face shape using the trained classifier (SVM/RF/MLP → ONNX)
- FR-6: System returns a confidence score alongside the classification

### 7.3 User Input
- FR-7: User selects hair texture from a fixed set of options (self-reported, not detected)
- FR-8: User selects maintenance preference from a fixed set of options

### 7.4 Recommendation Engine
- FR-9: Hairstyle catalog is stored as structured data (JSON), not hardcoded in application logic
- FR-10: Engine is split into distinct stages: Feature Extraction → Compatibility Scoring → Ranking → Explanation → Barber Sheet
- FR-11: Compatibility score is a weighted function of face shape, hair texture, and maintenance preference against each catalog entry
- FR-12: Engine returns a ranked list (not just a single top result) with a score per entry

### 7.5 Explanation Engine
- FR-13: System generates a plain-language explanation for each recommended style, abstracted behind an "Explanation Engine" interface (V1 implementation: template + LLM polishing; implementation is swappable without changing the engine's contract)

### 7.6 Barber Instruction Sheet
- FR-14: For a selected style, system generates a structured instruction sheet: sides length/guard, top length, texture/cut method, finish, maintenance interval

## 8. Non-Functional Requirements

- Photo processing (quality check + classification) should complete within a few seconds on typical hardware
- System should degrade gracefully on low-quality photos (clear error, not a silent wrong answer)
- Hairstyle catalog additions should require no code changes, only data entries
- Explanation Engine implementation must be swappable (LLM vendor, template logic, or future fine-tuned model) without changing calling code

## 9. Success Metrics

- Face shape classifier accuracy on a held-out validation set (target to be set once first training run completes)
- % of uploaded photos that pass quality check on first try
- Qualitative: does the explanation for a recommendation feel specific and trustworthy, not generic, in informal user testing

## 10. Future Roadmap (not specced in V1)

- Live camera capture with real-time overlay
- 3D hair mesh / neural rendering try-on
- Hair density, hairline, beard, skin tone detection (pending viable datasets)
- Growth timeline simulator
- Celebrity look-alike matching
- Conversational AI chat for adjusting recommendations
- Barber/salon-facing tools

## 11. Open Questions

- What's the target/minimum acceptable accuracy for the face-shape classifier before V1 ships?
- Is the hairstyle catalog being built manually (curated by you/team) or sourced from an existing dataset?
- Any preference on LLM provider for the Explanation Engine's V1 implementation?
