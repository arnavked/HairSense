# HairMatch — Feature Ticket List (V1)

**Version:** 1.0
**Status:** Draft for review
**Companion to:** 01_PRD_V1.md, 02_Technical_Architecture.md, 03_Security_Access_Document.md, 04_Frontend_Specification.md

---

## Notes on Structure

Tickets are grouped by area and roughly ordered by dependency. Status reflects what's already known to be done or in progress from prior work; everything else is planned, not started.

Legend: ✅ Done · 🔄 In Progress · ⬜ Not Started

---

## CV / ML

| ID | Ticket | Status |
|---|---|---|
| CV-001 | Face detection + landmark extraction | ✅ Done |
| CV-002 | Face-shape labeling tool for training data | ✅ Done |
| CV-003 | Train face-shape classifier (SVM/RF/MLP) | 🔄 In Progress |
| CV-004 | Export classifier to ONNX | 🔄 In Progress |
| CV-005 | Validate classifier accuracy on held-out set | ⬜ |
| CV-006 | Image quality assessment: blur detection | ⬜ |
| CV-007 | Image quality assessment: lighting/exposure check | ⬜ |
| CV-008 | Image quality assessment: yaw angle check (from landmarks) | ⬜ |
| CV-009 | Image quality assessment: occlusion + multiple-face detection | ⬜ |
| CV-010 | Wire quality assessment as a pre-check stage before classification | ⬜ |

## Recommendation Engine

| ID | Ticket | Status |
|---|---|---|
| RE-001 | Define hairstyle catalog JSON schema | ⬜ |
| RE-002 | Populate initial hairstyle catalog (target: ~30 entries) | ⬜ |
| RE-003 | Feature Extraction stage: combine classifier output + user inputs | ⬜ |
| RE-004 | Compatibility Scoring stage: weighted scoring function against catalog | ⬜ |
| RE-005 | Ranking stage: sort and return top-N scored entries | ⬜ |
| RE-006 | Unit tests for scoring logic against known catalog entries | ⬜ |

## Explanation Engine

| ID | Ticket | Status |
|---|---|---|
| EX-001 | Define Explanation Engine interface (input/output contract) | ⬜ |
| EX-002 | V1 implementation: template + LLM polishing | ⬜ |
| EX-003 | Confirm LLM provider and data-handling approach (per Security doc open question) | ⬜ |
| EX-004 | Prompt for explanation generation (face shape + texture + style → explanation text) | ⬜ |

## Barber Sheet

| ID | Ticket | Status |
|---|---|---|
| BS-001 | Define barber sheet fields per catalog entry (guard sizes, texture, finish, maintenance interval) | ⬜ |
| BS-002 | Barber Sheet Generator: populate template from selected catalog entry | ⬜ |

## Backend / API

| ID | Ticket | Status |
|---|---|---|
| BE-001 | Photo upload endpoint (with type/size validation) | ⬜ |
| BE-002 | Orchestration: quality check → classification → response | ⬜ |
| BE-003 | Recommendation endpoint (inputs: face shape, texture, maintenance → ranked results) | ⬜ |
| BE-004 | Explanation endpoint | ⬜ |
| BE-005 | Barber sheet endpoint | ⬜ |
| BE-006 | Rate limiting on upload/analysis endpoints | ⬜ |
| BE-007 | Logging: ensure no raw photo/landmark data is logged (per SA-10) | ⬜ |
| BE-008 | Ensure photos are not persisted beyond request lifecycle (per SA-1) | ⬜ |

## Frontend

| ID | Ticket | Status |
|---|---|---|
| FE-001 | Landing screen | ⬜ |
| FE-002 | Upload screen (with client-side pre-checks) | ⬜ |
| FE-003 | Quality check failure feedback screen | ⬜ |
| FE-004 | Questions screen (texture, maintenance) | ⬜ |
| FE-005 | Results dashboard (ranked list with scores) | ⬜ |
| FE-006 | Style detail screen (explanation) | ⬜ |
| FE-007 | Barber sheet screen | ⬜ |
| FE-008 | Loading states across the flow | ⬜ |
| FE-009 | Error states (network failure, generic backend error) | ⬜ |

## Cross-Cutting

| ID | Ticket | Status |
|---|---|---|
| XC-001 | HTTPS/TLS on all endpoints | ⬜ |
| XC-002 | End-to-end test: full flow from upload to barber sheet | ⬜ |

---

## Explicitly Not Ticketed (Future Roadmap, not V1)

Live camera/AR, 3D try-on, hair density/hairline/beard/skin-tone detection, growth simulator, celebrity matching, AI chat re-ranking, accounts/save-and-compare, barber/salon-facing tools.
