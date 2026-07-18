# HairMatch — Security & Access Document (V1)

**Version:** 1.0
**Status:** Draft for review
**Companion to:** 01_PRD_V1.md, 02_Technical_Architecture.md

---

## 1. Overview

V1 is a stateless, single-session photo analysis flow (per the open question in the PRD — this document assumes no user accounts unless that's confirmed otherwise; see Section 6). Even without accounts, the system handles facial images, which are sensitive by nature, so this document defines how photos are handled, stored (or not), and protected.

## 2. Data Classification

| Data | Sensitivity | Notes |
|---|---|---|
| Uploaded photo | High | Biometric-adjacent (face image) |
| Extracted landmarks | Medium | Derived from face, not directly identifying on its own but still sensitive |
| Face shape + confidence | Low | Classification label, not identifying |
| Hair texture / maintenance answers | Low | Self-reported preferences |
| Recommendation results / barber sheet | Low | Non-identifying output |

## 3. Photo Handling Principles

- SA-1: Uploaded photos are processed in memory / transient storage only, and deleted immediately after the pipeline completes (quality check → landmarks → classification), unless the user explicitly opts to save a result
- SA-2: No photo is stored longer than necessary to serve the immediate request
- SA-3: No photo or derived biometric data is used to train models without explicit, separate user consent — this is a distinct consent action from simply using the app
- SA-4: Photos are transmitted over encrypted connections (HTTPS/TLS) only

## 4. Access Control (V1 scope)

Since V1 has no user accounts, access control is minimal by design:

- SA-5: No authentication required to use the core flow (upload → recommend → barber sheet)
- SA-6: No cross-user data access exists, because no user data persists between sessions
- SA-7: If a "save this result" feature is added, it becomes the first point where any access control (e.g. session token, or lightweight account) is needed — flagged as a V1.1 decision, not V1 scope, unless confirmed otherwise (see Open Questions)

## 5. Infrastructure & Operational Security

- SA-8: API endpoints rate-limited to prevent abuse (e.g. repeated large photo uploads)
- SA-9: Uploaded file validation: type/size checks before processing (reject non-image files, oversized uploads)
- SA-10: Logs must not contain raw photo data or full landmark coordinates — logging should be limited to non-identifying operational data (timestamps, success/failure, latency)
- SA-11: LLM API calls (Explanation Engine) should not transmit the raw photo — only the derived, non-identifying feature set (face shape, texture, maintenance preference, style name) is sent to the LLM provider

## 6. Open Questions (blocking finalization of this document)

- **Are user accounts in scope for V1?** If yes, this document needs authentication, session management, and stored-data retention/deletion sections added. As currently scoped (stateless, no accounts), those sections are not applicable.
- **Is a "save your results" feature in V1?** If yes, some persistent storage of non-photo data (face shape, style choices) would need its own retention and deletion policy.
- **Which LLM provider will the Explanation Engine use in V1**, and what's their data retention policy for API inputs? This affects SA-11 and should be confirmed before implementation.
