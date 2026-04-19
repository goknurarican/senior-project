# NeuroCart — Measuring the Effect of UX Friction on Consumer Behavior via EEG & Eye Tracking

> **Graduation Thesis** · Industrial Engineering / Management Information Systems  
> Göknur Arıcan · 2025–2026

[![Next.js](https://img.shields.io/badge/Next.js-14-black?logo=next.js)](https://nextjs.org)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![SQLite](https://img.shields.io/badge/SQLite-WAL-003B57?logo=sqlite)](https://sqlite.org)
[![EEG](https://img.shields.io/badge/EEG-BrainVision-blueviolet)](https://brainproducts.com)
[![Eye Tracking](https://img.shields.io/badge/Eye%20Tracking-Gazepoint-teal)](https://gazept.com)
[![LSL](https://img.shields.io/badge/Sync-Lab%20Streaming%20Layer-orange)](https://labstreaminglayer.org)

---

## Overview

**NeuroCart** is a controlled laboratory experiment platform designed to investigate how UX friction (deliberately induced usability failures) affects consumer purchasing decisions — measured through simultaneous **EEG** and **eye tracking** recordings.

Participants browse and shop on a realistic e-commerce storefront. A subset of sessions are injected with friction scenarios (image load failures, button delays, payment retries, etc.) at configurable probabilities. The lab pipeline synchronizes behavioral clickstream data with millisecond-accurate physiological signals, enabling event-locked EEG analysis (ERPs) and gaze-based attention mapping.

### Research Questions

1. Do UX friction events elicit measurable EEG responses (e.g., frustration-related frontal alpha asymmetry, P300 novelty responses)?
2. Does friction-induced cognitive load alter gaze patterns — fixation duration, saccade rate, pupil dilation?
3. Which friction types most strongly predict cart abandonment and checkout drop-off?

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Lab Environment                      │
│                                                         │
│  ┌──────────────┐    ┌───────────────┐                  │
│  │  BrainVision │    │   Gazepoint   │                  │
│  │   Recorder   │    │  Eye Tracker  │                  │
│  │  (64-ch EEG) │    │  (150 Hz)     │                  │
│  └──────┬───────┘    └──────┬────────┘                  │
│         │ parallel port      │ TCP socket                │
│         ▼                    ▼                           │
│  ┌──────────────────────────────────────┐               │
│  │          trigger_server.py           │               │
│  │  • Sends EEG markers via inpoutx64   │               │
│  │  • Streams Gazepoint data via LSL    │               │
│  │  • REST bridge → Next.js platform   │               │
│  └──────────────────┬───────────────────┘               │
│                     │                                    │
│  ┌──────────────────▼───────────────────┐               │
│  │       Next.js E-Commerce Platform    │               │
│  │  • Realistic storefront (products,   │               │
│  │    cart, checkout, search)           │               │
│  │  • injection-sdk.js — client-side    │               │
│  │    scenario engine                   │               │
│  │  • 4 experiment groups (A/B/C/ctrl)  │               │
│  │  • Real-time admin dashboard         │               │
│  └──────────────────┬───────────────────┘               │
│                     │                                    │
│  ┌──────────────────▼───────────────────┐               │
│  │           experiment.db (SQLite)     │               │
│  │  sessions · events · eye_data        │               │
│  │  scenario_triggers · eeg_markers     │               │
│  └──────────────────────────────────────┘               │
│                                                         │
│  ┌──────────────────────────────────────┐               │
│  │        lab_panel.py (Tkinter)        │               │
│  │  One-click subject packaging →       │               │
│  │  CSV exports + Google Drive backup   │               │
│  └──────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14 · React 18 · TypeScript · Tailwind CSS |
| Backend | Next.js API Routes · SQLite (WAL mode) |
| EEG Interface | Python · `inpoutx64` parallel port driver · BrainVision Recorder |
| Eye Tracking | Python · Gazepoint TCP API · 150 Hz raw gaze stream |
| Physiological Sync | Lab Streaming Layer (LSL) · `pylsl` |
| Scenario Engine | Vanilla JS injection SDK (`public/injection-sdk.js`) |
| Data Export | Python · CSV · Google Drive API |
| Lab UI | Python · Tkinter |

---

## Experiment Design

Participants are randomly assigned to one of four groups on first visit. Assignment is deterministic (session-cookie-based) so the experience is consistent across pages.

| Group | Traffic Share | Friction Probability | Description |
|---|---|---|---|
| **Control** | 25 % | 0 % | Clean experience — no scenarios |
| **Variant A** | 25 % | ~30 % | Low-friction exposure |
| **Variant B** | 25 % | ~60 % | Medium-friction exposure |
| **Variant C** | 25 % | ~90 % | High-friction exposure |

**Guardrails** — max 2 scenarios per session · 120 s cooldown between triggers · payment step is never permanently blocked · all overlays are dismissible (WCAG-compliant).

### Friction Scenario Catalog (16 types)

| Category | Scenario ID | Description |
|---|---|---|
| Visual | `slow_image` | Product images load with artificial delay |
| Visual | `broken_image` | Placeholder shown instead of product image |
| Visual | `skeleton_prolong` | Skeleton loaders persist longer than expected |
| Interaction | `button_delay` | Add-to-cart button has a 2 s lag |
| Interaction | `first_click_miss` | First button click is swallowed silently |
| Interaction | `feedback_late` | Toast notifications appear after a delay |
| Navigation | `search_irrelevant` | Search returns temporarily shuffled results |
| Navigation | `facet_reset_once` | Active filters reset on next interaction |
| Navigation | `sort_reset` | Sort order reverts to default |
| Pricing | `price_change` | Price-increase warning shown at checkout |
| Coupon | `coupon_min_spend` | Coupon rejected with minimum-spend error |
| Coupon | `coupon_expired` | Coupon shown as expired mid-checkout |
| Payment | `3ds_soft_fail` | First 3DS authentication attempt fails |
| Payment | `payment_retry_timeout` | Payment times out, then succeeds on retry |
| Overlay | `overlay_blocking` | Dismissible modal interrupts browsing |
| Network | `network_jitter` | Simulated latency spikes on API calls |

---

## EEG & Gaze Signal Pipeline

```
Browser event (e.g. scenario_trigger)
        │
        ▼
POST /send_trigger  →  trigger_server.py
        │
        ├─► parallel port OUT  →  BrainVision (hardware marker, ~1 ms accuracy)
        ├─► LSL push_sample    →  StreamOutlet "ExperimentMarkers"
        └─► SQLite INSERT      →  eeg_markers table (wall_time_ms + session_id)

Gazepoint (150 Hz TCP stream)
        │
        ▼
trigger_server.py  →  buffered INSERT (75-row batches, 500 ms flush)
        └─► SQLite eye_data table (gaze_x, gaze_y, pupil_left, pupil_right, bpogv)
```

Post-session, `package_subject.py` exports per-subject CSV bundles:

- `eeg_markers.csv` — marker label, wall time, session-relative offset
- `eye_data.csv` — full 150 Hz gaze log with pupil dilation
- `events.csv` — complete behavioral clickstream
- `scenario_triggers.csv` — scenario onset/offset with active group

---

## Repository Structure

```
.
├── pages/                  # Next.js pages & API routes
│   ├── api/                # REST endpoints
│   │   ├── admin/          # Admin-only endpoints (stats, scenario CRUD)
│   │   ├── events/         # Behavioral event ingestion
│   │   ├── scenarios/      # Active-scenario serving
│   │   └── session/        # Session creation & group assignment
│   ├── admin/              # Admin dashboard (experiments, sessions, scenarios)
│   ├── product/            # Product detail pages
│   ├── index.tsx           # Storefront homepage
│   ├── products.tsx        # Product listing
│   ├── cart.tsx            # Shopping cart
│   └── checkout.tsx        # Checkout & payment flow
├── components/             # Shared React components
├── public/
│   └── injection-sdk.js    # Client-side scenario engine
├── lib/                    # DB helpers & utilities
├── trigger_server.py       # EEG marker + Gazepoint bridge (Flask)
├── data_logger.py          # SQLite writer with WAL + write-buffering
├── lab_panel.py            # Technician GUI (subject packaging)
├── package_subject.py      # CSV export & Drive backup per subject
├── backup_drive.py         # Google Drive uploader
├── diagnose.py             # System diagnostics
└── experiment.db           # SQLite database (auto-created)
```

---

## Lab Setup & Running an Experiment

### Prerequisites

- Node.js ≥ 18
- Python ≥ 3.10 with: `flask flask-cors pylsl google-auth google-api-python-client`
- BrainVision Recorder + `inpoutx64.dll` for parallel-port triggers
- Gazepoint Control running and streaming on `127.0.0.1:4242`

### 1. Install dependencies

```bash
npm install
pip install flask flask-cors pylsl google-auth google-api-python-client
```

### 2. Start the experiment platform

```bash
npm run dev                  # Next.js → http://localhost:3000
python trigger_server.py     # EEG + gaze bridge
```

### 3. Run a participant session

1. Calibrate Gazepoint, start gaze recording
2. Start BrainVision recording
3. Seat participant at `http://localhost:3000`
4. Monitor live in Admin → Sessions (`http://localhost:3000/admin`)

### 4. Package subject data

```
SONRAKI_DENEK.bat    # opens lab_panel.py
```

The panel exports all CSVs for the most recent session and optionally backs up to Google Drive.

### Admin Credentials (dev/lab only)

| Role | Email | Password |
|---|---|---|
| Admin | admin@test.com | admin123 |
| Participant | user@test.com | user123 |

---

## Data Schema

| Table | Description |
|---|---|
| `sessions` | Participant sessions, experiment group, timestamps |
| `events` | Full behavioral clickstream (page views, clicks, scroll, cart) |
| `eye_data` | 150 Hz Gazepoint stream (gaze XY, pupil size, validity flags) |
| `eeg_markers` | Hardware trigger log with wall-clock and session-relative time |
| `scenario_triggers` | Scenario onset/offset, type, active group |
| `experiments` | Experiment run definitions |
| `scenarios` | Scenario configurations with parameters |
| `products` · `cart_items` | Storefront data |

---

## Key Design Decisions

**WAL mode on SQLite** — Gazepoint streams at 150 Hz; WAL allows concurrent reads from Next.js while Python writes without SQLITE_BUSY errors.

**Write buffering for eye data** — Samples are batched in a 75-row in-memory buffer flushed every 500 ms, reducing commit frequency from 150/s to 2/s.

**Deterministic group assignment** — Experiment group is derived from the session cookie UUID, ensuring a participant sees the same friction level across all pages in a session.

**Injection SDK isolation** — The client-side scenario engine (`injection-sdk.js`) is entirely self-contained; it polls `/api/scenarios/active` independently and applies DOM manipulations without touching application logic.

**Parallel port markers** — Hardware EEG markers are sent via `inpoutx64.dll` at the trigger moment (~1 ms jitter), enabling precise ERP epoch locking regardless of network latency.

---

## License

This project was developed as a graduation thesis and is shared for academic reference. Please cite appropriately if you build on this work.
