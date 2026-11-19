# FactCheck — web frontend

The website for the FactCheck REST API (`src/factcheck_api`): a landing page, one workspace for
text, image, video and audio investigations, and an interactive report built from the API's real
JSON response. React 19 + TypeScript + Vite 7 + Tailwind CSS 4, with Framer Motion and Lucide icons.
No accounts, no history, nothing stored in the browser.

## Run it locally

You need Node.js 20+ and the backend running.

1. Start the backend (from the project root):

   ```bash
   .venv\Scripts\python.exe -m uvicorn factcheck_api.app:app --reload
   ```

2. Install and start the frontend (from `frontend/`):

   ```bash
   npm install
   ```
   ```bash
   npm run dev
   ```

3. Open **http://127.0.0.1:5173**.

The backend's default CORS origins already include `http://localhost:5173` and
`http://127.0.0.1:5173`. If the frontend runs elsewhere, add its origin to
`FACTCHECK_API_CORS_ORIGINS` in the backend's environment.

| Command | What it does |
|---|---|
| `npm run dev` | Development server with hot reload |
| `npm run build` | Type-check (`tsc -b`) and production build into `dist/` |
| `npm run preview` | Serve the production build on port 5173 |
| `npm run typecheck` | Type-check only |

## Configuration

Copy `.env.example` to `.env.local` to change any of these:

| Variable | Default | Meaning |
|---|---|---|
| `VITE_API_BASE_URL` | `http://127.0.0.1:8000` | Where the FastAPI backend runs |
| `VITE_MAX_IMAGE_MB` / `_VIDEO_MB` / `_AUDIO_MB` | 25 / 500 / 500 | Upload limits shown and checked before uploading. Keep them equal to the backend's `FACTCHECK_API_MAX_*_MB`; the backend enforces its own either way |
| `VITE_MAX_TEXT_CHARS` | 4000 | The fact-check engine's input limit |

## Structure

```
src/
  lib/          config (env), types (mirror of FactCheckResponse), api (the only module that calls the
                backend), verdict styles, formatting
  hooks/        useInvestigation (request state, lives in App), useRoute (#/ and #/investigate),
                useHealth (GET /health), useObjectUrl (local previews)
  components/
    landing/    HeroSection, HowItWorks, LandingPage
    workspace/  ModalitySelector, TextInput, FileUpload, VerifyButton, InvestigationLoader,
                ErrorState, BackendStatus, InvestigationWorkspace
    report/     InvestigationReport, ClaimCard, VerdictBadge, EvidenceScore, SourcesList,
                MediaEvidence (OCR overlay, video/audio timeline, transcript)
    fx/         TextFx (scramble decode, mask rise, blur-in, rotating word, marker, count-up, typewriter,
                strike & replace), LoaderFx (indeterminate bar, typing dots, equalizer, dot wave)
  lib/pdfReport.ts  the downloadable PDF report
    visual/     OceanBackground (canvas particles), Sonar, BrandMark
    layout/     Navbar, Footer
    ui/         Toast
```

## What the report shows (all from the API response)

- **Summary:** the overall label and sentence, verdict counts with a distribution bar, the number of
  claims checked and the server's processing time.
- **Claim cards:** verdict, evidence gauge (or support/against gauges for CONFLICTING), the CLI's
  explanation, correction, the exact source excerpt, citations grouped as supporting / contradicting /
  context with clickable sources, and validation details (rejected citations, sources considered,
  notes). Empty fields are hidden.
- **Image:** the uploaded image with the OCR line boxes drawn on top (`report.extraction.quality.lines`).
  Hovering a claim highlights the lines it was read from.
- **Video / audio:** a player for the uploaded file, a timeline of every claim's spoken or on-screen
  appearances coloured by verdict, the timed transcript (and on-screen text for video) with recognition
  confidence. Clicking a timestamp plays from there.
- Claims over the per-run limit and claims removed by validation, warnings and notes.
- **Download PDF report** builds an A4 report in the browser from the same response
  (`src/lib/pdfReport.ts`, jsPDF loaded only on click). It uses the app's Deep Ocean look: dark pages
  with a soft teal glow, the brand mark in a running header, glass-style cards and the verdict colours.
  It contains:
  - a cover with the sonar motif, the overall result, the verdict bar and legend, stat cards (claims
    checked, processing time, sources cited, input), details (file, date, AI model) and warnings;
  - every claim as a card: verdict badge, evidence meter (support/against for CONFLICTING), explanation,
    correction, source excerpt, timestamps or OCR lines, source cards with clickable links grouped by
    role, and validation details;
  - claims not verified or removed, with reasons;
  - "How to read this report" notes;
  - an appendix with the full transcript, on-screen text or OCR text, with recognition confidence;
  - page numbers.

  The built-in PDF fonts cover Latin-1 only, so other scripts and symbols are replaced, for
  example → becomes "->".

## Design tokens

All colours and type sizes come from the `@theme` block in `src/index.css`, so the UI stays uniform:

- surfaces `abyss`, `deep`, `glass`; accent `teal` (plus `teal-bright`, `teal-deep`); text `ice`, `muted`;
- verdicts `supported`, `refuted`, `warn`, `conflict`, `neutral`, `failed`;
- sizes `text-micro` (labels), `text-lead`, and headings `h-page`, `h-section`, `h-card`;
- shared components `.glass`, `.panel`, `.hud`, `.btn`, `.chip`, `.field`, `.dropzone`.

Use the Tailwind classes (`text-refuted`, `bg-teal/10`, …) rather than raw hex values. SVG attributes and
the PDF read the same colours from `src/lib/palette.ts`; keep the two in sync.

## Demo sign-in

The workspace (`#/investigate`) requires signing in. Opening it while signed out leads to the login page
(`#/login`). The demo credentials are:

| User ID | Password |
|---|---|
| `admin` | `admin123` |

Change them with `VITE_DEMO_USER` / `VITE_DEMO_PASSWORD` in `.env.local`. There's no database or
backend call: the check runs in the browser (`src/lib/auth.ts`), and the session is kept in
`sessionStorage`, so it ends when the tab closes. The navbar shows the signed-in user and a sign-out
button.

> **This is a demo gate, not security.** The credentials are part of the JavaScript bundle, and the API
> itself has no login, so anyone with technical knowledge can bypass it. Use a real authentication
> service before deploying publicly.

## Smooth scrolling (Lenis) and motion

- [Lenis](https://github.com/darkroomengineering/lenis) (`src/lib/smoothScroll.ts`) gives the whole
  page eased, momentum-style scrolling. Anchor jumps ("How it works") and page changes go through the
  same easing.
- Inner scroll areas (transcripts, on-screen text, OCR lines) use `data-lenis-prevent`, so they scroll
  on their own.
- Scroll-linked effects: hero parallax (the text drifts up and fades, the sonar sinks and tilts) and a
  teal scroll-progress line at the top of the page.
- Page changes fade in with a slight rise and blur.
- With reduced motion turned on in the system, Lenis is off and normal scrolling is used, the
  parallax stops, and the background shows a still frame.

## Background video

`public/media/ocean-loop.mp4` (12-second seamless loop, 960×540, H.264, about 330 KB) and its poster
`ocean-poster.jpg` are **original**. They're rendered procedurally by `scripts/make_ocean_video.py`
(drifting light caustics, light shafts, rising particles), so there are no licensing questions. It
plays muted and looped behind a dark tint, under the canvas particles and the perspective grid.

To regenerate it:

```bash
..\.venv\Scripts\python.exe scripts\make_ocean_video.py
```

To use your own footage instead, replace those two files, keeping the names. A dark clip under about
5 MB works best.

## Loading screen: liquid fill with real checkpoints

The loader is a glass orb that fills with liquid as the investigation moves through four checkpoints
(`src/lib/progress.ts`, `src/components/fx/LiquidFill.tsx`, `Checkpoints.tsx`). Every checkpoint is
an event the browser actually observes. Requests use `XMLHttpRequest`, which reports upload and
download bytes.

| Checkpoint | Source | Fill |
|---|---|---|
| Uploading evidence / Text submitted | Real upload bytes (percentage shown) | 6 → 24% |
| Analysing on the server | Request fully sent; the server reports no progress while it works | 24 → 88%, **estimated** |
| Receiving the report | Real download bytes | 88 → 97% |
| Report ready | Response received | 100%, then the report appears |

Inside "Analysing", the level eases toward the next checkpoint over the usual processing time. That
time is scaled by the media's length for video and audio, and the level never reaches the checkpoint
until the response really arrives. The checkpoint is tagged **est. fill**, and the loader explains
this in plain words.

## Upload constraints (per mode)

`src/lib/fileCheck.ts` checks every file before it is uploaded. The backend validates everything
again.

1. **Picker filter.** The file dialog's `accept` is that mode's MIME family plus its extensions (for
   example `video/*,.mp4,...`), so other kinds of files aren't offered.
2. **Extension allow-list.** The same lists as the backend.
3. **Declared type.** An `image/*` file is refused in Video mode, and so on for each mode.
4. **Real content.** The first bytes are read and identified: PNG, JPEG, GIF, WebP, MP4/MOV
   (`ftyp`), M4A, Matroska/WebM, Ogg, ASF, AVI, MPEG, FLV, MP3, WAV, FLAC, AIFF, AMR, CAF, AC3, PDF
   and ZIP. A photo renamed to `.mp4`, or an MP3 renamed to `.mp4`, is rejected with "This file is
   an image/audio file, not a video. Use Image/Audio mode for it."
5. **Size and length.** At most 25/500/500 MB. Media length is read by the browser before upload:
   video at most 10 minutes, audio at most 60 minutes, the backend defaults.

## Design system

`src/index.css` defines one set of surfaces and controls, used everywhere for a uniform look:
- `.glass`: frosted card with a gradient light-edge border.
- `.glass-strong`: the selected or active state.
- `.panel`: a major section, 24 px radius, with a top light edge. `.hud` adds corner brackets.
- `.btn` with `.btn-primary` / `.btn-glass` / `.btn-ghost`, sizes `.btn-sm` / `.btn-lg`, and
  `.btn-danger` for destructive hover.
- `.chip`, `.field` (inputs) and `.dropzone`.

The background adds a slowly drifting perspective grid under the particles.

## Honest by design

- The loader's checkpoints are real browser events. The only estimated part, the fill during server
  analysis, is labelled as an estimate, and the server-side step names describe the pipeline rather
  than track it.
- The decorative "evidence fragments" on the landing page contain only redacted bars, never text or
  numbers.
- The evidence score is labelled as an experimental rubric, not a probability. Recognition confidence
  is labelled as recognition, not truth.
- INSUFFICIENT_EVIDENCE and similar verdicts are results, not errors. Real failures (server offline,
  timeout, invalid file, missing keys) get their own error panel with a retry option where it makes
  sense. Your input is kept.

## Notes and limitations

- **Long requests.** Video and audio requests run in a single HTTP request; the browser waits up to
  45 minutes. Leaving the workspace doesn't cancel the request, and "Stop waiting" only stops the
  browser waiting (the server finishes the job anyway). The backend processes one video or audio job
  at a time.
- **Windows Smart App Control.** On this machine it blocks Rollup's native module, so
  `package.json` overrides `rollup` with the official WebAssembly build (`@rollup/wasm-node`). It
  produces the same output, just a little slower to build.
- **Fonts.** Fraunces, Manrope and JetBrains Mono load from Google Fonts. Offline, the page falls back
  to system fonts.
- **Reduced motion.** With reduced motion turned on in the system, the sonar, particles and
  transitions stay still.
