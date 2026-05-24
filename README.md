# Chronos — Stopwatch & Timer

A precision time instrument built with React + Vite.

## Features

- **Stopwatch** — Start, pause, resume, reset, lap tracking with best/worst highlighting
- **Timer** — Custom input (h/m/s), quick presets, countdown with animated ring
- Animated SVG progress ring for both modes
- Retro-futuristic dark UI with `Share Tech Mono` digits

## Tech Stack

- React 18 + Vite 5
- CSS Modules (zero external CSS libraries)
- `requestAnimationFrame` for smooth 60fps updates
- Custom hooks: `useStopwatch`, `useTimer`

## Getting Started

```bash
npm install
npm run dev
```

Then open [http://localhost:5173](http://localhost:5173).

## Build

```bash
npm run build
npm run preview
```

## Deploy

Works out of the box with **Netlify**, **Vercel**, or **GitHub Pages** (via `vite build`).
