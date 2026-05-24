export function pad(n, len = 2) {
  return String(Math.floor(Math.abs(n))).padStart(len, '0')
}

export function formatHMS(ms) {
  const total = Math.max(ms, 0)
  const h = Math.floor(total / 3_600_000)
  const m = Math.floor((total % 3_600_000) / 60_000)
  const s = Math.floor((total % 60_000) / 1_000)
  return `${pad(h)}:${pad(m)}:${pad(s)}`
}

export function formatMs(ms) {
  return '.' + pad(Math.floor((Math.max(ms, 0) % 1_000) / 10))
}

export function inputsToMs(h, m, s) {
  return ((Number(h) || 0) * 3600 + (Number(m) || 0) * 60 + (Number(s) || 0)) * 1000
}
