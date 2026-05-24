import { useState, useRef, useCallback } from 'react'

export function useStopwatch() {
  const [elapsed, setElapsed] = useState(0)
  const [running, setRunning] = useState(false)
  const [laps, setLaps] = useState([])

  const rafRef = useRef(null)
  const startRef = useRef(0)
  const baseRef = useRef(0)
  const lapBaseRef = useRef(0)

  const tick = useCallback(() => {
    const now = performance.now()
    const total = baseRef.current + (now - startRef.current)
    setElapsed(total)
    rafRef.current = requestAnimationFrame(tick)
  }, [])

  const start = useCallback(() => {
    startRef.current = performance.now()
    setRunning(true)
    rafRef.current = requestAnimationFrame(tick)
  }, [tick])

  const pause = useCallback(() => {
    cancelAnimationFrame(rafRef.current)
    baseRef.current += performance.now() - startRef.current
    setRunning(false)
  }, [])

  const reset = useCallback(() => {
    cancelAnimationFrame(rafRef.current)
    baseRef.current = 0
    lapBaseRef.current = 0
    setElapsed(0)
    setRunning(false)
    setLaps([])
  }, [])

  const lap = useCallback(() => {
    const now = performance.now()
    const total = baseRef.current + (now - startRef.current)
    const lapTime = total - lapBaseRef.current
    lapBaseRef.current = total
    setLaps(prev => [{ n: prev.length + 1, t: lapTime }, ...prev])
  }, [])

  return { elapsed, running, laps, start, pause, reset, lap }
}
