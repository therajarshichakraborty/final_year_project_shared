import { useState, useRef, useCallback } from 'react'

export function useTimer(initialMs) {
  const [remaining, setRemaining] = useState(initialMs)
  const [running, setRunning] = useState(false)
  const [expired, setExpired] = useState(false)

  const rafRef = useRef(null)
  const startRef = useRef(0)
  const remainingRef = useRef(initialMs)
  const totalRef = useRef(initialMs)

  const tick = useCallback(() => {
    const elapsed = performance.now() - startRef.current
    const left = remainingRef.current - elapsed
    if (left <= 0) {
      setRemaining(0)
      setRunning(false)
      setExpired(true)
      return
    }
    setRemaining(left)
    rafRef.current = requestAnimationFrame(tick)
  }, [])

  const start = useCallback((ms) => {
    if (ms !== undefined) {
      remainingRef.current = ms
      totalRef.current = ms
      setRemaining(ms)
    }
    startRef.current = performance.now()
    setRunning(true)
    setExpired(false)
    rafRef.current = requestAnimationFrame(tick)
  }, [tick])

  const pause = useCallback(() => {
    cancelAnimationFrame(rafRef.current)
    remainingRef.current -= performance.now() - startRef.current
    setRunning(false)
  }, [])

  const reset = useCallback((ms) => {
    cancelAnimationFrame(rafRef.current)
    const val = ms ?? totalRef.current
    remainingRef.current = val
    totalRef.current = val
    setRemaining(val)
    setRunning(false)
    setExpired(false)
  }, [])

  const setTotal = useCallback((ms) => {
    totalRef.current = ms
    remainingRef.current = ms
    setRemaining(ms)
    setExpired(false)
  }, [])

  return {
    remaining,
    total: totalRef,
    running,
    expired,
    start,
    pause,
    reset,
    setTotal,
  }
}
