import { useState, useEffect } from 'react'
import { useTimer } from '../hooks/useTimer'
import { formatHMS, formatMs, inputsToMs } from '../utils/time'
import RingDisplay from './RingDisplay'
import Controls from './Controls'
import styles from './Panel.module.css'
import timerStyles from './Timer.module.css'

const PRESETS = [
  { label: '1 min', m: 1 },
  { label: '3 min', m: 3 },
  { label: '5 min', m: 5 },
  { label: '10 min', m: 10 },
  { label: '25 min', m: 25 },
]

export default function Timer() {
  const [hours, setHours] = useState(0)
  const [minutes, setMinutes] = useState(5)
  const [seconds, setSeconds] = useState(0)

  const initialMs = inputsToMs(hours, minutes, seconds)
  const timer = useTimer(initialMs)

  // sync input changes to timer when not running
  useEffect(() => {
    if (!timer.running) {
      const ms = inputsToMs(hours, minutes, seconds)
      timer.setTotal(ms)
    }
  }, [hours, minutes, seconds])

  const handlePreset = (m) => {
    setHours(0)
    setMinutes(m)
    setSeconds(0)
    const ms = inputsToMs(0, m, 0)
    timer.reset(ms)
  }

  const handleToggle = () => {
    if (timer.expired) {
      timer.reset()
      return
    }
    if (timer.running) {
      timer.pause()
    } else {
      const ms = inputsToMs(hours, minutes, seconds)
      timer.start(timer.remaining > 0 ? undefined : ms)
    }
  }

  const handleReset = () => {
    const ms = inputsToMs(hours, minutes, seconds)
    timer.reset(ms)
  }

  const progress = timer.total.current > 0
    ? timer.remaining / timer.total.current
    : 0

  const startLabel = timer.expired
    ? 'Restart'
    : timer.running
    ? 'Pause'
    : timer.remaining < timer.total.current && timer.remaining > 0
    ? 'Resume'
    : 'Start'

  const controls = [
    {
      label: startLabel,
      onClick: handleToggle,
      variant: timer.running ? 'pause' : 'primaryTimer',
      disabled: !timer.running && inputsToMs(hours, minutes, seconds) === 0 && !timer.expired,
    },
    {
      label: 'Reset',
      onClick: handleReset,
      variant: 'secondary',
    },
  ]

  return (
    <div className={styles.panel}>
      <RingDisplay
        mainText={formatHMS(timer.remaining)}
        msText={formatMs(timer.remaining)}
        label="Timer"
        running={timer.running}
        progress={progress}
        variant="timer"
      />

      {timer.expired && (
        <div className={timerStyles.expired}>⏰ Time&apos;s Up!</div>
      )}

      {!timer.running && (
        <>
          <div className={timerStyles.inputSection}>
            <div className={timerStyles.inputLabel}>Set Duration</div>
            <div className={timerStyles.inputs}>
              {[
                { label: 'Hours', value: hours, setter: setHours, max: 99 },
                { label: 'Minutes', value: minutes, setter: setMinutes, max: 59 },
                { label: 'Seconds', value: seconds, setter: setSeconds, max: 59 },
              ].map(({ label, value, setter, max }) => (
                <div key={label} className={timerStyles.inputWrap}>
                  <input
                    className={timerStyles.field}
                    type="number"
                    min={0}
                    max={max}
                    value={value}
                    onChange={e => setter(Math.min(max, Math.max(0, parseInt(e.target.value) || 0)))}
                  />
                  <span className={timerStyles.fieldLabel}>{label}</span>
                </div>
              ))}
            </div>
            <div className={timerStyles.presets}>
              {PRESETS.map(p => (
                <button
                  key={p.label}
                  className={timerStyles.preset}
                  onClick={() => handlePreset(p.m)}
                >
                  {p.label}
                </button>
              ))}
            </div>
          </div>
          <div className={timerStyles.divider} />
        </>
      )}

      <Controls buttons={controls} />
    </div>
  )
}
