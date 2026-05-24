import { useMemo } from 'react'
import { formatHMS, formatMs } from '../utils/time'
import styles from './LapList.module.css'

function pad(n) {
  return String(Math.floor(n)).padStart(2, '0')
}

export default function LapList({ laps }) {
  const times = useMemo(() => laps.map(l => l.t), [laps])
  const best = useMemo(() => Math.min(...times), [times])
  const worst = useMemo(() => Math.max(...times), [times])

  if (laps.length === 0) return null

  return (
    <div className={styles.section}>
      <div className={styles.heading}>
        <span>Lap Times</span>
        <span className={styles.badge}>{laps.length} laps</span>
      </div>
      <div className={styles.list}>
        {laps.map(lap => {
          const isBest = laps.length > 1 && lap.t === best
          const isWorst = laps.length > 1 && lap.t === worst
          return (
            <div key={lap.n} className={styles.row}>
              <span className={styles.num}>LAP {pad(lap.n)}</span>
              <span className={`${styles.time} ${isBest ? styles.best : isWorst ? styles.worst : ''}`}>
                {formatHMS(lap.t)}{formatMs(lap.t)}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
