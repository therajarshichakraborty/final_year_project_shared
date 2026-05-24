import { useStopwatch } from '../hooks/useStopwatch'
import { formatHMS, formatMs } from '../utils/time'
import RingDisplay from './RingDisplay'
import Controls from './Controls'
import LapList from './LapList'
import styles from './Panel.module.css'

export default function Stopwatch() {
  const { elapsed, running, laps, start, pause, reset, lap } = useStopwatch()

  // Ring progress: animate through seconds within a minute
  const progress = (elapsed / 1000 % 60) / 60

  const controls = [
    {
      label: running ? 'Pause' : elapsed > 0 ? 'Resume' : 'Start',
      onClick: running ? pause : start,
      variant: running ? 'pause' : 'primary',
    },
    {
      label: 'Lap',
      onClick: lap,
      variant: 'secondary',
      disabled: !running,
    },
    {
      label: 'Reset',
      onClick: reset,
      variant: 'secondary',
    },
  ]

  return (
    <div className={styles.panel}>
      <RingDisplay
        mainText={formatHMS(elapsed)}
        msText={formatMs(elapsed)}
        label="Stopwatch"
        running={running}
        progress={progress}
        variant="stopwatch"
      />
      <Controls buttons={controls} />
      <LapList laps={laps} />
    </div>
  )
}
