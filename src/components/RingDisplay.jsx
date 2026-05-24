import styles from './RingDisplay.module.css'

const CIRC = 2 * Math.PI * 108 // r=108

export default function RingDisplay({
  mainText,
  msText,
  label,
  running,
  progress = 0,
  variant = 'stopwatch',
}) {
  const offset = CIRC * (1 - Math.min(Math.max(progress, 0), 1))

  return (
    <div className={styles.ring}>
      <svg className={styles.svg} viewBox="0 0 240 240">
        <circle className={styles.bg} cx="120" cy="120" r="108" />
        <circle
          className={`${styles.progress} ${variant === 'timer' ? styles.timer : styles.stopwatch}`}
          cx="120"
          cy="120"
          r="108"
          strokeDasharray={CIRC}
          strokeDashoffset={offset}
        />
      </svg>
      <div className={styles.inner}>
        <div className={styles.main}>{mainText}</div>
        <div className={`${styles.ms} ${variant === 'timer' ? styles.msTimer : ''}`}>
          {msText}
        </div>
        <div className={styles.label}>
          <span className={`${styles.dot} ${running ? styles.dotRunning : ''}`} />
          {label}
        </div>
      </div>
    </div>
  )
}
