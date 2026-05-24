import styles from './Controls.module.css'

export default function Controls({ buttons }) {
  return (
    <div className={styles.controls}>
      {buttons.map(({ label, onClick, variant = 'secondary', disabled = false }) => (
        <button
          key={label}
          className={`${styles.btn} ${styles[variant]}`}
          onClick={onClick}
          disabled={disabled}
        >
          {label}
        </button>
      ))}
    </div>
  )
}
