import { useState } from 'react'
import Stopwatch from './components/Stopwatch'
import Timer from './components/Timer'
import styles from './App.module.css'

export default function App() {
  const [tab, setTab] = useState('stopwatch')

  return (
    <div className={styles.app}>
      <header className={styles.header}>
        <h1 className={styles.title}>Chronos</h1>
        <p className={styles.subtitle}>Precision Time Instrument</p>
      </header>

      <div className={styles.tabs}>
        <button
          className={`${styles.tab} ${tab === 'stopwatch' ? styles.active : ''}`}
          onClick={() => setTab('stopwatch')}
        >
          Stopwatch
        </button>
        <button
          className={`${styles.tab} ${tab === 'timer' ? styles.activeTimer : ''}`}
          onClick={() => setTab('timer')}
        >
          Timer
        </button>
      </div>

      <main className={styles.main}>
        {tab === 'stopwatch' ? <Stopwatch /> : <Timer />}
      </main>
    </div>
  )
}
