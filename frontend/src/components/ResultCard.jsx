/**
 * tone: "ok" | "bad" | "neutral"
 * confidence: number between 0 and 1 (optional)
 */
export default function ResultCard({ tone, title, detail, confidence }) {
  const pct =
    typeof confidence === "number" ? Math.round(confidence * 100) : null;

  return (
    <div className={`result result-${tone}`} role="status" aria-live="polite">
      <div className="result-icon" aria-hidden="true">
        {tone === "ok" && (
          <svg width="22" height="22" viewBox="0 0 24 24">
            <path d="M5 12.5l4.5 4.5L19 7.5" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        )}
        {tone === "bad" && (
          <svg width="22" height="22" viewBox="0 0 24 24">
            <path d="M6 6l12 12M18 6L6 18" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" />
          </svg>
        )}
        {tone === "neutral" && (
          <svg width="22" height="22" viewBox="0 0 24 24">
            <path d="M6 12h12" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" />
          </svg>
        )}
      </div>
      <div className="result-body">
        <p className="result-title">{title}</p>
        {detail && <p className="result-detail">{detail}</p>}
        {pct !== null && (
          <div className="conf">
            <div className="conf-row">
              <span>Model confidence</span>
              <strong>{pct}%</strong>
            </div>
            <div className="conf-track">
              <div className="conf-fill" style={{ width: `${pct}%` }} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
