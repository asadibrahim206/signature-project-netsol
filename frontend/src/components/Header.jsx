const TABS = [
  { id: "detect", label: "Signature detection" },
  { id: "verify", label: "Signature verification" },
];

export default function Header({ mode, onChange }) {
  return (
    <header className="header">
      <div className="header-inner">
        <div className="brand">
          <svg width="30" height="30" viewBox="0 0 64 64" aria-hidden="true">
            <rect width="64" height="64" rx="14" fill="#1D3FBB" />
            <path
              d="M12 40c6-14 10-20 13-20s-1 20 4 20 6-12 10-12 2 8 7 8"
              fill="none"
              stroke="#fff"
              strokeWidth="4"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          <span className="brand-name">SignatureLens</span>
        </div>

        <nav className="tabs" role="tablist" aria-label="Choose a tool">
          {TABS.map((t) => (
            <button
              key={t.id}
              role="tab"
              aria-selected={mode === t.id}
              aria-controls={`panel-${t.id}`}
              className={`tab ${mode === t.id ? "is-active" : ""}`}
              onClick={() => onChange(t.id)}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </div>
    </header>
  );
}
