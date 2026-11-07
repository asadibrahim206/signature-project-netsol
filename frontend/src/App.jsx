import { useState } from "react";
import Header from "./components/Header.jsx";
import DetectionPage from "./pages/DetectionPage.jsx";
import VerificationPage from "./pages/VerificationPage.jsx";

export default function App() {
  const [mode, setMode] = useState("detect"); // "detect" | "verify"

  return (
    <div className="app">
      <Header mode={mode} onChange={setMode} />
      <main className="main" id={`panel-${mode}`} role="tabpanel">
        {mode === "detect" ? <DetectionPage /> : <VerificationPage />}
      </main>
      <footer className="footer">
        Results come from a trained model and should be reviewed by a person
        before any decision is made.
      </footer>
    </div>
  );
}
