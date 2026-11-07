import { useState } from "react";
import ImageUpload from "../components/ImageUpload.jsx";
import ResultCard from "../components/ResultCard.jsx";
import Spinner from "../components/Spinner.jsx";
import { verifySignature } from "../services/api.js";

export default function VerificationPage() {
  const [original, setOriginal] = useState(null);
  const [check, setCheck] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | loading | done | error
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  function clearResult() {
    setResult(null);
    setError("");
    setStatus("idle");
  }

  function reset() {
    setOriginal(null);
    setCheck(null);
    clearResult();
  }

  async function run() {
    if (!original || !check) return;
    setStatus("loading");
    setError("");
    setResult(null);
    try {
      const data = await verifySignature(original, check);
      setResult(data);
      setStatus("done");
    } catch (e) {
      setError(e.message || "Something went wrong. Try again.");
      setStatus("error");
    }
  }

  const loading = status === "loading";
  const isOriginal = result?.verdict === "original";

  return (
    <div className="page">
      <div className="page-head">
        <h1>Check if a signature is original or forged</h1>
        <p>
          Upload a signature you know is genuine, then the signature you want to
          check. The model will compare them.
        </p>
      </div>

      <div className="stack">
        <div className="grid-2">
          <ImageUpload
            id="verify-original"
            label="Original signature"
            hint="The genuine reference signature"
            file={original}
            onSelect={(f) => {
              setOriginal(f);
              clearResult();
            }}
            onClear={() => {
              setOriginal(null);
              clearResult();
            }}
            disabled={loading}
          />
          <ImageUpload
            id="verify-check"
            label="Signature to check"
            hint="The signature you want to verify"
            file={check}
            onSelect={(f) => {
              setCheck(f);
              clearResult();
            }}
            onClear={() => {
              setCheck(null);
              clearResult();
            }}
            disabled={loading}
          />
        </div>

        <div className="actions">
          <button
            type="button"
            className="btn btn-primary"
            onClick={run}
            disabled={!original || !check || loading}
          >
            {loading && <Spinner />}
            {loading ? "Comparing…" : "Verify signature"}
          </button>
          {(original || check || result) && !loading && (
            <button type="button" className="btn btn-ghost" onClick={reset}>
              Start over
            </button>
          )}
        </div>

        {status === "done" && result && (
          <ResultCard
            tone={isOriginal ? "ok" : "bad"}
            title={isOriginal ? "Original signature" : "Forged signature"}
            detail={
              isOriginal
                ? "The signature to check matches the original."
                : "The signature to check does not match the original."
            }
            confidence={result.confidence}
          />
        )}

        {status === "error" && (
          <div className="error-box" role="alert">
            <strong>Could not verify these signatures.</strong>
            <span>{error}</span>
          </div>
        )}
      </div>
    </div>
  );
}
