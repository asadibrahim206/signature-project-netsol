import { useState } from "react";
import ImageUpload from "../components/ImageUpload.jsx";
import ResultCard from "../components/ResultCard.jsx";
import Spinner from "../components/Spinner.jsx";
import { detectSignature } from "../services/api.js";

export default function DetectionPage() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | loading | done | error
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  function reset() {
    setFile(null);
    setResult(null);
    setError("");
    setStatus("idle");
  }

  function pick(f) {
    setFile(f);
    setResult(null);
    setError("");
    setStatus("idle");
  }

  async function run() {
    if (!file) return;
    setStatus("loading");
    setError("");
    setResult(null);
    try {
      const data = await detectSignature(file);
      setResult(data);
      setStatus("done");
    } catch (e) {
      setError(e.message || "Something went wrong. Try again.");
      setStatus("error");
    }
  }

  const loading = status === "loading";

  return (
    <div className="page">
      <div className="page-head">
        <h1>Find a signature in a document</h1>
        <p>
          Upload a scan or photo of a document. The model will tell you whether
          it contains a signature.
        </p>
      </div>

      <div className="stack">
        <ImageUpload
          id="detect-image"
          label="Document image"
          hint="PNG, JPG or WebP, up to 10 MB"
          file={file}
          onSelect={pick}
          onClear={reset}
          disabled={loading}
        >
          {result?.boxes?.map((b, i) => (
            <span
              key={i}
              className="box"
              style={{
                left: `${b.x1 * 100}%`,
                top: `${b.y1 * 100}%`,
                width: `${(b.x2 - b.x1) * 100}%`,
                height: `${(b.y2 - b.y1) * 100}%`,
              }}
            />
          ))}
        </ImageUpload>

        <div className="actions">
          <button
            type="button"
            className="btn btn-primary"
            onClick={run}
            disabled={!file || loading}
          >
            {loading && <Spinner />}
            {loading ? "Checking…" : "Detect signature"}
          </button>
          {(file || result) && !loading && (
            <button type="button" className="btn btn-ghost" onClick={reset}>
              Start over
            </button>
          )}
        </div>

        {status === "done" && result && (
          <ResultCard
            tone={result.detected ? "ok" : "neutral"}
            title={result.detected ? "Signature found" : "No signature found"}
            detail={
              result.detected
                ? "The highlighted area on the image shows where the model found it."
                : "The model did not find a signature in this image."
            }
            confidence={result.confidence}
          />
        )}

        {status === "error" && (
          <div className="error-box" role="alert">
            <strong>Could not check this image.</strong>
            <span>{error}</span>
          </div>
        )}
      </div>
    </div>
  );
}
