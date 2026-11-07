import { useEffect, useRef, useState } from "react";

const MAX_MB = 10;
const ACCEPTED = ["image/png", "image/jpeg", "image/webp"];

/**
 * Drag-and-drop / click image picker with preview.
 * `children` is rendered on top of the preview (used for bounding boxes).
 */
export default function ImageUpload({
  id,
  label,
  hint,
  file,
  onSelect,
  onClear,
  disabled = false,
  children,
}) {
  const [preview, setPreview] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef(null);

  useEffect(() => {
    if (!file) {
      setPreview(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function accept(f) {
    if (!f) return;
    if (!ACCEPTED.includes(f.type)) {
      setError("Use a PNG, JPG or WebP image.");
      return;
    }
    if (f.size > MAX_MB * 1024 * 1024) {
      setError(`This image is larger than ${MAX_MB} MB. Choose a smaller one.`);
      return;
    }
    setError("");
    onSelect(f);
  }

  function clear() {
    if (inputRef.current) inputRef.current.value = "";
    setError("");
    onClear();
  }

  return (
    <section className="upload">
      <h2 className="upload-label">{label}</h2>

      {file && preview ? (
        <div className="preview">
          <div className="preview-stage">
            <div className="preview-frame">
              <img src={preview} alt={`${label} preview`} />
              <div className="overlay">{children}</div>
            </div>
          </div>
          <div className="preview-bar">
            <span className="file-name" title={file.name}>
              {file.name}
            </span>
            <button
              type="button"
              className="link-btn"
              onClick={clear}
              disabled={disabled}
            >
              Remove
            </button>
          </div>
        </div>
      ) : (
        <label
          htmlFor={id}
          className={`dropzone ${dragging ? "is-dragging" : ""} ${
            disabled ? "is-disabled" : ""
          }`}
          onDragOver={(e) => {
            e.preventDefault();
            if (!disabled) setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            if (!disabled) accept(e.dataTransfer.files?.[0]);
          }}
        >
          <input
            ref={inputRef}
            id={id}
            type="file"
            accept={ACCEPTED.join(",")}
            className="sr-only"
            disabled={disabled}
            onChange={(e) => accept(e.target.files?.[0])}
          />
          <svg width="34" height="34" viewBox="0 0 24 24" aria-hidden="true">
            <path
              d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M4 16v2.5A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5V16"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          <span className="dz-title">Drop an image here or browse</span>
          <span className="dz-hint">{hint}</span>
        </label>
      )}

      {error && (
        <p className="field-error" role="alert">
          {error}
        </p>
      )}
    </section>
  );
}
