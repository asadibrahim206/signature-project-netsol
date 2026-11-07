/**
 * ============================================================
 *  API LAYER  –  the ONLY file teammates need to edit
 * ============================================================
 *
 * Mock mode (default):  VITE_USE_MOCK=true  in .env  -> fake results
 * Real mode:            VITE_USE_MOCK=false and set VITE_API_URL
 *
 * ---- Expected backend contract (change below if yours differs) ----
 *
 * POST {VITE_API_URL}/api/detect        (multipart/form-data)
 *   field:  image  (file)
 *   reply:  {
 *             "signature_detected": true,
 *             "confidence": 0.93,                       // 0..1, optional
 *             "boxes": [                                // optional, normalized 0..1
 *               { "x1": 0.55, "y1": 0.70, "x2": 0.90, "y2": 0.88 }
 *             ]
 *           }
 *
 * POST {VITE_API_URL}/api/verify        (multipart/form-data)
 *   fields: original (file), test (file)
 *   reply:  {
 *             "result": "original" | "forged",
 *             "confidence": 0.87                        // 0..1, optional
 *           }
 *
 * If the backend uses different URLs, field names or reply keys,
 * change them in the three marked places below.
 */

const BASE_URL = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");
const USE_MOCK = import.meta.env.VITE_USE_MOCK !== "false";

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

async function postForm(path, formData) {
  let res;
  try {
    res = await fetch(`${BASE_URL}${path}`, { method: "POST", body: formData });
  } catch {
    throw new Error(
      "Cannot reach the server. Check that the API is running and VITE_API_URL is correct."
    );
  }

  if (!res.ok) {
    let msg = `The server returned an error (${res.status}).`;
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") msg = body.detail;
      else if (typeof body?.error === "string") msg = body.error;
    } catch {
      /* keep default message */
    }
    throw new Error(msg);
  }
  return res.json();
}

/**
 * @param {File} file
 * @returns {Promise<{detected: boolean, confidence?: number, boxes?: Array}>}
 */
export async function detectSignature(file) {
  if (USE_MOCK) {
    await wait(1200);
    const detected = Math.random() > 0.35;
    return {
      detected,
      confidence: 0.8 + Math.random() * 0.19,
      boxes: detected ? [{ x1: 0.52, y1: 0.68, x2: 0.9, y2: 0.88 }] : [],
    };
  }

  const form = new FormData();
  form.append("image", file); // <-- (1) field name

  const data = await postForm("/api/detect", form); // <-- (2) URL

  return {
    detected: Boolean(data.signature_detected), // <-- (3) reply keys
    confidence: data.confidence,
    boxes: data.boxes ?? [],
  };
}

/**
 * @param {File} original  genuine reference signature
 * @param {File} test      signature to check
 * @returns {Promise<{verdict: "original"|"forged", confidence?: number}>}
 */
export async function verifySignature(original, test) {
  if (USE_MOCK) {
    await wait(1400);
    return {
      verdict: Math.random() > 0.5 ? "original" : "forged",
      confidence: 0.75 + Math.random() * 0.24,
    };
  }

  const form = new FormData();
  form.append("original", original); // <-- (1) field names
  form.append("test", test);

  const data = await postForm("/api/verify", form); // <-- (2) URL

  return {
    verdict: String(data.result).toLowerCase() === "forged" ? "forged" : "original", // <-- (3) reply keys
    confidence: data.confidence,
  };
}
