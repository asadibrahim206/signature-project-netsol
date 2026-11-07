# SignatureLens – React frontend

Two tools in one clean UI:

- **Signature detection** – upload one document image → "Signature found / No signature found" (with a highlight box if the API returns one).
- **Signature verification** – upload an original signature + a signature to check → "Original / Forged".

The app works right now with **fake (mock) results**, so you can demo the UI before the model/API is ready.

## Run it

1. Install Node.js 18 or newer (https://nodejs.org).
2. Open this folder in VS Code, then open the terminal (Ctrl + `).
3. Install packages (first time only):
   ```
   npm install
   ```
4. Start the app:
   ```
   npm run dev
   ```
5. Open http://localhost:5173 if it does not open automatically.

## Folder map

```
src/
  App.jsx                   page shell + tab switching
  components/               Header, ImageUpload, ResultCard, Spinner
  pages/DetectionPage.jsx   detection screen
  pages/VerificationPage.jsx verification screen
  services/api.js           <-- the ONLY file teammates edit
  index.css                 all styles and colors (see :root at the top)
```

## For teammates: connecting the real API

1. Open `.env` and set:
   ```
   VITE_USE_MOCK=false
   VITE_API_URL=http://localhost:8000
   ```
2. Restart `npm run dev` (Vite reads `.env` only at startup).
3. Make the backend match the contract in the top comment of `src/services/api.js`, **or** edit the three marked lines in that file (URL, field name, reply keys) to match the backend.

Expected contract:

| Endpoint | Send (multipart) | Reply |
|---|---|---|
| `POST /api/detect` | `image` | `{ "signature_detected": true, "confidence": 0.93, "boxes": [{"x1":0.55,"y1":0.7,"x2":0.9,"y2":0.88}] }` |
| `POST /api/verify` | `original`, `test` | `{ "result": "original" \| "forged", "confidence": 0.87 }` |

`confidence` and `boxes` are optional. Box values are fractions (0–1) of image width/height.

**CORS:** if the backend is FastAPI, add this so the browser is allowed to call it:

```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"],
                   allow_methods=["*"], allow_headers=["*"])
```

## Build for production

```
npm run build
```
The output is in `dist/`.
