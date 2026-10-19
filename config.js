/* -----------------------------------------------------------------------------
   RedactIt web UI — runtime configuration.

   The UI is dependency-free static files, so there is no build-time environment
   to read. This file is the equivalent knob: change apiBase to point the page at
   whichever RedactIt backend origin should serve the requests.

   main.js resolves the base URL in this order:
     1. ?api=<url> in the page address (handy for testing a second backend)
     2. window.RedactItConfig.apiBase below
     3. the page's own origin + "/api/v1" (when the UI is served by the backend)

   Local development
   -----------------
   Backend:  uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1   (from backend/)
   UI:       python -m http.server 5173 --bind 127.0.0.1                     (from the repo root)
   Model:    llama-server on 127.0.0.1:8080 — the browser never talks to it.
             The backend calls the model, and its CORS allow-list already covers
             http://localhost:5173 and http://127.0.0.1:5173.

   Nothing secret belongs in this file. The browser sends only the files you pick
   and the instruction you type to the local backend; model settings, keys and
   storage paths stay in the backend's own environment.
----------------------------------------------------------------------------- */
window.RedactItConfig = {
  apiBase: "http://127.0.0.1:8000/api/v1",
};
