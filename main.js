/* RedactIt landing page behaviour.
 *
 * Every enhancement here is optional: the page's content, its navigation and the
 * Get Started scroll all work through HTML and CSS alone, so a failure in this
 * file (or no JavaScript at all) leaves a complete, readable page.
 */
(function () {
  "use strict";

  const motionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
  const desktopQuery = window.matchMedia("(min-width: 720px)");
  const prefersReducedMotion = () => motionQuery.matches;

  /* ------------------------------------------------- background video gate */
  // The loop is ~13.8 MB. Attach it only where motion is welcome and the network
  // does not look metered, so constrained devices never download it at all.
  function shouldLoadVideo() {
    if (prefersReducedMotion()) return false;

    const connection = navigator.connection;
    if (!connection) return true;
    if (connection.saveData) return false;

    const slow = /(^|\b)(2g|slow-2g)\b/.test(connection.effectiveType || "");
    return !slow;
  }

  function startBackgroundVideo() {
    const video = document.getElementById("hero-video");
    const still = document.querySelector(".hero__still");
    if (!video || !shouldLoadVideo()) return;

    // Two sources: the browser walks them in order, so a missing local file falls
    // through to the CDN without any manual error handling.
    const sources = [video.dataset.srcLocal, video.dataset.srcRemote].filter(Boolean);
    if (!sources.length) return;

    for (const src of sources) {
      const node = document.createElement("source");
      node.src = src;
      node.type = "video/mp4";
      video.appendChild(node);
    }

    // The still is the composition until a real frame exists; retire it only then,
    // so a refused autoplay or a dead CDN leaves the poster rather than black.
    video.addEventListener("playing", () => {
      video.classList.add("is-live");
      if (still) still.hidden = true;
    }, { once: true });

    video.load();
    const playback = video.play();
    if (playback && typeof playback.catch === "function") {
      playback.catch(() => {
        /* Autoplay refused: the still frame is the intended fallback. */
      });
    }
  }

  /* ------------------------------------------------------- gentle parallax */
  function initParallax() {
    const hero = document.getElementById("top");
    if (!hero || prefersReducedMotion()) return;

    let queued = false;
    const write = () => {
      queued = false;
      const progress = Math.min(Math.max(window.scrollY, 0), hero.offsetHeight);
      hero.style.setProperty("--parallax", (progress * 0.09).toFixed(2) + "px");
    };

    window.addEventListener(
      "scroll",
      () => {
        if (queued) return;
        queued = true;
        requestAnimationFrame(write);
      },
      { passive: true }
    );
  }

  function initMobileMenu() {
    const button = document.querySelector(".menu-orb");
    const sheet = document.getElementById("mobile-menu");
    if (!button || !sheet) return;

    const links = Array.from(sheet.querySelectorAll("a[href]"));

    const isOpen = () => button.getAttribute("aria-expanded") === "true";

    function open() {
      sheet.hidden = false;
      // Let the sheet paint before transitioning its opacity.
      requestAnimationFrame(() => sheet.classList.add("is-open"));
      button.setAttribute("aria-expanded", "true");
      button.setAttribute("aria-label", "Close navigation menu");
      document.body.classList.add("menu-open");
      (links[0] || button).focus();
    }

    function close(options) {
      const settings = options || {};
      sheet.classList.remove("is-open");
      button.setAttribute("aria-expanded", "false");
      button.setAttribute("aria-label", "Open navigation menu");
      document.body.classList.remove("menu-open");
      window.setTimeout(() => {
        sheet.hidden = true;
      }, prefersReducedMotion() ? 0 : 280);
      if (settings.returnFocus) button.focus();
    }

    button.addEventListener("click", () => (isOpen() ? close({ returnFocus: true }) : open()));

    sheet.addEventListener("click", (event) => {
      if (!sheet.contains(event.target)) close({ returnFocus: true });
    });

    links.forEach((link) => {
      link.addEventListener("click", () => close());
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && isOpen()) close({ returnFocus: true });
    });

    // Growing back to desktop width retires the sheet: the pill nav is visible again.
    const onViewportChange = (event) => {
      if (event.matches && isOpen()) close();
    };
    if (typeof desktopQuery.addEventListener === "function") {
      desktopQuery.addEventListener("change", onViewportChange);
    } else if (typeof desktopQuery.addListener === "function") {
      desktopQuery.addListener(onViewportChange);
    }

    document.addEventListener("keydown", (event) => {
      if (event.key !== "Tab" || !isOpen()) return;
      const focusables = [button].concat(links);
      const first = focusables[0];
      const last = focusables[focusables.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    });
  }

  /* ------------------------------------------- destination focus & reveal */
  function initScrollTarget() {
    const destination = document.getElementById("reference-section");
    if (!destination) return;

    document.querySelectorAll('a[href="#reference-section"]').forEach((link) => {
      link.addEventListener("click", () => {
        // The browser has already started the anchor scroll; move the sequential
        // focus start point without disturbing it.
        window.setTimeout(() => {
          destination.focus({ preventScroll: true });
        }, prefersReducedMotion() ? 0 : 420);
      });
    });
  }

  function initReveal() {
    const inner = document.querySelector(".reference__inner");
    if (!inner || prefersReducedMotion() || !("IntersectionObserver" in window)) return;

    // Opt in only now, so a browser without JS never sees hidden content.
    inner.classList.add("is-armed");

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          entry.target.classList.add("is-revealed");
          observer.unobserve(entry.target);
        });
      },
      { threshold: 0.25 }
    );
    observer.observe(inner);
  }

  /* ========================================================== backend UI ==
   * The composer in #reference-section is a thin client over the real FastAPI
   * backend: /analyze, /sessions/{id}/review, /sessions/{id}/redact and
   * /downloads/{token}. Nothing is simulated — every number, warning and file
   * name below comes from a backend response, and a failed request is reported
   * as a failure. The browser talks only to that backend; the model server is
   * the backend's own dependency.
   * ======================================================================== */

  // Mirrors the defaults in backend/app/config.py. The backend re-validates all
  // of it; this only lets the page explain a limit before a round trip fails.
  const LIMITS = {
    maxFiles: 5,
    maxFileBytes: 25 * 1024 * 1024,
    maxTotalBytes: 60 * 1024 * 1024,
    maxInstructionChars: 2000,
    extensions: ["pdf", "docx"],
  };

  function resolveApiBase() {
    const params = new URLSearchParams(window.location.search);
    const override = params.get("api");
    const configured = window.RedactItConfig && window.RedactItConfig.apiBase;
    const candidates = [override, configured, "/api/v1"];
    for (const candidate of candidates) {
      if (!candidate) continue;
      try {
        return new URL(candidate, window.location.href).href.replace(/\/+$/, "");
      } catch (error) {
        /* fall through to the next candidate */
      }
    }
    return new URL("/api/v1", window.location.href).href.replace(/\/+$/, "");
  }

  const API_BASE = resolveApiBase();

  function ApiError(code, message, status, details) {
    this.name = "ApiError";
    this.code = code;
    this.message = message;
    this.status = status;
    this.details = details || {};
  }
  ApiError.prototype = Object.create(Error.prototype);

  async function api(path, options) {
    let response;
    try {
      response = await fetch(API_BASE + path, options);
    } catch (cause) {
      throw new ApiError(
        "BACKEND_UNREACHABLE",
        "The RedactIt backend did not answer at " +
          API_BASE +
          ". Start it on this machine (uvicorn app.main:app --port 8000, from backend/) and retry.",
        0,
        {}
      );
    }

    const text = await response.text();
    let payload = null;
    if (text) {
      try {
        payload = JSON.parse(text);
      } catch (cause) {
        payload = null;
      }
    }

    if (!response.ok) {
      const body = payload && payload.error;
      throw new ApiError(
        body ? body.code : "HTTP_" + response.status,
        body
          ? body.message
          : text
            ? text.slice(0, 240)
            : "The backend replied with status " + response.status + " and no explanation.",
        response.status,
        body ? body.details || {} : {}
      );
    }
    return payload;
  }

  const backend = {
    analyze: (form) => api("/analyze", { method: "POST", body: form }),
    review: (id, approved, rejected) =>
      api("/sessions/" + encodeURIComponent(id) + "/review", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          approved_detection_ids: approved,
          rejected_detection_ids: rejected,
        }),
      }),
    redact: (id) => api("/sessions/" + encodeURIComponent(id) + "/redact", { method: "POST" }),
    discard: (id) => api("/sessions/" + encodeURIComponent(id) + "", { method: "DELETE" }),
  };

  /* ------------------------------------------------------------- helpers */
  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
    return node;
  }

  function formatBytes(value) {
    if (!Number.isFinite(value)) return "";
    if (value < 1024) return value + " B";
    if (value < 1024 * 1024) return (value / 1024).toFixed(0) + " KB";
    return (value / (1024 * 1024)).toFixed(1) + " MB";
  }

  function formatTime(iso) {
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return "";
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  function extensionOf(name) {
    const dot = name.lastIndexOf(".");
    return dot === -1 ? "" : name.slice(dot + 1).toLowerCase();
  }

  function initAnalysisUi() {
    const form = document.getElementById("analyze-form");
    const fileInput = document.getElementById("files");
    const fileList = document.getElementById("file-list");
    const instruction = document.getElementById("instruction");
    const instructionError = document.getElementById("instruction-error");
    const modeSelect = document.getElementById("mode");
    const submitBtn = document.getElementById("analyze-btn");
    const submitLabel = document.getElementById("analyze-btn-text");
    const clearBtn = document.getElementById("clear-btn");
    const chips = document.getElementById("example-chips");

    const results = document.getElementById("results");
    const resultsTitle = document.getElementById("results-title");
    const resultsMeta = document.getElementById("results-meta");
    const progress = document.getElementById("progress");
    const progressText = document.getElementById("progress-text");
    const notices = document.getElementById("notices");
    const plan = document.getElementById("plan");
    const planBody = document.getElementById("plan-body");
    const docs = document.getElementById("docs");
    const actionBar = document.getElementById("action-bar");
    const tally = document.getElementById("tally");
    const redactBtn = document.getElementById("redact-btn");
    const outputs = document.getElementById("outputs");
    const restartBtn = document.getElementById("restart-btn");

    if (!form || !results) return;

    const state = {
      files: [],
      sessionId: null,
      detections: [],
      decisions: new Map(),
      busy: false,
      redacted: false,
    };

    /* Model availability is no longer probed on load; it surfaces as a
     * MODEL_UNAVAILABLE notice on the analyze result, which is where the user can
     * act on it. */

    /* ------------------------------------------------------- file picking */
    function fileProblem(file, index) {
      if (LIMITS.extensions.indexOf(extensionOf(file.name)) === -1) {
        return "Unsupported type — only PDF and DOCX are accepted.";
      }
      if (file.size > LIMITS.maxFileBytes) {
        return "Too large — the limit is " + formatBytes(LIMITS.maxFileBytes) + ".";
      }
      if (index >= LIMITS.maxFiles) {
        return "Beyond the " + LIMITS.maxFiles + "-file limit.";
      }
      let total = 0;
      for (let i = 0; i <= Math.min(index, LIMITS.maxFiles - 1); i += 1) {
        total += state.files[i].size;
      }
      if (total > LIMITS.maxTotalBytes) {
        return "The batch exceeds " + formatBytes(LIMITS.maxTotalBytes) + " in total.";
      }
      return "";
    }

    function renderFiles() {
      clear(fileList);
      state.files.forEach((file, index) => {
        const problem = fileProblem(file, index);
        const chip = el("li", problem ? "file-chip file-chip--bad" : "file-chip");
        chip.appendChild(el("span", "file-chip__name", file.name));
        chip.appendChild(el("span", "file-chip__size", formatBytes(file.size)));
        if (problem) chip.appendChild(el("span", "file-chip__size", problem));

        const remove = el("button", "file-chip__x", "×");
        remove.type = "button";
        remove.setAttribute("aria-label", "Remove " + file.name);
        remove.addEventListener("click", () => {
          state.files.splice(index, 1);
          renderFiles();
        });
        chip.appendChild(remove);
        fileList.appendChild(chip);
      });
      syncClearButton();
    }

    fileInput.addEventListener("change", () => {
      const picked = Array.prototype.slice.call(fileInput.files || []);
      const seen = new Set(state.files.map((file) => file.name + ":" + file.size));
      picked.forEach((file) => {
        const key = file.name + ":" + file.size;
        if (seen.has(key)) return;
        seen.add(key);
        state.files.push(file);
      });
      fileInput.value = "";
      renderFiles();
    });

    /* ------------------------------------------------------- instruction */
    function syncClearButton() {
      clearBtn.hidden = !instruction.value.trim() && state.files.length === 0;
    }

    function setFieldError(message) {
      if (message) {
        instructionError.hidden = false;
        instructionError.textContent = message;
        instruction.setAttribute("aria-invalid", "true");
      } else {
        instructionError.hidden = true;
        instructionError.textContent = "";
        instruction.removeAttribute("aria-invalid");
      }
    }

    instruction.addEventListener("input", () => {
      setFieldError("");
      syncClearButton();
    });

    instruction.addEventListener("keydown", (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        event.preventDefault();
        form.requestSubmit();
      }
    });

    chips.addEventListener("click", (event) => {
      const chip = event.target.closest("[data-example]");
      if (!chip) return;
      instruction.value = chip.getAttribute("data-example");
      setFieldError("");
      syncClearButton();
      instruction.focus();
    });

    clearBtn.addEventListener("click", () => {
      instruction.value = "";
      state.files = [];
      renderFiles();
      setFieldError("");
      syncClearButton();
      // With no session behind it the results panel only describes an attempt that
      // is already over, so clearing the composer clears that too.
      if (!state.sessionId) resetSession();
      fileInput.focus();
    });

    /* ----------------------------------------------------- busy handling */
    function setBusy(busy, label) {
      state.busy = busy;
      submitBtn.disabled = busy;
      submitBtn.classList.toggle("is-busy", busy);
      submitLabel.textContent = busy ? label || "Working…" : "Analyze documents";
      progress.hidden = !busy;
      if (busy) progressText.textContent = label || "Working…";
      updateTally();
    }

    /* --------------------------------------------------------- notices */
    function addNotice(kind, code, message) {
      const notice = el("div", "notice notice--" + kind);
      if (code) notice.appendChild(el("span", "notice__code", code));
      notice.appendChild(document.createTextNode(message));
      notices.appendChild(notice);
    }

    function renderNotices(data) {
      clear(notices);
      (data.warnings || []).forEach((warning) => addNotice("warn", "", warning));
      (data.document_failures || []).forEach((failure) =>
        addNotice("bad", failure.code, failure.document_name + ": " + failure.message)
      );
    }

    /* ---------------------------------------------------- interpretation */
    function renderPlan(data) {
      clear(planBody);
      const interpretation = data.interpretation || {};
      plan.hidden = true;

      const rows = [];
      rows.push([
        "Route",
        interpretation.model_used
          ? "Gemma proposed, the deterministic parser validated (" +
            (interpretation.source || "model+deterministic") +
            ")"
          : "Keyword parsing only — the model was not used for this run",
      ]);
      if (interpretation.targets && interpretation.targets.length) {
        rows.push([
          "Removing",
          interpretation.targets
            .map((target) => target.term + " (" + target.category + ")")
            .join(", "),
        ]);
      } else {
        rows.push(["Removing", "nothing the parser could act on"]);
      }
      if (interpretation.exclusions && interpretation.exclusions.length) {
        rows.push([
          "Keeping",
          interpretation.exclusions
            .map((exclusion) => {
              const protectedTexts = (exclusion.protected_texts || []).join(", ");
              return exclusion.term + (protectedTexts ? " → " + protectedTexts : "");
            })
            .join(", "),
        ]);
      }
      if (interpretation.document_scope === "named") {
        rows.push(["Scope", "only the named documents: " + (interpretation.document_names || []).join(", ")]);
      }
      const summary = data.summary || {};
      if (summary.unmatched_exact_terms && summary.unmatched_exact_terms.length) {
        rows.push(["Not found", summary.unmatched_exact_terms.join(", ")]);
      }
      if (summary.suppressed_by_exclusion) {
        rows.push([
          "Suppressed",
          summary.suppressed_by_exclusion +
            " match(es) were dropped because the interpretation also read them as things to keep",
        ]);
      }
      (interpretation.warnings || []).forEach((warning) => rows.push(["Warning", warning]));
      if (interpretation.rejected_model_proposals && interpretation.rejected_model_proposals.length) {
        rows.push([
          "Rejected",
          "The parser refused these model proposals: " +
            interpretation.rejected_model_proposals.join(", "),
        ]);
      }
      const limitations = (data.limitations || []).concat(
        (data.documents || []).reduce((all, document) => all.concat(document.limitations || []), [])
      );
      const uniqueLimitations = Array.from(new Set(limitations));
      if (uniqueLimitations.length) {
        rows.push(["Limits", uniqueLimitations.length + " known limitations — listed below"]);
      }

      rows.forEach(([key, value]) => {
        const row = el("div", "plan__row");
        row.appendChild(el("span", "plan__key", key));
        row.appendChild(el("span", "", value));
        planBody.appendChild(row);
      });

      uniqueLimitations.forEach((limitation) => {
        const row = el("div", "plan__row");
        row.appendChild(el("span", "plan__key", "Limit"));
        row.appendChild(el("span", "", limitation));
        planBody.appendChild(row);
      });

      if (rows.length) plan.hidden = false;
    }

    /* --------------------------------------------------- detection rows */
    function updateTally() {
      let approved = 0;
      state.decisions.forEach((decision) => {
        if (decision === "redact") approved += 1;
      });
      const total = state.detections.length;
      tally.textContent =
        approved === 0
          ? "Nothing approved yet. Every detection defaults to keep, so nothing will be removed until you say so."
          : approved +
            " of " +
            total +
            " detection(s) approved for removal. The other " +
            (total - approved) +
            " stay untouched.";
      redactBtn.disabled = approved === 0 || state.busy || state.redacted;
    }

    function detectionRow(detection) {
      const row = el("div", "det");
      row.dataset.id = detection.detection_id;

      const body = el("div", "det__body");
      body.appendChild(el("span", "det__text", detection.matched_text));

      const context = el("span", "det__ctx");
      if (detection.context_before) {
        context.appendChild(el("span", "", "…" + detection.context_before));
      }
      context.appendChild(el("mark", "", " " + detection.matched_text + " "));
      if (detection.context_after) {
        context.appendChild(el("span", "", detection.context_after + "…"));
      }
      body.appendChild(context);

      const tags = el("div", "det__tags");
      tags.appendChild(el("span", "tag", detection.category));
      if (detection.pattern_name) tags.appendChild(el("span", "tag", detection.pattern_name));
      tags.appendChild(
        el("span", "tag", detection.source === "both" ? "instruction + model" : detection.source)
      );
      if (detection.source === "model" || detection.source === "both") {
        tags.appendChild(el("span", "tag tag--model", Math.round(detection.confidence * 100) + "%"));
      }
      if (detection.heuristic) tags.appendChild(el("span", "tag tag--flag", "heuristic"));
      if (detection.fuzzy) tags.appendChild(el("span", "tag tag--flag", "fuzzy match"));
      if (detection.page) tags.appendChild(el("span", "tag", "page " + detection.page));
      else if (detection.paragraph_path) tags.appendChild(el("span", "tag", detection.paragraph_path));
      if (detection.occurrence > 1) {
        tags.appendChild(el("span", "tag", "occurrence " + detection.occurrence));
      }
      body.appendChild(tags);
      row.appendChild(body);

      const choice = el("div", "det__choice");
      choice.setAttribute("role", "radiogroup");
      choice.setAttribute(
        "aria-label",
        "Decision for " + detection.matched_text + " in " + detection.document_name
      );
      [
        ["keep", "Keep"],
        ["redact", "Redact"],
      ].forEach(([value, label], optionIndex) => {
        const inputId = detection.detection_id + "-" + value;
        const input = el("input");
        input.type = "radio";
        input.id = inputId;
        input.name = "decision-" + detection.detection_id;
        input.value = value;
        input.checked = (state.decisions.get(detection.detection_id) || "keep") === value;
        input.disabled = state.redacted;

        const option = el("label", "det__choice__opt", label);
        option.htmlFor = inputId;
        if (optionIndex) option.classList.add("det__choice__opt--last");

        input.addEventListener("change", () => {
          if (!input.checked) return;
          state.decisions.set(detection.detection_id, value);
          row.classList.toggle("det--redact", value === "redact");
          updateTally();
        });
        choice.appendChild(input);
        choice.appendChild(option);
      });
      row.appendChild(choice);

      if (state.decisions.get(detection.detection_id) === "redact") {
        row.classList.add("det--redact");
      }
      return row;
    }

    function renderDocuments(data) {
      clear(docs);
      const byDocument = new Map();
      (data.documents || []).forEach((document) => {
        byDocument.set(document.document_id, { summary: document, detections: [] });
      });
      state.detections.forEach((detection) => {
        const bucket = byDocument.get(detection.document_id);
        if (bucket) bucket.detections.push(detection);
      });

      if (!byDocument.size) {
        addNotice("bad", "NO_DOCUMENTS", "The backend analysed no documents in this run.");
        return;
      }

      byDocument.forEach((bucket) => {
        const section = el("section", "doc");
        const head = el("div", "doc__head");
        head.appendChild(el("h4", "doc__name", bucket.summary.document_name));
        const meta = [
          (bucket.summary.file_kind || "").toUpperCase(),
          bucket.summary.page_count ? bucket.summary.page_count + " pages" : "",
          formatBytes(bucket.summary.size_bytes),
          bucket.detections.length
            ? bucket.detections.length + " candidate(s)"
            : "no candidates found",
        ]
          .filter(Boolean)
          .join(" · ");
        head.appendChild(el("span", "doc__meta", meta));

        if (bucket.detections.length && !state.redacted) {
          const bulk = el("div", "doc__bulk");
          [["redact", "All redact"], ["keep", "All keep"]].forEach(([value, label]) => {
            const button = el("button", "ghost-btn", label);
            button.type = "button";
            button.addEventListener("click", () => {
              bucket.detections.forEach((detection) =>
                state.decisions.set(detection.detection_id, value)
              );
              renderDocuments(data);
              updateTally();
            });
            bulk.appendChild(button);
          });
          head.appendChild(bulk);
        }
        section.appendChild(head);

        const list = el("div", "dets");
        if (bucket.detections.length) {
          bucket.detections.forEach((detection) => list.appendChild(detectionRow(detection)));
        } else {
          list.appendChild(el("p", "doc__empty", "No candidate matches in this document."));
        }
        section.appendChild(list);

        (bucket.summary.warnings || []).concat(bucket.summary.limitations || []).forEach((note) => {
          section.appendChild(el("p", "doc__note", note));
        });

        docs.appendChild(section);
      });
    }

    /* ---------------------------------------------------- results header */
    function renderAnalysis(data) {
      state.sessionId = data.session_id;
      state.detections = data.detections || [];
      state.decisions = new Map();
      state.redacted = false;
      clear(outputs);

      const summary = data.summary || {};
      resultsTitle.textContent = summary.detections
        ? "Review what was found"
        : "No candidates to review";
      resultsMeta.textContent =
        "Session " +
        data.session_id.slice(0, 8) +
        " · " +
        (summary.documents_analysed || 0) +
        " document(s) analysed · " +
        (summary.detections || 0) +
        " candidate match(es) · expires " +
        formatTime(data.expires_at);

      renderNotices(data);
      if (data.interpretation && data.interpretation.ambiguous) {
        addNotice(
          "warn",
          "INSTRUCTION_AMBIGUOUS",
          data.interpretation.clarification_needed ||
            "The instruction was ambiguous, so only the parts the parser could confirm were used."
        );
      }
      renderPlan(data);
      renderDocuments(data);
      updateTally();
      actionBar.hidden = state.detections.length === 0 || state.redacted;
    }

    /* ------------------------------------------------ analyze submission */
    function validate() {
      const text = instruction.value.trim();
      if (!state.files.length) {
        setFieldError("Add at least one PDF or DOCX document to analyse.");
        fileInput.focus();
        return null;
      }
      for (let index = 0; index < state.files.length; index += 1) {
        const file = state.files[index];
        const problem = fileProblem(file, index);
        if (problem) {
          setFieldError(file.name + ": " + problem);
          return null;
        }
      }
      if (!text) {
        setFieldError("Describe what should be redacted, or pick an example below.");
        instruction.focus();
        return null;
      }
      if (text.length > LIMITS.maxInstructionChars) {
        setFieldError(
          "The instruction is " +
            text.length +
            " characters; the backend accepts at most " +
            LIMITS.maxInstructionChars +
            "."
        );
        instruction.focus();
        return null;
      }
      setFieldError("");
      return { files: state.files.slice(), text };
    }

    async function analyze() {
      const payload = validate();
      if (!payload) return;

      const formBody = new FormData();
      payload.files.forEach((file) => formBody.append("files", file, file.name));
      formBody.append("instruction", payload.text);
      formBody.append("interpretation_mode", modeSelect.value);

      results.hidden = false;
      resultsTitle.textContent = "Analysing";
      resultsMeta.textContent = "";
      clear(notices);
      clear(planBody);
      clear(docs);
      clear(outputs);
      plan.hidden = true;
      actionBar.hidden = true;
      setBusy(true, "Reading the documents and interpreting the instruction…");

      try {
        const data = await backend.analyze(formBody);
        renderAnalysis(data);
        results.scrollIntoView({ block: "nearest", behavior: prefersReducedMotion() ? "auto" : "smooth" });
      } catch (error) {
        state.sessionId = null;
        resultsTitle.textContent = "Analysis failed";
        resultsMeta.textContent = "";
        clear(docs);
        addNotice("bad", error.code || "ERROR", error.message);
        if (error.code === "MODEL_UNAVAILABLE" || error.code === "MODEL_NOT_LOADED") {
          addNotice(
            "warn",
            "",
            "You can switch the mode control from “Local Gemma” to “Keywords only” to analyse " +
              "without the model, or start llama-server and retry."
          );
        }
        plan.hidden = true;
        actionBar.hidden = true;
      } finally {
        setBusy(false);
      }
    }

    form.addEventListener("submit", (event) => {
      event.preventDefault();
      if (state.busy) return;
      analyze();
    });

    /* --------------------------------------------- review, redact, output */
    // The verification report carries the on-disk name (prefixed with the document
    // id) while the outcome and download entries carry the display name, so match
    // on the exact name first and then on the suffix.
    function findReport(reports, name) {
      for (let i = 0; i < reports.length; i += 1) {
        if (reports[i].output_name === name) return reports[i];
      }
      for (let i = 0; i < reports.length; i += 1) {
        if (reports[i].output_name.slice(-name.length) === name) return reports[i];
      }
      return null;
    }

    function renderOutputs(data) {
      clear(outputs);
      const reports = data.verification || [];

      (data.outcomes || []).forEach((outcome) => {
        const output =
          (data.outputs || []).find((item) => item.document_id === outcome.document_id) || null;
        const report = findReport(reports, outcome.output_name);
        const card = el(
          "article",
          data.residual_text_found ? "output output--bad" : "output"
        );

        const head = el("div", "output__head");
        head.appendChild(el("h4", "output__name", outcome.output_name));
        head.appendChild(
          el(
            "span",
            "output__meta",
            outcome.applied + " redaction(s) applied to a copy of " + outcome.source_name
          )
        );
        card.appendChild(head);

        if (output) {
          const link = el("a", "output__link");
          link.href = new URL(output.download_url, API_BASE).href;
          link.setAttribute("download", output.output_name);
          link.appendChild(
            el("span", "", "Download " + formatBytes(output.size_bytes))
          );
          card.appendChild(link);
          card.appendChild(
            el(
              "p",
              "output__meta",
              "Link valid until " +
                formatTime(output.expires_at) +
                " · SHA-256 " +
                output.sha256.slice(0, 12) +
                "…"
            )
          );
        }

        outcome.skipped.forEach((skip) => addNotice("warn", "SKIPPED", skip));
        outcome.warnings.forEach((warning) => addNotice("warn", "OUTPUT", warning));

        if (report) {
          const verify = el("div", "verify");
          verify.appendChild(
            el(
              "p",
              "verify__title",
              "Residual-text check · " +
                report.coverage +
                " · " +
                report.checked +
                " string(s) checked"
            )
          );
          const list = el("div", "verify__list");
          (report.results || []).forEach((result) => {
            const line = el("div", "verify__row");
            line.appendChild(el("span", "verify__text", result.text));
            line.appendChild(
              el("span", "verify__status verify__status--" + result.status, result.status)
            );
            line.appendChild(el("span", "verify__detail", result.detail));
            list.appendChild(line);
          });
          if (!(report.results || []).length) {
            list.appendChild(
              el("div", "verify__row", report.not_verifiable_reason || "Nothing to report.")
            );
          }
          verify.appendChild(list);
          card.appendChild(verify);
        }

        outputs.appendChild(card);
      });

      if (data.residual_text_found) {
        addNotice(
          "bad",
          "RESIDUAL_TEXT",
          "Verification found text you approved for removal still present in an output. " +
            "Do not distribute these files; inspect them manually."
        );
      }

      const limitations = Array.from(new Set(data.limitations || []));
      if (limitations.length) {
        const block = el("details", "plan");
        const summary = el("summary", "", "What this check cannot prove");
        block.appendChild(summary);
        const body = el("div", "plan__body");
        limitations.forEach((limitation) => {
          const row = el("div", "plan__row");
          row.appendChild(el("span", "plan__key", "Limit"));
          row.appendChild(el("span", "", limitation));
          body.appendChild(row);
        });
        block.appendChild(body);
        outputs.appendChild(block);
      }
    }

    async function redact() {
      if (!state.sessionId) return;
      const approved = [];
      const rejected = [];
      state.detections.forEach((detection) => {
        if (state.decisions.get(detection.detection_id) === "redact") {
          approved.push(detection.detection_id);
        } else {
          rejected.push(detection.detection_id);
        }
      });

      actionBar.hidden = true;
      clear(outputs);
      setBusy(true, "Recording your review and applying the approved redactions…");

      try {
        await backend.review(state.sessionId, approved, rejected);
        const data = await backend.redact(state.sessionId);
        state.redacted = true;
        resultsTitle.textContent = "Sanitised copies ready";
        resultsMeta.textContent =
          data.status + " · originals untouched · " + (data.outputs || []).length + " file(s)";
        renderOutputs(data);
        // The session is now REDACTED: the backend refuses a second review, so the
        // decisions are locked in place rather than re-rendered.
        docs.querySelectorAll("input, .doc__bulk button").forEach((node) => {
          node.disabled = true;
        });
        restartBtn.textContent = "Start a new analysis";
      } catch (error) {
        addNotice("bad", error.code || "ERROR", error.message);
        actionBar.hidden = false;
        if (error.code === "SESSION_EXPIRED" || error.code === "SESSION_NOT_FOUND") {
          resetSession();
        }
      } finally {
        setBusy(false);
      }
    }

    redactBtn.addEventListener("click", () => {
      if (state.busy) return;
      redact();
    });

    /* ----------------------------------------------------------- restart */
    function resetSession() {
      const id = state.sessionId;
      state.sessionId = null;
      state.detections = [];
      state.decisions = new Map();
      state.redacted = false;
      results.hidden = true;
      clear(notices);
      clear(planBody);
      clear(docs);
      clear(outputs);
      plan.hidden = true;
      actionBar.hidden = true;
      restartBtn.textContent = "Discard and start over";
      if (id) backend.discard(id).catch(() => {});
    }

    restartBtn.addEventListener("click", () => {
      resetSession();
      fileInput.focus();
    });

    syncClearButton();
  }

  startBackgroundVideo();
  initParallax();
  initMobileMenu();
  initScrollTarget();
  initReveal();
  initAnalysisUi();
})();
