(function () {
  "use strict";

  // Thread ID -> createdAt mapping, populated from intercepted API responses
  var threadDates = {};

  // ── Fetch interceptor ────────────────────────────────────────────
  var _fetch = window.fetch;
  window.fetch = async function () {
    var response = await _fetch.apply(this, arguments);
    try {
      var url =
        typeof arguments[0] === "string"
          ? arguments[0]
          : arguments[0] && arguments[0].url
            ? arguments[0].url
            : "";
      if (url.indexOf("/threads") !== -1) {
        var clone = response.clone();
        var data = await clone.json();
        if (data && Array.isArray(data.data)) {
          data.data.forEach(function (t) {
            if (t.id && t.createdAt) threadDates[t.id] = t.createdAt;
          });
          scheduleGrouping();
        }
      }
    } catch (_) {}
    return response;
  };

  // ── Date helpers (Portuguese) ────────────────────────────────────
  var MONTHS = [
    "janeiro",
    "fevereiro",
    "março",
    "abril",
    "maio",
    "junho",
    "julho",
    "agosto",
    "setembro",
    "outubro",
    "novembro",
    "dezembro",
  ];

  function dayKey(d) {
    return (
      d.getFullYear() +
      "-" +
      String(d.getMonth() + 1).padStart(2, "0") +
      "-" +
      String(d.getDate()).padStart(2, "0")
    );
  }

  function formatLabel(dateStr) {
    var d = new Date(dateStr);
    var now = new Date();
    var y = new Date(now);
    y.setDate(y.getDate() - 1);

    if (dayKey(d) === dayKey(now)) return "Hoje";
    if (dayKey(d) === dayKey(y)) return "Ontem";

    var s = d.getDate() + " de " + MONTHS[d.getMonth()];
    if (d.getFullYear() !== now.getFullYear()) s += " de " + d.getFullYear();
    return s;
  }

  // ── DOM grouping ─────────────────────────────────────────────────
  var pending = false;

  function scheduleGrouping() {
    if (pending) return;
    pending = true;
    requestAnimationFrame(function () {
      pending = false;
      groupThreads();
    });
  }

  function groupThreads() {
    var container = document.getElementById("thread-history");
    if (!container) return;

    // Remove old headers
    container.querySelectorAll(".thread-date-header").forEach(function (el) {
      el.remove();
    });

    var items = container.querySelectorAll('[id^="thread-"]');
    var lastKey = null;

    items.forEach(function (item) {
      if (item.classList.contains("thread-date-header")) return;

      var tid = item.id.replace(/^thread-/, "");
      var created = threadDates[tid];
      if (!created) return;

      var dk = dayKey(new Date(created));
      if (dk !== lastKey) {
        lastKey = dk;
        var h = document.createElement("div");
        h.className = "thread-date-header";
        h.textContent = formatLabel(created);
        item.parentNode.insertBefore(h, item);
      }
    });
  }

  // ── Sidebar live-inject ──────────────────────────────────────────
  // Chainlit's sidebar only fetches threads on page load.
  // Poll briefly after page load to pick up the current conversation
  // once the first message is sent (thread gets persisted on first msg).

  var threadTemplate = null;
  var injectedIds = new Set();
  var pollTimer = null;
  var pollCount = 0;
  var MAX_POLLS = 12; // 12 * 5s = 60s

  function captureTemplate() {
    if (threadTemplate) return true;
    var container = document.getElementById("thread-history");
    if (!container) return false;
    var first = container.querySelector('[id^="thread-"]');
    if (first) {
      threadTemplate = first.cloneNode(true);
      return true;
    }
    return false;
  }

  function knownSidebarIds() {
    var ids = new Set();
    var container = document.getElementById("thread-history");
    if (!container) return ids;
    container.querySelectorAll('[id^="thread-"]').forEach(function (el) {
      if (!el.classList.contains("thread-date-header")) {
        ids.add(el.id);
      }
    });
    return ids;
  }

  function injectThread(container, thread) {
    var domId = "thread-" + thread.id;
    if (injectedIds.has(domId)) return;
    if (container.querySelector("#" + CSS.escape(domId))) return;

    // Clone template if available, otherwise build minimal element
    var node;
    if (threadTemplate) {
      node = threadTemplate.cloneNode(true);
      node.id = domId;
      var link = node.querySelector("a");
      if (link) link.href = "/thread/" + thread.id;
      var txt = node.querySelector(".truncate");
      if (txt) txt.textContent = thread.name || "Nova Sessão";
      // Remove any active/selected state from the template
      var btn = node.querySelector("[data-active]");
      if (btn) btn.removeAttribute("data-active");
    } else {
      node = document.createElement("li");
      node.id = domId;
      var a = document.createElement("a");
      a.href = "/thread/" + thread.id;
      a.style.cssText =
        "display:block;padding:6px 12px;font-size:0.875rem;border-radius:6px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:inherit;text-decoration:none;";
      a.textContent = thread.name || "Nova Sessão";
      a.onmouseenter = function () {
        a.style.backgroundColor = "hsl(var(--sidebar-accent))";
      };
      a.onmouseleave = function () {
        a.style.backgroundColor = "";
      };
      node.appendChild(a);
    }

    // Insert at top (newest first)
    if (container.firstChild) {
      container.insertBefore(node, container.firstChild);
    } else {
      container.appendChild(node);
    }
    injectedIds.add(domId);
  }

  function pollForNewThreads() {
    pollCount++;
    if (pollCount > MAX_POLLS) {
      clearInterval(pollTimer);
      pollTimer = null;
      return;
    }

    var container = document.getElementById("thread-history");
    if (!container) return;

    captureTemplate();
    var existing = knownSidebarIds();

    _fetch("/project/threads", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pagination: { first: 20 }, filter: {} }),
    })
      .then(function (r) {
        return r.json();
      })
      .then(function (data) {
        if (!data || !Array.isArray(data.data)) return;

        var injected = false;
        data.data.forEach(function (t) {
          if (t.id && t.createdAt) threadDates[t.id] = t.createdAt;
          var domId = "thread-" + t.id;
          if (!existing.has(domId)) {
            injectThread(container, t);
            injected = true;
          }
        });

        if (injected) {
          scheduleGrouping();
          // Stop polling once we've successfully injected
          clearInterval(pollTimer);
          pollTimer = null;
        }
      })
      .catch(function () {});
  }

  function startPolling() {
    if (pollTimer) return;
    pollTimer = setInterval(pollForNewThreads, 5000);
  }

  // ── Observer ─────────────────────────────────────────────────────
  function observe() {
    var el = document.getElementById("thread-history");
    if (el) {
      new MutationObserver(function () {
        scheduleGrouping();
      }).observe(el, { childList: true, subtree: true });
      scheduleGrouping();
      startPolling();
      return;
    }
    new MutationObserver(function (_, obs) {
      if (document.getElementById("thread-history")) {
        obs.disconnect();
        observe();
      }
    }).observe(document.body, { childList: true, subtree: true });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", observe);
  } else {
    observe();
  }
})();
