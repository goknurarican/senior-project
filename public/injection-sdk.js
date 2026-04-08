// public/injection-sdk.js
(function () {
  "use strict";

  // Tarayıcının orijinal alert'ini saklıyoruz (sonradan çağırabilmek için)
  window.__originalAlert__ = window.alert;

  // Feedback Late senaryosu için alert'i yeniden tanımlıyoruz
  window.alert = function (message) {
    // Feedback Late açıksa → bildirim gecikmeli çıkacak
    if (window.__FEEDBACK_LATE__) {
      setTimeout(() => {
        window.__originalAlert__(message); // gecikmiş alert
      }, 1200); // 1.2 saniye gecikme
    }
    // Feedback Late kapalıysa → alert normal şekilde hemen çıkar
    else {
      window.__originalAlert__(message);
    }
  };

  window.originalFetch = window.fetch;

  window.ExperimentSDK = {
    sessionId: null,
    experimentGroup: null,
    phase: "control",
    scenarios: [],
    triggeredScenarios: new Map(),
    eventQueue: [],
    pageLoadTime: Date.now(),
    lastScenarioTime: 0,

    // === MOUSE TRACKING ===
    mouseTrajectory: [],
    lastMouseTime: 0,
    MOUSE_THROTTLE_MS: 100,
    experimentStartTime: performance.now(),

    // === AYARLAR ===
    COOLDOWN_MS: 5000,
    MAX_SCENARIOS_PER_SESSION: 9999,

    // Tekrar listener başlatmayı önlemek için
    _watcherStarted: false,
    _routeListenerAttached: false,
    _sessionListenerAttached: false,
    _cartListenersAttached: false,
    _mouseTrackingAttached: false,
    _flushIntervalStarted: false,

    init: async function () {
      try {
        const res = await window.originalFetch("/api/session/info?t=" + Date.now());
        const data = await res.json();
        this.sessionId = data.sessionId || null;
        this.experimentGroup = data.experimentGroup || "control";
        this.phase = data.phase || "control";
      } catch (e) {
        return;
      }

      if (!this._routeListenerAttached) {
        this._routeListenerAttached = true;

        window.addEventListener("route:change", async () => {
          if (window.fetch !== window.originalFetch) window.fetch = window.originalFetch;
          document.body.style.cursor = "default";

          const oldOverlay = document.getElementById("blocking-overlay");
          if (oldOverlay) oldOverlay.remove();

          this.pageLoadTime = Date.now();
          await this.loadScenarios();
        });
      }

      if (!this._sessionListenerAttached) {
        this._sessionListenerAttached = true;

        window.addEventListener("session:update", async (e) => {
          const newGroup = e.detail?.experimentGroup;

          if (newGroup && newGroup !== this.experimentGroup) {
            console.log(
              `🔄 Session değişti: ${this.experimentGroup} -> ${newGroup}. SDK Yeniden Başlatılıyor...`
            );

            this.experimentGroup = newGroup;
            this.sessionId = e.detail?.sessionId || this.sessionId;
            this.phase = e.detail?.phase || this.phase || "task";

            if (window.fetch !== window.originalFetch) window.fetch = window.originalFetch;
            document.body.style.cursor = "default";

            const oldOverlay = document.getElementById("blocking-overlay");
            if (oldOverlay) oldOverlay.remove();

            if (this.phase !== "control" && this.experimentGroup !== "control") {
              await this.loadScenarios();
            }
          }
        });
      }

      this.trackPageView();

      if (!this._mouseTrackingAttached) {
        this.initMouseTracking();
        this._mouseTrackingAttached = true;
      }

      if (!this._flushIntervalStarted) {
        setInterval(() => this.flushEvents(), 3000);
        this._flushIntervalStarted = true;
      }

      if (this.phase !== "control" && this.experimentGroup !== "control") {
        await this.loadScenarios();

        if (!this._watcherStarted) {
          this.startScenarioWatcher();
          this._watcherStarted = true;
        }

        if (!this._cartListenersAttached) {
          this.attachCartListeners();
          this._cartListenersAttached = true;
        }
      }
    },

    loadScenarios: async function () {
      try {
        const res = await window.originalFetch(
          `/api/scenarios/active?page=${encodeURIComponent(
            window.location.pathname
          )}&group=${this.experimentGroup}&t=${Date.now()}`
        );
        const allScenarios = await res.json();
        this.scenarios = Array.isArray(allScenarios)
          ? allScenarios.filter((s) => s.enabled === 1)
          : [];
        console.log(`📦 Yüklendi (${window.location.pathname}):`, this.scenarios.map((s) => s.name));
      } catch (e) {}
    },

    startScenarioWatcher: function () {
      setInterval(() => {
        const now = Date.now();
        if (this.lastScenarioTime && now - this.lastScenarioTime < this.COOLDOWN_MS) return;
        if (now - this.pageLoadTime < 1000) return;

        const shuffled = [...this.scenarios].sort(() => Math.random() - 0.5);

        for (const scenario of shuffled) {
          if (scenario.type === "search_irrelevant" && !window.location.search.includes("search=")) {
            continue;
          }

          let effectiveProbability = Number(scenario.probability || 0);
          if (this.experimentGroup === "control") effectiveProbability = 0;
          else if (this.experimentGroup === "variant_a") effectiveProbability *= 0.3;
          else if (this.experimentGroup === "variant_b") effectiveProbability *= 0.6;
          // variant_c = tam olasılık

          if (Math.random() <= effectiveProbability) {
            this.executeScenario(scenario);
            break;
          }
        }
      }, 1000);
    },

    executeScenario: function (scenario) {
      if (scenario.selector && scenario.selector !== "") {
        const elements = document.querySelectorAll(scenario.selector);
        if (elements.length === 0) {
          scenario.retryCount = (scenario.retryCount || 0) + 1;
          if (scenario.retryCount <= 10) {
            setTimeout(() => this.executeScenario(scenario), 500);
            return;
          } else {
            scenario.retryCount = 0;
            return;
          }
        }
      }

      const now = Date.now();
      if (this.lastScenarioTime && now - this.lastScenarioTime < this.COOLDOWN_MS) return;

      scenario.retryCount = 0;
      this.triggeredScenarios.set(scenario.id, now);
      this.lastScenarioTime = now;

      let params = {};
      try {
        params = JSON.parse(scenario.params || "{}");
      } catch (e) {
        params = {};
      }

      console.log("⚡️ ÇALIŞIYOR:", scenario.name);

      // Basit marker
      try {
        window.originalFetch("http://127.0.0.1:5001/send_negative_trigger", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ scenario: scenario.name }),
        }).catch(() => {});
      } catch (err) {
        console.warn("Python marker sunucusuna ulaşılamadı.");
      }

      // LSL / detaylı marker
      try {
        window.originalFetch("http://127.0.0.1:5001/send_negative_trigger", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            scenario_name: scenario.name,
            scenario_type: scenario.type,
            session_id: this.sessionId,
            experiment_group: this.experimentGroup,
            phase: this.phase,
            page_url: window.location.href,
            timestamp: Date.now(),
          }),
        }).catch(() => {});
      } catch (err) {
        console.warn("Python marker sunucusuna ulaşılamadı.");
      }

      // Veritabanı loglaması
      window.originalFetch("/api/scenarios/trigger", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sessionId: this.sessionId,
          scenarioId: scenario.id,
          status: "triggered",
          details: { name: scenario.name, type: scenario.type, timestamp: now },
        }),
      }).catch(() => {});

      switch (scenario.type) {
        case "slow_image":
          this.slowImageLoad(scenario.selector, params.delay || 3000);
          break;
        case "broken_image":
          this.brokenImage(scenario.selector);
          break;
        case "skeleton_prolong":
          this.skeletonProlong(scenario.selector, params.delay || 3000);
          break;
        case "search_irrelevant":
          this.searchIrrelevant(params.duration || 5000);
          break;
        case "button_delay":
          this.buttonDelay(scenario.selector, params.delay || 4000);
          break;
        case "first_click_miss":
          this.firstClickMiss(scenario.selector);
          break;
        case "feedback_late":
          this.inputLag(2000);
          break;
        case "network_jitter":
          this.networkJitter(params.delay || 2000);
          break;
        case "overlay_blocking":
          this.overlayBlocking(params.duration || 4000);
          break;
        case "price_change":
          this.priceChangeWarning(params.change_percent || 5);
          break;
        case "coupon_min_spend":
          this.couponError("coupon_min_spend");
          break;
        case "coupon_expired":
          this.couponError("coupon_expired");
          break;
        case "facet_reset_once":
          this.resetFilters();
          break;
        case "sort_reset":
          window.dispatchEvent(new Event("sort:reset"));
          break;
      }
    },

    attachCartListeners: function () {
      window.addEventListener("cart:refresh", () => {
        console.log("😈 Sepet eklendi! Şoklama yapılıyor...");
        this.overlayBlocking(3000);
        this.networkJitter(4000);
      });
    },

    initMouseTracking: function () {
      document.addEventListener("mousemove", (e) => {
        const now = performance.now();
        if (now - this.lastMouseTime > this.MOUSE_THROTTLE_MS) {
          this.mouseTrajectory.push({
            x: e.clientX,
            y: e.clientY,
            t: now - this.experimentStartTime,
          });
          this.lastMouseTime = now;
          if (this.mouseTrajectory.length > 500) this.mouseTrajectory.shift();
        }
      });

      document.addEventListener("click", (e) => {
        this.logEvent("mouse_click", {
          x: e.clientX,
          y: e.clientY,
          target: e.target.tagName,
          className: e.target.className,
        });
      });

      document.addEventListener("scroll", () => {
        if (performance.now() - this.lastMouseTime > 500) {
          this.logEvent("scroll", { scrollY: window.scrollY });
          this.lastMouseTime = performance.now();
        }
      });
    },

    slowImageLoad: function (selector, delay) {
      const allImages = Array.from(document.querySelectorAll(selector || "img"));
      if (allImages.length === 0) return;

      const selectedImages = allImages.sort(() => 0.5 - Math.random()).slice(0, 5);
      selectedImages.forEach((img) => {
        const originalSrc = img.src;
        img.style.transition = "all 0.5s ease";
        img.style.filter = "blur(20px) grayscale(100%)";
        img.style.opacity = "0.3";
        img.style.transform = "scale(0.95)";
        setTimeout(() => {
          img.src = originalSrc + (originalSrc.includes("?") ? "&" : "?") + "t=" + Date.now();
          img.onload = () => {
            img.style.filter = "none";
            img.style.opacity = "1";
            img.style.transform = "scale(1)";
          };
        }, delay);
      });
    },

    networkJitter: function (delay) {
      if (window.fetch !== window.originalFetch) return;
      const baseDelay = Math.max(delay, 2000);
      console.log(`🐌 AĞ ÇÖKTÜ: ${baseDelay}ms`);
      document.body.style.cursor = "progress";

      window.fetch = function (...args) {
        const url = args[0] ? args[0].toString() : "";
        if (
          url.includes("_next") ||
          url.includes("/api/events") ||
          url.includes("/api/scenarios")
        ) {
          return window.originalFetch.apply(this, args);
        }
        let dynamicDelay = baseDelay;
        if (url.includes("/api/cart") || url.includes("/api/products")) dynamicDelay = baseDelay * 3;

        return new Promise((resolve) => {
          setTimeout(() => {
            resolve(window.originalFetch.apply(this, args));
          }, dynamicDelay);
        });
      };

      setTimeout(() => {
        window.fetch = window.originalFetch;
        document.body.style.cursor = "default";
      }, 10000);
    },

    overlayBlocking: function (duration) {
      const old = document.getElementById("blocking-overlay");
      if (old) old.remove();

      const overlay = document.createElement("div");
      overlay.id = "blocking-overlay";
      overlay.style.cssText = `
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        background: rgba(0, 0, 0, 0.85);
        z-index: 2147483647;
        pointer-events: auto;
        cursor: not-allowed;
        display: flex; align-items: center; justify-content: center; flex-direction: column;
      `;

      overlay.innerHTML = `
        <div style="background:white; padding:40px; border-radius:12px; text-align:center;">
          <div style="width:60px; height:60px; border:6px solid #f3f3f3; border-top:6px solid #ef4444; border-radius:50%; animation:spin 1s linear infinite; margin:0 auto 20px;"></div>
          <h2 style="margin:0 0 10px; color:#1f2937; font-size:20px; font-weight:bold;">Connection Lost</h2>
          <p style="margin:0; color:#6b7280;">Re-establishing secure connection...</p>
        </div>
        <style>@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }</style>
      `;

      document.body.appendChild(overlay);
      setTimeout(() => {
        if (overlay.parentNode) overlay.remove();
      }, duration);
    },

    resetFilters: function () {
      document.body.style.cursor = "wait";
      setTimeout(() => {
        document.querySelectorAll('input[type="radio"]').forEach((radio) => {
          radio.checked = radio.value === "all";
        });
        document.querySelectorAll('input[type="search"], input[type="text"]').forEach((input) => {
          input.value = "";
        });
        const url = new URL(window.location.href);
        url.searchParams.delete("category");
        url.searchParams.delete("search");
        window.history.replaceState({}, "", url.toString());
        window.dispatchEvent(new Event("filters:reset"));
        document.body.style.cursor = "default";
        this.showToast("⚠️ Filters reset to All.", "info");
        window.scrollTo(0, 0);
      }, 800);
    },

    inputLag: function (duration) {
      const inputs = document.querySelectorAll('input[type="text"], input[type="search"]');
      inputs.forEach((input) => {
        if (input.dataset.lag === "true") return;
        input.dataset.lag = "true";

        let timer = null;
        const handler = (e) => {
          if (e.key.length === 1) {
            e.preventDefault();
            clearTimeout(timer);
            timer = setTimeout(() => {
              input.value += e.key;
            }, 500);
          }
        };

        input.addEventListener("keydown", handler);
        this.showToast("Keyboard input latency detected.", "warning");

        setTimeout(() => {
          input.removeEventListener("keydown", handler);
          input.dataset.lag = "false";
        }, duration);
      });
    },

    buttonDelay: function (selector, delay) {
      const buttons = document.querySelectorAll(selector || ".add-to-cart");
      buttons.forEach((btn) => {
        if (btn.dataset.broken === "true") return;
        btn.dataset.broken = "true";
        const txt = btn.innerText;

        const handler = function (e) {
          e.preventDefault();
          e.stopImmediatePropagation();
          btn.disabled = true;
          btn.style.cursor = "not-allowed";
          btn.style.opacity = "0.7";
          btn.innerText = "Stuck...";

          setTimeout(() => {
            btn.disabled = false;
            btn.style.cursor = "pointer";
            btn.style.opacity = "1";
            btn.innerText = txt;
            btn.dataset.broken = "false";
            btn.removeEventListener("click", handler, true);
          }, delay);
        };

        btn.addEventListener("click", handler, true);
      });
    },

    skeletonProlong: function (selector, delay) {
      const cards = document.querySelectorAll(selector || ".product-card");
      if (cards.length === 0) return;

      cards.forEach((card) => {
        const mask = document.createElement("div");
        mask.style.cssText =
          "position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: #e5e7eb; opacity: 0.9; z-index: 10; display:flex; align-items:center; justify-content:center; color:#666; font-size:12px;";
        mask.innerText = "Loading...";
        const originalPos = card.style.position;
        card.style.position = "relative";
        card.appendChild(mask);
        setTimeout(() => {
          mask.remove();
          card.style.position = originalPos;
        }, delay);
      });
    },

    searchIrrelevant: function () {
      const products = document.querySelectorAll(".product-card");
      if (products.length < 2) return;
      const parent = products[0].parentNode;
      const shuffled = Array.from(products).sort(() => Math.random() - 0.5);
      shuffled.forEach((node) => parent.appendChild(node));
      this.showToast("Search index corrupted.", "warning");
    },

    brokenImage: function (selector) {
      const images = document.querySelectorAll(selector || "img");
      if (images.length > 0) {
        const randomImg = images[Math.floor(Math.random() * images.length)];
        if (randomImg) {
          randomImg.removeAttribute("src");
          randomImg.removeAttribute("srcset");
          randomImg.style.backgroundColor = "#fee2e2";
          randomImg.style.border = "2px dashed #ef4444";
          randomImg.style.minHeight = "150px";
          randomImg.setAttribute("alt", "BROKEN_ASSET_404");
        }
      }
    },

    firstClickMiss: function (selector) {
      const buttons = document.querySelectorAll(selector || "button");
      buttons.forEach((btn) => {
        if (btn.dataset.miss === "true") return;
        btn.dataset.miss = "true";

        const handler = (e) => {
          e.preventDefault();
          e.stopImmediatePropagation();
          btn.style.transform = "translate(15px, 15px)";
          setTimeout(() => {
            btn.style.transform = "none";
          }, 200);
        };

        btn.addEventListener("click", handler, true);
        setTimeout(() => {
          btn.removeEventListener("click", handler, true);
          btn.dataset.miss = "false";
        }, 500);
      });
    },

    priceChangeWarning: function () {
      if (!window.location.pathname.includes("cart")) return;
      const banner = document.createElement("div");
      banner.className = "bg-red-50 text-red-700 p-4 mb-4 rounded border border-red-200 font-bold";
      banner.innerText = "⚠️ UYARI: Sepet toplamı döviz dalgalanması nedeniyle güncellendi.";
      const main = document.querySelector("main");
      if (main) main.insertBefore(banner, main.firstChild);
    },

    couponError: function (type) {
      this.showToast(type === "coupon_expired" ? "Code Expired" : "⚠️ Kupon Süresi Dolmuş", "error");
    },

    showToast: function (message, type) {
      const toast = document.createElement("div");
      toast.style.cssText = `position:fixed; top:20px; right:20px; padding:15px 25px; background:${
        type === "error" ? "#dc2626" : "#d97706"
      }; color:white; border-radius:8px; z-index:99999; font-family:sans-serif; font-weight:bold; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);`;
      toast.innerText = message;
      document.body.appendChild(toast);
      setTimeout(() => toast.remove(), 4000);
    },

    trackPageView: function () {
      this.logEvent("page_view", { url: window.location.href });
    },

    logEvent: function (eventType, eventData) {
      this.eventQueue.push({
        sessionId: this.sessionId,
        experimentGroup: this.experimentGroup,
        phase: this.phase || "task",
        eventType: eventType,
        eventData: eventData,
        pageUrl: window.location.href,
        timestamp: Date.now(),
        relative_t_ms: performance.now() - this.experimentStartTime,
      });
    },

    flushEvents: async function () {
      if (this.mouseTrajectory.length > 0) {
        this.logEvent("mouse_trajectory", {
          path: [...this.mouseTrajectory],
        });
        this.mouseTrajectory = [];
      }

      if (this.eventQueue.length === 0) return;

      const events = [...this.eventQueue];
      this.eventQueue = [];

      try {
        await window.originalFetch("/api/events/batch", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(events),
        });
      } catch (e) {}
    },

    threeDSSoftFail: function () {},
    paymentTimeout: function () {},
  };
})();