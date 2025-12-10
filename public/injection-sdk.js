// public/injection-sdk.js
(function() {
  'use strict';

  window.originalFetch = window.fetch;

  window.ExperimentSDK = {
    sessionId: null,
    experimentGroup: null,
    scenarios: [],
    triggeredScenarios: new Map(),
    eventQueue: [],
    pageLoadTime: Date.now(),
    lastScenarioTime: 0,

    // AYARLAR: Çok daha agresif
    COOLDOWN_MS: 800, // Bekleme süresini 0.8 saniyeye düşürdüm (Daha sık saldırı)
    MAX_SCENARIOS_PER_SESSION: 9999,

    init: async function() {
      try {
        const res = await window.originalFetch('/api/session/info');
        const data = await res.json();
        this.sessionId = data.sessionId || null;
        this.experimentGroup = data.experimentGroup || 'control';
      } catch (e) { return; }

      if (this.experimentGroup !== 'control') {
        await this.loadScenarios();
        this.startScenarioWatcher();
        this.attachCartListeners(); // Sepet dinleyicisi

        // Rota değişimi
        window.addEventListener('route:change', async () => {
             // Sayfa değişirken her şeyi temizle
             if (window.fetch !== window.originalFetch) window.fetch = window.originalFetch;
             document.body.style.cursor = 'default';
             const oldOverlay = document.getElementById('blocking-overlay');
             if(oldOverlay) oldOverlay.remove();

             this.pageLoadTime = Date.now();
             await this.loadScenarios();
        });
      }
      window.addEventListener('session:update', async (e) => {
          // Gelen yeni grup bilgisini al
          const newGroup = e.detail?.experimentGroup;

          // Eğer grup değiştiyse veya SDK henüz control modundaysa
          if (newGroup && newGroup !== this.experimentGroup) {
              console.log(`🔄 Session değişti: ${this.experimentGroup} -> ${newGroup}. SDK Yeniden Başlatılıyor...`);

              this.experimentGroup = newGroup;
              this.sessionId = e.detail?.sessionId;

              // Temizlik yap
              if (window.fetch !== window.originalFetch) window.fetch = window.originalFetch;
              document.body.style.cursor = 'default';
              const oldOverlay = document.getElementById('blocking-overlay');
              if(oldOverlay) oldOverlay.remove();

              // Yeniden Yükle
              if (this.experimentGroup !== 'control') {
                  await this.loadScenarios();
                  // Eğer watcher zaten çalışıyorsa tekrar başlatmaya gerek yok,
                  // ama senaryo listesi (this.scenarios) güncellendiği için yeni senaryolar devreye girer.
              }
          }
      });
      this.trackPageView();
      setInterval(() => this.flushEvents(), 3000);
    },

    loadScenarios: async function() {
      try {
        const res = await window.originalFetch(`/api/scenarios/active?page=${encodeURIComponent(window.location.pathname)}&group=${this.experimentGroup}`);
        const allScenarios = await res.json();
        this.scenarios = allScenarios.filter(s => s.enabled === 1);
        console.log(`📦 Yüklendi (${window.location.pathname}):`, this.scenarios.map(s => s.name));
      } catch (e) {}
    },

    startScenarioWatcher: function() {
      setInterval(() => {
        const now = Date.now();
        if (this.lastScenarioTime && (now - this.lastScenarioTime) < this.COOLDOWN_MS) return;
        if (now - this.pageLoadTime < 1000) return;

        // Shuffle (Karıştırma) - Homojenliği engeller
        const shuffled = [...this.scenarios].sort(() => Math.random() - 0.5);

        for (const scenario of shuffled) {
          // Arama kontrolü
          if (scenario.type === 'search_irrelevant' && !window.location.search.includes('search=')) continue;

          let effectiveProbability = scenario.probability;
          if (this.experimentGroup === 'control') effectiveProbability = 0;
          else if (this.experimentGroup === 'variant_a') effectiveProbability = scenario.probability * 0.3;
          else if (this.experimentGroup === 'variant_b') effectiveProbability = scenario.probability * 0.6;
          // Variant C = %100

          if (Math.random() <= effectiveProbability) {
             this.executeScenario(scenario);
             break; // Sadece 1 tane çalıştır
          }
        }
      }, 1000);
    },

    executeScenario: function(scenario) {
      // Retry (Bekleme)
      if (scenario.selector && scenario.selector !== '') {
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
      if (this.lastScenarioTime && (now - this.lastScenarioTime) < this.COOLDOWN_MS) return;

      scenario.retryCount = 0;
      this.triggeredScenarios.set(scenario.id, now);
      this.lastScenarioTime = now;

      const params = JSON.parse(scenario.params || '{}');
      console.log('⚡️ ÇALIŞIYOR:', scenario.name);

      // Logla
      window.originalFetch('/api/scenarios/trigger', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionId: this.sessionId,
          scenarioId: scenario.id,
          status: 'triggered',
          details: { name: scenario.name, type: scenario.type, timestamp: now }
        })
      }).catch(() => {});

      switch(scenario.type) {
        // GÖRSEL
        case 'slow_image':      this.slowImageLoad(scenario.selector, params.delay || 3000); break;
        case 'broken_image':    this.brokenImage(scenario.selector); break;
        case 'skeleton_prolong':this.skeletonProlong(scenario.selector, params.delay || 3000); break;
        case 'search_irrelevant': this.searchIrrelevant(params.duration || 5000); break;

        // ETKİLEŞİM
        case 'button_delay':    this.buttonDelay(scenario.selector, params.delay || 4000); break;
        case 'first_click_miss':this.firstClickMiss(scenario.selector); break;

        // "Feedback Late" yerine artık INPUT LAG var
        case 'feedback_late':   this.inputLag(2000); break;

        // AĞ
        case 'network_jitter':  this.networkJitter(params.delay || 2000); break;
        case 'overlay_blocking':this.overlayBlocking(params.duration || 4000); break;

        // DİĞER
        case 'price_change':    this.priceChangeWarning(params.change_percent || 5); break;
        case 'coupon_min_spend':this.couponError('coupon_min_spend'); break;
        case 'coupon_expired':  this.couponError('coupon_expired'); break;
        case 'facet_reset_once': this.resetFilters(); break;
        case 'sort_reset':       this.resetFilters(); break;
      }
    },

    // --- SEPET TUZAĞI (KESİN ÇALIŞAN VERSİYON) ---
    attachCartListeners: function() {
        window.addEventListener('cart:refresh', () => {
            console.log('😈 Sepet eklendi! Şoklama yapılıyor...');
            // Hem Overlay hem Jitter aynı anda
            this.overlayBlocking(3000);
            this.networkJitter(4000);
        });
    },

    // ----------------------------------------------------
    // İMPLEMENTASYONLAR
    // ----------------------------------------------------

    // 1. SLOW IMAGE (Rastgele İndeksler)
    slowImageLoad: function(selector, delay) {
      const allImages = Array.from(document.querySelectorAll(selector || 'img'));
      if(allImages.length === 0) return;

      // Rastgele karıştır
      const shuffled = allImages.sort(() => 0.5 - Math.random());

      // Rastgele 5 tanesini seç
      const selectedImages = shuffled.slice(0, 5);

      selectedImages.forEach((img) => {
          const originalSrc = img.src;
          img.style.transition = 'all 0.5s ease';
          img.style.filter = 'blur(20px) grayscale(100%)';
          img.style.opacity = '0.3';
          img.style.transform = 'scale(0.95)';

          setTimeout(() => {
            // Cache bust ile resmi gerçekten yeniden yüklet
            img.src = originalSrc + (originalSrc.includes('?') ? '&' : '?') + 't=' + Date.now();
            img.onload = () => {
                img.style.filter = 'none';
                img.style.opacity = '1';
                img.style.transform = 'scale(1)';
            };
          }, delay);
      });
    },

    // 2. NETWORK JITTER (Görsel İmleç Değişimi + Ağır Lag)
    networkJitter: function(delay) {
      if (window.fetch !== window.originalFetch) return;

      const baseDelay = Math.max(delay, 2000);
      console.log(`🐌 AĞ ÇÖKTÜ: ${baseDelay}ms`);

      // KULLANICIYA HİSSETTİR: Mouse'u "Yükleniyor" yap
      document.body.style.cursor = 'progress';

      window.fetch = function(...args) {
        const url = args[0] ? args[0].toString() : '';
        // Sistem dosyalarını koru
        if (url.includes('_next') || url.includes('/api/events') || url.includes('/api/scenarios')) {
            return window.originalFetch.apply(this, args);
        }

        let dynamicDelay = baseDelay;
        // Cart ve Products için aşırı yavaş
        if (url.includes('/api/cart') || url.includes('/api/products')) {
            dynamicDelay = baseDelay * 3;
        }

        return new Promise((resolve) => {
          setTimeout(() => {
              resolve(window.originalFetch.apply(this, args));
          }, dynamicDelay);
        });
      };

      // 10 saniye sonra imleci ve ağı düzelt
      setTimeout(() => {
          window.fetch = window.originalFetch;
          document.body.style.cursor = 'default';
      }, 10000);
    },

    // 3. OVERLAY BLOCKING (Görünürlük Artırıldı)
    overlayBlocking: function(duration) {
      const old = document.getElementById('blocking-overlay');
      if (old) old.remove();

      const overlay = document.createElement('div');
      overlay.id = 'blocking-overlay';
      // CSS: Tam siyah, en üst katman
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
      setTimeout(() => { if (overlay.parentNode) overlay.remove(); }, duration);
    },

    // 4. RESET FILTERS (Daha Şiddetli)
    resetFilters: function() {
        const inputs = document.querySelectorAll('input[type="radio"]:checked, input[type="checkbox"]:checked');
        if (inputs.length === 0) return;

        // "İşlem yapılıyor" gibi gösterip sonra hepsini sil
        document.body.style.cursor = 'wait';

        setTimeout(() => {
            inputs.forEach(input => input.checked = false);
            // Sayfayı en üste fırlat (Disorient)
            window.scrollTo(0, 0);
            document.body.style.cursor = 'default';
            this.showToast('⚠️ Filter service unavailable. Resetting view.', 'error');
        }, 800);
    },

    // 5. INPUT LAG (Yeni Feedback Late)
    inputLag: function(duration) {
        const inputs = document.querySelectorAll('input[type="text"], input[type="search"]');
        inputs.forEach(input => {
            if(input.dataset.lag === 'true') return;
            input.dataset.lag = 'true';

            // Kullanıcı her tuşa bastığında...
            input.addEventListener('keydown', (e) => {
                // Eğer özel tuş değilse (backspace vs.)
                if(e.key.length === 1) {
                    e.preventDefault(); // Yazmayı engelle
                    // 1 saniye sonra yaz
                    setTimeout(() => {
                        input.value += e.key;
                    }, 500);
                }
            });

            this.showToast('Keyboard input latency detected.', 'warning');

            // 5 saniye sonra düzelt
            setTimeout(() => {
                // Event listener'ı tam kaldırmak zor olduğu için sadece görsel uyarı veriyoruz
                // Basitlik adına reload gerekebilir ama şimdilik bu yeterli
            }, 5000);
        });
    },

    buttonDelay: function(selector, delay) {
      const buttons = document.querySelectorAll(selector || '.add-to-cart');
      buttons.forEach(btn => {
        if (btn.dataset.broken === 'true') return;
        const txt = btn.innerText; const handler = btn.onclick;

        btn.dataset.broken = 'true';
        btn.onclick = function(e) {
          e.preventDefault(); e.stopPropagation();
          btn.disabled = true;
          btn.style.cursor = 'not-allowed';
          btn.style.opacity = '0.7';
          btn.innerText = 'Stuck...'; // Mesaj değişti

          setTimeout(() => {
            btn.disabled = false; btn.style.cursor = 'pointer'; btn.style.opacity = '1'; btn.innerText = txt; btn.dataset.broken = 'false';
            if (handler) handler.call(btn, e);
          }, delay);
        };
      });
    },

    // Diğerleri...
    skeletonProlong: function(selector, delay) {
        const cards = document.querySelectorAll(selector || '.product-card');
        if (cards.length === 0) return;
        cards.forEach(card => {
            const mask = document.createElement('div');
            mask.style.cssText = `position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: #e5e7eb; opacity: 0.9; z-index: 10; display:flex; align-items:center; justify-content:center; color:#666; font-size:12px;`;
            mask.innerText = "Loading...";
            const originalPos = card.style.position;
            card.style.position = 'relative';
            card.appendChild(mask);
            setTimeout(() => { mask.remove(); card.style.position = originalPos; }, delay);
        });
    },

    searchIrrelevant: function(duration) {
        const products = document.querySelectorAll('.product-card');
        if (products.length < 2) return;
        const parent = products[0].parentNode;
        const shuffled = Array.from(products).sort(() => Math.random() - 0.5);
        shuffled.forEach(node => parent.appendChild(node));
        this.showToast('Search index corrupted.', 'warning');
    },

    brokenImage: function(selector) {
      const images = document.querySelectorAll(selector || 'img');
      if (images.length > 0) {
        const randomImg = images[Math.floor(Math.random() * images.length)];
        if(randomImg) {
            randomImg.removeAttribute('src');
            randomImg.removeAttribute('srcset');
            randomImg.style.backgroundColor = '#fee2e2'; // Kırmızımsı arka plan
            randomImg.style.border = '2px dashed #ef4444';
            randomImg.style.minHeight = '150px';
            randomImg.setAttribute('alt', 'BROKEN_ASSET_404');
        }
      }
    },

    firstClickMiss: function(selector) {
       const buttons = document.querySelectorAll(selector || 'button');
       buttons.forEach(btn => {
         if (btn.dataset.miss === 'true') return;
         const handler = btn.onclick;
         btn.dataset.miss = 'true';
         btn.onclick = function(e) {
             e.preventDefault(); e.stopPropagation();
             btn.style.transform = 'translate(15px, 15px)'; // Daha fazla kaçsın
             setTimeout(()=> btn.style.transform = 'none', 200);
             btn.onclick = handler;
         }
       });
    },

    priceChangeWarning: function(changePercent) {
      if (!window.location.pathname.includes('cart')) return;
      const banner = document.createElement('div');
      banner.className = 'bg-red-50 text-red-700 p-4 mb-4 rounded border border-red-200 font-bold';
      banner.innerText = `⚠️ SYSTEM ALERT: Cart total updated due to currency fluctuation.`;
      const main = document.querySelector('main');
      if(main) main.insertBefore(banner, main.firstChild);
    },

    couponError: function(type) { this.showToast(type === 'coupon_expired' ? 'Code Expired' : 'Minimum Spend Error', 'error'); },

    showToast: function(message, type) {
        const toast = document.createElement('div');
        toast.style.cssText = `position:fixed; top:20px; right:20px; padding:15px 25px; background:${type==='error'?'#dc2626':'#d97706'}; color:white; border-radius:8px; z-index:99999; font-family:sans-serif; font-weight:bold; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);`;
        toast.innerText = message;
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 4000);
    },

    trackPageView: function() { this.logEvent('page_view', { url: window.location.href }); },
    logEvent: function(type, data) { this.eventQueue.push({ sessionId: this.sessionId, eventType: type, eventData: data, pageUrl: window.location.href, timestamp: Date.now() }); },
    flushEvents: async function() {
      if (this.eventQueue.length === 0) return;
      const events = [...this.eventQueue]; this.eventQueue = [];
      try { await window.originalFetch('/api/events/batch', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(events) }); } catch (e) {}
    },

    threeDSSoftFail: function() {}, paymentTimeout: function() {}
  };
})();