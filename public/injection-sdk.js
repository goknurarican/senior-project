// public/injection-sdk.js
(function() {
  'use strict';

  window.ExperimentSDK = {
    sessionId: null,
    experimentGroup: null,
    scenarios: [],
    triggeredScenarios: new Map(), // Changed to Map to track cooldown times
    eventQueue: [],
    pageLoadTime: Date.now(),
    scenariosPerSession: 0,
    lastScenarioTime: 0,
    COOLDOWN_MS: 12000, // 12 seconds cooldown between scenarios
    MAX_SCENARIOS_PER_SESSION: 10,
    
    init: async function() {
      // Get session info




      const res = await fetch('/api/session/info');
const data = await res.json();
this.sessionId = data.sessionId || null;
this.experimentGroup = data.experimentGroup || 'control';

if (!this.sessionId) {
  console.error('[SDK] /api/session/info sessionId döndürmedi:', data);
} else {
  console.log('🚀 SDK Initialized:', {
    sessionId: this.sessionId.substring(0, 8),
    group: this.experimentGroup
  });
}


// Eğer sessionId yoksa senaryo çalıştırma (sadece event loglasın)
if (!this.sessionId) {
  console.warn('[SDK] No sessionId returned from /api/session/info, staying in control mode');
  this.experimentGroup = 'control';
}

      // Load scenarios based on experiment group
      if (this.experimentGroup !== 'control') {
        await this.loadScenarios();
        this.startScenarioWatcher();
      }
      
      // Start event tracking
      this.trackPageView();
      this.attachEventListeners();
      
      // Flush events every 2 seconds
      setInterval(() => this.flushEvents(), 2000);
    },
    
    
    loadScenarios: async function() {
      const res = await fetch(`/api/scenarios/active?page=${encodeURIComponent(window.location.pathname)}&group=${this.experimentGroup}`);
      const allScenarios = await res.json();
      
      // Filter only enabled scenarios
      this.scenarios = allScenarios.filter(s => s.enabled === 1);
      
      console.log('📦 Loaded scenarios:', this.scenarios.map(s => ({
        name: s.name,
        probability: s.probability,
        enabled: s.enabled
      })));
    },
    
    startScenarioWatcher: function() {
      // Check for scenario triggers every second
      setInterval(() => {
        const now = Date.now();
        const timeSinceLoad = now - this.pageLoadTime;
        
        // Check cooldown
        if (this.lastScenarioTime && (now - this.lastScenarioTime) < this.COOLDOWN_MS) {
          return; // Still in cooldown
        }
        
        // Check max scenarios per session
        if (this.scenariosPerSession >= this.MAX_SCENARIOS_PER_SESSION) {
          return;
        }
        
        // Try to execute scenarios based on time and probability
        this.scenarios.forEach(scenario => {
          // Skip if already triggered
          if (this.triggeredScenarios.has(scenario.id)) {
            return;
          }
          
          // Skip if disabled
          if (!scenario.enabled) {
            return;
          }
          
          // Check probability based on experiment group
          let effectiveProbability = scenario.probability;
          
          // Adjust probability based on experiment group
          if (this.experimentGroup === 'control') {
            effectiveProbability = 0; // Control group gets no scenarios
          } else if (this.experimentGroup === 'variant_a') {
            effectiveProbability = scenario.probability * 0.3; // Low tier: 30% of original
          } else if (this.experimentGroup === 'variant_b') {
            effectiveProbability = scenario.probability * 0.6; // Medium tier: 60% of original
          } else if (this.experimentGroup === 'variant_c') {
            effectiveProbability = scenario.probability; // Full tier: 100% of original
          }

          // Random check
          if (Math.random() > effectiveProbability) {
            return;
          }

          // Time-based trigger (after 3 seconds on page)
          if (timeSinceLoad > 3000) {
            this.executeScenario(scenario);
          }
        });
      }, 1000);
    },

    executeScenario: function(scenario) {
      if (this.triggeredScenarios.has(scenario.id)) return;

      // Check cooldown and limits
      const now = Date.now();
      if (this.lastScenarioTime && (now - this.lastScenarioTime) < this.COOLDOWN_MS) {
        return;
      }

      if (this.scenariosPerSession >= this.MAX_SCENARIOS_PER_SESSION) {
        return;
      }

      this.triggeredScenarios.set(scenario.id, now);
      this.scenariosPerSession++;
      this.lastScenarioTime = now;

      const params = JSON.parse(scenario.params || '{}');

      console.log('⚡ Executing scenario:', scenario.name, params);
      this.logEvent('scenario_start', {
        scenario_id: scenario.id,
        type: scenario.type,
        name: scenario.name
      });

      // Execute based on scenario type
      switch(scenario.type) {
        // Loading/Visual scenarios
        case 'slow_image':
          this.slowImageLoad(scenario.selector, params.delay || 1500);
          break;
        case 'broken_image':
          this.brokenImage(scenario.selector);
          break;
        case 'skeleton_prolong':
          this.skeletonProlong(scenario.selector, params.delay || 2000);
          break;

        // Interaction/Friction scenarios
        case 'button_delay':
          this.buttonDelay(scenario.selector, params.delay || 1200);
          break;
        case 'first_click_miss':
          this.firstClickMiss(scenario.selector);
          break;
        case 'feedback_late':
          this.feedbackLate(params.delay || 1500);
          break;

        // Search/Navigation scenarios
        case 'search_irrelevant':
          this.searchIrrelevant(params.duration || 5000);
          break;
        case 'facet_reset_once':
        case 'sort_reset':
          this.resetFilters();
          break;

        // Cart/Checkout scenarios
        case 'price_change':
          this.priceChangeWarning(params.change_percent || 5);
          break;
        case 'coupon_min_spend':
        case 'coupon_expired':
          this.couponError(scenario.type);
          break;

        // Payment scenarios
        case '3ds_soft_fail':
          this.threeDSSoftFail();
          break;
        case 'payment_retry_timeout':
          this.paymentTimeout();
          break;

        // Overlay scenarios
        case 'overlay_blocking':
          this.overlayBlocking(params.duration || 4000);
          break;

        // Network scenarios
        case 'network_jitter':
          this.networkJitter(params.delay || 500);
          break;

        default:
          console.warn('Unknown scenario type:', scenario.type);
      }

      // Log scenario trigger
      fetch('/api/scenarios/trigger', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionId: this.sessionId,
          scenarioId: scenario.id,
          status: 'triggered'
        })
      });

      // Log scenario end after duration
      setTimeout(() => {
        this.logEvent('scenario_end', {
          scenario_id: scenario.id,
          type: scenario.type,
          name: scenario.name
        });
      }, params.delay || params.duration || 2000);
    },


    // Scenario Implementations
    slowImageLoad: function(selector, delay) {
      const images = document.querySelectorAll(selector || '.product-image, img');
      images.forEach((img, index) => {
        if (index < 3) { // Affect first 3 images
          const originalSrc = img.src;
          img.classList.add('blur-sm', 'animate-pulse');
          img.style.filter = 'blur(8px)';

          setTimeout(() => {
            img.src = originalSrc + '?t=' + Date.now(); // Force reload
            img.style.filter = 'none';
            img.classList.remove('blur-sm', 'animate-pulse');
          }, delay);
        }
      });
    },

    brokenImage: function(selector) {
      const images = document.querySelectorAll(selector || '.product-image');
      if (images.length > 0) {
        const randomImg = images[Math.floor(Math.random() * Math.min(3, images.length))];
        randomImg.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="400" height="300"%3E%3Crect fill="%23f3f4f6" width="400" height="300"/%3E%3Ctext fill="%23999" x="50%25" y="50%25" text-anchor="middle" dy=".3em"%3EImage not available%3C/text%3E%3C/svg%3E';
      }
    },

    skeletonProlong: function(selector, delay) {
      const elements = document.querySelectorAll(selector || '.product-card');
      elements.forEach(el => {
        el.style.opacity = '0.5';
        el.classList.add('animate-pulse');
        setTimeout(() => {
          el.style.opacity = '1';
          el.classList.remove('animate-pulse');
        }, delay);
      });
    },

    buttonDelay: function(selector, delay) {
      const buttons = document.querySelectorAll(selector || '.add-to-cart, button[type="submit"]');
      buttons.forEach(btn => {
        const originalHandler = btn.onclick;
        btn.onclick = function(e) {
          e.preventDefault();
          e.stopPropagation();

          // Visual feedback
          btn.disabled = true;
          btn.style.opacity = '0.6';
          btn.style.cursor = 'wait';
          const originalText = btn.textContent;
          btn.textContent = 'Processing...';

          setTimeout(() => {
            btn.disabled = false;
            btn.style.opacity = '1';
            btn.style.cursor = 'pointer';
            btn.textContent = originalText;

            // Trigger original action
            if (originalHandler) {
              originalHandler.call(btn, e);
            } else {
              btn.click();
            }
          }, delay);

          return false;
        };
      });
    },

    firstClickMiss: function(selector) {
      const buttons = document.querySelectorAll(selector || 'button');
      buttons.forEach(btn => {
        let isFirstClick = true;
        const originalHandler = btn.onclick;

        btn.onclick = function(e) {
          if (isFirstClick) {
            e.preventDefault();
            e.stopPropagation();
            isFirstClick = false;

            // Visual hint
            btn.style.transform = 'scale(0.98)';
            setTimeout(() => {
              btn.style.transform = '';
            }, 200);

            return false;
          }

          if (originalHandler) {
            return originalHandler.call(btn, e);
          }
        };
      });
    },

    feedbackLate: function(delay) {
      // Intercept toast notifications
      const originalAlert = window.alert;
      window.alert = function(message) {
        setTimeout(() => {
          originalAlert(message);
        }, delay);
      };
    },

    searchIrrelevant: function(duration) {
      const products = document.querySelectorAll('.product-card');
      const parent = products[0]?.parentNode;
      if (!parent) return;

      // Save original order
      const originalOrder = Array.from(products).map((p, i) => {
        p.dataset.originalOrder = i;
        return p;
      });

      // Shuffle
      const shuffled = [...originalOrder].sort(() => Math.random() - 0.5);
      shuffled.forEach(p => parent.appendChild(p));

      this.showToast('Updating search results...', 'info', duration);

      // Restore after duration
      setTimeout(() => {
        originalOrder.forEach(p => parent.appendChild(p));
      }, duration);
    },

    resetFilters: function() {
      // Reset checkboxes and radios
      document.querySelectorAll('input[type="checkbox"], input[type="radio"]').forEach(input => {
        if (input.name.includes('filter') || input.name.includes('category')) {
          input.checked = false;
        }
      });

      // Reset select dropdowns
      document.querySelectorAll('select').forEach(select => {
        select.selectedIndex = 0;
      });

      this.showToast('Filters reset', 'warning', 10000);
      console.log('resetFilters çalıştı')
    },

    priceChangeWarning: function(changePercent) {
      if (window.location.pathname.includes('checkout') || window.location.pathname.includes('cart')) {
        const banner = document.createElement('div');
        banner.className = 'bg-yellow-50 border-l-4 border-yellow-400 p-4 mb-4';
        banner.innerHTML = `
          <div class="flex">
            <div class="flex-shrink-0">⚠️</div>
            <div class="ml-3">
              <p class="text-sm text-yellow-700">
                Price updated: ${changePercent}% change detected since you added items to cart.
              </p>
            </div>
          </div>
        `;

        const container = document.querySelector('main');
        if (container) {
          container.insertBefore(banner, container.firstChild);
          setTimeout(() => banner.remove(), 5000);
        }
      }
    },

    couponError: function(type) {
      const couponInput = document.querySelector('input[placeholder*="Coupon"]');
      if (couponInput) {
        const parent = couponInput.parentElement;
        const error = document.createElement('div');
        error.className = 'text-red-500 text-sm mt-1';
        error.textContent = type === 'coupon_min_spend'
          ? 'Minimum spend of ₺500 required for this coupon'
          : 'This coupon has expired';
        parent.appendChild(error);

        setTimeout(() => error.remove(), 3000);
      }
    },

    threeDSSoftFail: function() {
      window.ExperimentSDK.firstPaymentAttempt = true;
      console.log('💳 3DS will fail on first attempt');
    },

    paymentTimeout: function() {
      // Intercept fetch for payment endpoints
      const originalFetch = window.fetch;
      let isFirst = true;

      window.fetch = function(...args) {
        if (args[0].includes('checkout') && isFirst) {
          isFirst = false;
          return new Promise((resolve, reject) => {
            setTimeout(() => {
              reject(new Error('Request timeout'));
            }, 1500);
          });
        }
        return originalFetch.apply(this, args);
      };
    },

    overlayBlocking: function(duration) {
      const overlay = document.createElement('div');
      overlay.className = 'fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center';
      overlay.innerHTML = `
        <div class="bg-white rounded-lg p-8 max-w-md mx-4">
          <h2 class="text-2xl font-bold mb-4">Special Offer!</h2>
          <p class="text-gray-600 mb-4">Get 10% off your first order with code WELCOME10</p>
          <button class="close-overlay bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700">
            Continue Shopping
          </button>
        </div>
      `;

      document.body.appendChild(overlay);

      const closeBtn = overlay.querySelector('.close-overlay');
      closeBtn.onclick = () => overlay.remove();

      // Auto-close after duration
      setTimeout(() => {
        if (overlay.parentNode) {
          overlay.remove();
        }
      }, duration);
    },

    networkJitter: function(delay) {
      // Add artificial delay to all network requests
      const originalFetch = window.fetch;
      window.fetch = function(...args) {
        return new Promise((resolve) => {
          setTimeout(() => {
            resolve(originalFetch.apply(this, args));
          }, delay);
        });
      };

      // Reset after 10 seconds
      setTimeout(() => {
        window.fetch = originalFetch;
      }, 10000);
    },

    showToast: function(message, type = 'info', duration = 3000) {
      const toast = document.createElement('div');
      toast.className = `fixed top-4 right-4 px-6 py-3 rounded-lg shadow-lg text-white z-50 ${
        type === 'warning' ? 'bg-yellow-500' : 
        type === 'error' ? 'bg-red-500' : 
        'bg-blue-500'
      }`;
      toast.textContent = message;
      document.body.appendChild(toast);

      setTimeout(() => {
        toast.remove();
      }, duration);
    },

    trackPageView: function() {
      this.logEvent('page_view', {
        url: window.location.href,
        referrer: document.referrer,
        title: document.title
      });
    },

    attachEventListeners: function() {
      // Click tracking
      document.addEventListener('click', (e) => {
        const target = e.target;
        if (target.tagName === 'BUTTON' || target.tagName === 'A') {
          this.logEvent('click', {
            element: target.tagName,
            text: target.textContent?.substring(0, 50),
            className: target.className,
            href: target.href
          });
        }
      });

      // Scroll tracking
      let scrollTimer;
      window.addEventListener('scroll', () => {
        clearTimeout(scrollTimer);
        scrollTimer = setTimeout(() => {
          this.logEvent('scroll', {
            scrollY: window.scrollY,
            scrollHeight: document.documentElement.scrollHeight,
            percentage: (window.scrollY / document.documentElement.scrollHeight) * 100
          });
        }, 500);
      });
    },

    logEvent: function(eventType, eventData) {
      this.eventQueue.push({
        sessionId: this.sessionId,
        eventType: eventType,
        eventData: eventData,
        pageUrl: window.location.href,
        timestamp: Date.now()
      });
    },

    flushEvents: async function() {
      if (this.eventQueue.length === 0) return;

      const events = [...this.eventQueue];
      this.eventQueue = [];

      try {
        await fetch('/api/events/batch', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(events)
        });
      } catch (error) {
        // Re-add events to queue on failure
        this.eventQueue.push(...events);
      }
    }
  };

  // Don't auto-init - let the app call init() explicitly
})();