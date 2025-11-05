// public/injection-sdk.js
(function() {
  'use strict';

  window.ExperimentSDK = {
    sessionId: null,
    experimentGroup: null,
    scenarios: [],
    triggeredScenarios: new Set(),
    eventQueue: [],
    
    init: async function() {
      // Get session info
      const res = await fetch('/api/session/info');
      const data = await res.json();
      this.sessionId = data.sessionId;
      this.experimentGroup = data.experimentGroup;
      
      // Only load scenarios for non-control groups
      if (this.experimentGroup !== 'control') {
        await this.loadScenarios();
        this.startScenarioWatcher();
      }
      
      // Start event tracking
      this.trackPageView();
      this.attachEventListeners();
      
      // Flush events every 2 seconds
      setInterval(() => this.flushEvents(), 2000);
      
      console.log('SDK Initialized:', this.experimentGroup);
    },
    
    loadScenarios: async function() {
      const res = await fetch(`/api/scenarios/active?page=${encodeURIComponent(window.location.pathname)}`);
      this.scenarios = await res.json();
    },
    
    startScenarioWatcher: function() {
      // Check time-based scenarios every second
      setInterval(() => {
        const now = Date.now();
        const pageLoadTime = performance.timing.navigationStart;
        const timeSinceLoad = now - pageLoadTime;
        
        this.scenarios.forEach(scenario => {
          if (this.triggeredScenarios.has(scenario.id)) return;
          
          // Random probability check
          if (Math.random() > scenario.probability) return;
          
          // Time-based trigger (after 5 seconds on page)
          if (timeSinceLoad > 5000) {
            this.executeScenario(scenario);
          }
        });
      }, 1000);
    },
    
    executeScenario: function(scenario) {
      if (this.triggeredScenarios.has(scenario.id)) return;
      this.triggeredScenarios.add(scenario.id);
      
      const params = JSON.parse(scenario.params || '{}');
      
      console.log('Executing scenario:', scenario.name);
      this.logEvent('scenario_start', { scenario_id: scenario.id, type: scenario.type });
      
      switch(scenario.type) {
        case 'slow_image':
          this.slowImageLoad(scenario.selector, params.delay);
          break;
        case 'button_delay':
          this.buttonDelay(scenario.selector, params.delay);
          break;
        case 'search_irrelevant':
          this.searchIrrelevant(params.duration);
          break;
        case 'price_change':
          this.priceChangeWarning(params.change_percent);
          break;
        case '3ds_soft_fail':
          this.threeDSSoftFail();
          break;
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
      
      setTimeout(() => {
        this.logEvent('scenario_end', { scenario_id: scenario.id, type: scenario.type });
      }, params.delay || params.duration || 2000);
    },
    
    slowImageLoad: function(selector, delay) {
      const images = document.querySelectorAll(selector || '.product-image');
      images.forEach((img, index) => {
        if (index < 2) { // Only affect first 2 images
          const originalSrc = img.src;
          img.style.filter = 'blur(8px)';
          img.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="400" height="300"%3E%3Crect fill="%23ddd" width="400" height="300"/%3E%3Ctext fill="%23999" x="50%25" y="50%25" text-anchor="middle" dy=".3em"%3ELoading...%3C/text%3E%3C/svg%3E';
          
          setTimeout(() => {
            img.src = originalSrc;
            img.style.filter = 'none';
          }, delay);
        }
      });
    },
    
    buttonDelay: function(selector, delay) {
      const buttons = document.querySelectorAll(selector || '.add-to-cart');
      buttons.forEach(btn => {
        const originalClick = btn.onclick;
        btn.onclick = function(e) {
          e.preventDefault();
          btn.disabled = true;
          btn.style.opacity = '0.5';
          const originalText = btn.textContent;
          btn.textContent = 'Processing...';
          
          setTimeout(() => {
            btn.disabled = false;
            btn.style.opacity = '1';
            btn.textContent = originalText;
            if (originalClick) originalClick.call(btn, e);
          }, delay);
        };
      });
    },
    
    searchIrrelevant: function(duration) {
      // Temporarily scramble search results
      const products = document.querySelectorAll('.product-card');
      const parent = products[0]?.parentNode;
      if (!parent) return;
      
      const shuffled = Array.from(products).sort(() => Math.random() - 0.5);
      shuffled.forEach(p => parent.appendChild(p));
      
      // Show notification
      this.showToast('Updating search results...', 'info');
      
      // Restore after duration
      setTimeout(() => {
        const resorted = Array.from(products).sort((a, b) => {
          return parseInt(a.dataset.order || '0') - parseInt(b.dataset.order || '0');
        });
        resorted.forEach(p => parent.appendChild(p));
      }, duration);
    },
    
    priceChangeWarning: function(changePercent) {
      if (window.location.pathname === '/checkout') {
        this.showToast(`Price updated: ${changePercent}% change detected`, 'warning', 5000);
      }
    },
    
    threeDSSoftFail: function() {
      window.ExperimentSDK.firstPaymentAttempt = true;
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
  
  // Auto-init on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => window.ExperimentSDK.init());
  } else {
    window.ExperimentSDK.init();
  }
})();
