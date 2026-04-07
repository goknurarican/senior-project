// global.d.ts
export {};

declare global {
  interface Window {
    __originalAlert__?: typeof window.alert;
    __FEEDBACK_LATE__?: boolean;
    originalFetch?: typeof fetch;

    ExperimentSDK?: {
      sessionId: string | null;
      experimentGroup: string | null;
      phase?: string | null;

      scenarios: any[];
      triggeredScenarios: Map<number, number>;
      eventQueue: any[];

      pageLoadTime: number;
      lastScenarioTime: number;

      mouseTrajectory: {
        x: number;
        y: number;
        t: number;
      }[];

      lastMouseTime: number;
      MOUSE_THROTTLE_MS: number;
      experimentStartTime: number;
      COOLDOWN_MS: number;
      MAX_SCENARIOS_PER_SESSION: number;

      init: () => Promise<void>;
      loadScenarios: () => Promise<void>;
      startScenarioWatcher: () => void;
      executeScenario: (scenario: any) => void;
      attachCartListeners: () => void;
      initMouseTracking: () => void;
      flushEvents: () => Promise<void>;

      slowImageLoad: (selector: string, delay: number) => void;
      networkJitter: (delay: number) => void;
      overlayBlocking: (duration: number) => void;
      resetFilters: () => void;
      inputLag: (duration: number) => void;
      buttonDelay: (selector: string, delay: number) => void;
      skeletonProlong: (selector: string, delay: number) => void;
      searchIrrelevant: (duration: number) => void;
      brokenImage: (selector: string) => void;
      firstClickMiss: (selector: string) => void;
      priceChangeWarning: (changePercent: number) => void;
      couponError: (type: string) => void;
      showToast: (message: string, type: string) => void;
      trackPageView: () => void;
      logEvent: (eventType: string, eventData: any) => void;
      threeDSSoftFail: () => void;
      paymentTimeout: () => void;
    };
  }
}