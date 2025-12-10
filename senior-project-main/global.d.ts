// global.d.ts
export {};

declare global {
  interface Window {
    ExperimentSDK?: {
      init: () => Promise<void>;
      // .entries() kullandığın için bunun bir Map olduğunu varsayıyorum
      triggeredScenarios: Map<string, any>;
      executeScenario: (scenario: any) => void;
    };
  }
}
