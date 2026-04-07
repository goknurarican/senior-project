// pages/test-scenarios.tsx
import Layout from "../components/Layout";
import { useEffect, useState } from "react";
import { scenarios, SessionInfo } from "../types/types";
import TestProduct from "../ui/molecule/TestProduct";
import TestForm from "../ui/molecule/TestForm";
import ScenarioTable from "../ui/organism/ScenarioTable";

export default function TestScenarios() {
  const [scenarios, setScenarios] = useState<scenarios[]>([]);
  const [sessionInfo, setSessionInfo] = useState<SessionInfo>(null);
  const [triggeredScenarios, setTriggeredScenarios] = useState<[number, number][]>([]);

  useEffect(() => {
    fetchData();

    const interval = setInterval(() => {
      if (window.ExperimentSDK?.triggeredScenarios) {
        const triggered = Array.from(
          window.ExperimentSDK.triggeredScenarios.entries()
        ) as [number, number][];
        setTriggeredScenarios(triggered);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  const fetchData = async () => {
    const sessionRes: Response = await fetch("/api/session/info");
    const sessionData: SessionInfo = await sessionRes.json();
    setSessionInfo(sessionData);

    const scenariosRes: Response = await fetch(
      `/api/scenarios/active?page=/test-scenarios&group=${sessionData.experimentGroup}`
    );
    const scenariosData = await scenariosRes.json();
    setScenarios(scenariosData);
  };

  const forceExecuteScenario = (scenario: scenarios) => {
    if (window.ExperimentSDK) {
      window.ExperimentSDK.executeScenario(scenario);
    }
  };

  const handleFinishExperiment = async () => {
    try {
      const sessionId = sessionInfo?.sessionId;

      if (!sessionId) {
        alert("Session bulunamadı");
        return;
      }

      const scenarioIds = triggeredScenarios.map(([id]) => Number(id));
      const mouseData = window.ExperimentSDK?.mouseTrajectory || [];
      const eyeData: any[] = [];

      if (window.ExperimentSDK?.flushEvents) {
        await window.ExperimentSDK.flushEvents();
      }

      const response = await fetch("/api/experiment/end", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        credentials: "include",
        body: JSON.stringify({
          sessionId,
          triggeredScenarios: scenarioIds,
          mouseData,
          eyeData,
          finishedAt: Date.now(),
          pageUrl: window.location.href,
        }),
      });

      const result = await response.json();

      if (!result.ok) {
        alert(result.message || "Deney kapatılamadı");
        return;
      }

      if (window.ExperimentSDK?.triggeredScenarios) {
        window.ExperimentSDK.triggeredScenarios.clear();
      }

      if (window.ExperimentSDK?.mouseTrajectory) {
        window.ExperimentSDK.mouseTrajectory = [];
      }

      sessionStorage.clear();
      localStorage.clear();

      alert("Deney başarıyla tamamlandı. Yeni denek için sistem sıfırlanıyor.");
      window.location.reload();
    } catch (error) {
      console.error("Deney bitirme hatası:", error);
      alert("Deney bitirilirken hata oluştu");
    }
  };

  return (
    <Layout>
      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">Scenario Testing Dashboard</h1>

        <div className="bg-blue-50 border-l-4 border-blue-500 p-4 mb-8">
          <h2 className="font-bold mb-2">Session Information</h2>
          <p>
            Session ID:
            <code className="bg-gray-100 px-2 py-1 rounded ml-2">
              {sessionInfo?.sessionId?.substring(0, 16)}...
            </code>
          </p>
          <p>
            Experiment Group:{" "}
            <span
              className={`px-3 py-1 rounded inline-block mt-1 ${
                sessionInfo?.experimentGroup === "control"
                  ? "bg-gray-200"
                  : sessionInfo?.experimentGroup === "variant_a"
                  ? "bg-green-200"
                  : sessionInfo?.experimentGroup === "variant_b"
                  ? "bg-blue-200"
                  : "bg-purple-200"
              }`}
            >
              {sessionInfo?.experimentGroup}
            </span>
          </p>
          <p className="text-sm text-gray-600 mt-2">
            {sessionInfo?.experimentGroup === "control" &&
              "No scenarios will trigger (control group)"}
            {sessionInfo?.experimentGroup === "variant_a" &&
              "Low probability tier (30%)"}
            {sessionInfo?.experimentGroup === "variant_b" &&
              "Medium probability tier (60%)"}
            {sessionInfo?.experimentGroup === "variant_c" &&
              "Full probability tier (100%)"}
          </p>
        </div>

        <div className="mb-8 flex justify-end">
          <button
            onClick={handleFinishExperiment}
            className="bg-red-600 hover:bg-red-700 text-white font-semibold px-5 py-2 rounded-lg shadow"
          >
            Deneyi Bitir
          </button>
        </div>

        <div className="bg-green-50 border-l-4 border-green-500 p-4 mb-8">
          <h2 className="font-bold mb-2">
            Triggered Scenarios ({triggeredScenarios.length})
          </h2>
          {triggeredScenarios.length > 0 ? (
            <ul className="space-y-1">
              {triggeredScenarios.map(([id, time]) => (
                <li key={id} className="text-sm">
                  Scenario #{id} - Triggered at{" "}
                  {new Date(time).toLocaleTimeString()}
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-gray-600">
              No scenarios triggered yet (wait 3+ seconds or click Force Execute)
            </p>
          )}
        </div>

        <div className="grid grid-cols-2 gap-6 mb-8">
          <TestProduct />
          <TestForm />
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-bold mb-4">Available Scenarios</h2>
          <div className="overflow-x-auto">
            <ScenarioTable
              scenarios={scenarios}
              forceExecuteScenario={forceExecuteScenario}
            />
          </div>
        </div>

        <div className="mt-8 bg-gray-50 rounded-lg p-6">
          <h3 className="font-bold mb-2">How to Test:</h3>
          <ol className="list-decimal list-inside space-y-1 text-sm text-gray-700">
            <li>Check your experiment group above (affects probability)</li>
            <li>Wait 3+ seconds for automatic scenario triggers based on probability</li>
            <li>Or click "Force Execute" to manually trigger any enabled scenario</li>
            <li>Watch for visual changes (blurred images, delayed buttons, overlays)</li>
            <li>Check the browser console for detailed logs</li>
            <li>Go to Admin Panel → Scenarios to enable/disable scenarios</li>
          </ol>
        </div>
      </div>
    </Layout>
  );
}