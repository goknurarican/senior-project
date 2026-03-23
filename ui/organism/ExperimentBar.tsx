// ui/organism/ExperimentBar.tsx
import React, { useState, useEffect } from 'react';

export default function ExperimentBar() {
  const [phase, setPhase] = useState<string>("control");
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    // ?t= parametresi ile tarayıcının önbelleğini (cache) deliyoruz
    fetch(`/api/session/info?t=${Date.now()}`)
      .then((r) => r.json())
      .then((data) => {
        if (data.phase) setPhase(data.phase);
      })
      .catch((err) => console.error("Phase fetch error:", err));
  }, []);

  const startPhase2 = async () => {
    const res = await fetch("/api/experiment/phase", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "start_variant" }),
    });
    const data = await res.json();

    if (data.status === "success") {
      alert(`Phase 1 Bitti! Sistem şimdi 2. Aşamaya (${data.phase}) geçiyor.`);
      // Hafızayı tamamen temizlemek ve yeni SDK'yı yüklemek için KESİN YENİLEME:
      window.location.href = "/";
    }
  };

  const endExperiment = async () => {
    const res = await fetch("/api/experiment/phase", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "end_experiment" }),
    });

    if (res.ok) {
      alert("Deney bitti! Katılımınız için teşekkür ederiz.");
      window.location.href = "/signup";
    }
  };

  if (!mounted || phase === "completed") return null;

  return (
    <div className="fixed bottom-0 left-0 w-full bg-slate-900 text-white p-4 flex justify-between items-center z-[99999] shadow-lg border-t-4 border-blue-500">
      <div className="font-bold text-lg">
        Deney Aşaması: <span className="text-blue-400">{phase === "control" ? "Phase 1" : "Phase 2"}</span>
      </div>
      <div>
        {phase === "control" ? (
          <button
            onClick={startPhase2}
            className="bg-blue-600 hover:bg-blue-500 px-6 py-2 rounded font-bold transition-colors shadow-md"
          >
            End Phase 1 & Start Phase 2
          </button>
        ) : (
          <button
            onClick={endExperiment}
            className="bg-green-600 hover:bg-green-500 px-6 py-2 rounded font-bold transition-colors shadow-md"
          >
            End Experiment
          </button>
        )}
      </div>
    </div>
  );
}