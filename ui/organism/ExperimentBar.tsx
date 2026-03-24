// ui/organism/ExperimentBar.tsx
// Bu dosyayı mevcut ExperimentBar ile değiştir.
// Layout.tsx'de zaten import ediliyor: import ExperimentBar from '../ui/organism/ExperimentBar';

import { useState, useEffect } from "react";

export default function ExperimentBar() {
  const [phase, setPhase] = useState<string | null>(null);
  const [group, setGroup] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch("/api/session/info")
      .then((r) => r.json())
      .then((data) => {
        setPhase(data.phase || "control");
        setGroup(data.assignedVariant || data.experimentGroup || "unknown");
      })
      .catch(() => setPhase("control"));
  }, []);

  const changePhase = async (action: string) => {
    setLoading(true);

    try {
      const res = await fetch("/api/experiment/phase", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action }),
      });
      const data = await res.json();

      if (data.status === "success") {
        if (action === "start_variant") {
          // Phase 2 başlıyor — sayfayı yenile ki SDK yeni phase'i alsın
          // ve senaryoları yüklesin
          alert(`Phase 2 başladı: ${data.phase}. Sayfa yenilenecek.`);
          window.location.reload();
        } else if (action === "end_experiment") {
          // Deney bitti — login sayfasına yönlendir (cookie'ler silindi)
          alert("Deney tamamlandı.");
          window.location.href = "/signup";
        }
      } else {
        alert("Hata: " + (data.error || "Bilinmeyen hata"));
        setLoading(false);
      }
    } catch (e) {
      alert("API hatası: " + e);
      setLoading(false);
    }
  };

  // Yükleniyor veya completed ise gösterme
  if (!phase || phase === "completed") return null;

  // Admin sayfalarında gösterme
  if (typeof window !== "undefined" && window.location.pathname.startsWith("/admin")) return null;

  return (
    <div
      style={{
        position: "fixed",
        bottom: 0,
        left: 0,
        right: 0,
        background: "#1e293b",
        padding: "10px 20px",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        zIndex: 9999,
        borderTop: "2px solid #3b82f6",
      }}
    >
      <div style={{ color: "#94a3b8", fontSize: 13 }}>
        <span style={{ marginRight: 16 }}>
          Phase: <strong style={{ color: phase === "control" ? "#22c55e" : "#ef4444" }}>{phase}</strong>
        </span>
        <span>
          Group: <strong style={{ color: "#60a5fa" }}>{group}</strong>
        </span>
      </div>

      {phase === "control" && (
        <button
          onClick={() => changePhase("start_variant")}
          disabled={loading}
          style={{
            padding: "8px 20px",
            fontSize: 14,
            fontWeight: 600,
            background: loading ? "#6b7280" : "#3b82f6",
            color: "white",
            border: "none",
            borderRadius: 6,
            cursor: loading ? "wait" : "pointer",
          }}
        >
          {loading ? "Switching..." : "End Phase 1 & Start Phase 2"}
        </button>
      )}

      {phase !== "control" && phase !== "completed" && (
        <button
          onClick={() => changePhase("end_experiment")}
          disabled={loading}
          style={{
            padding: "8px 20px",
            fontSize: 14,
            fontWeight: 600,
            background: loading ? "#6b7280" : "#dc2626",
            color: "white",
            border: "none",
            borderRadius: 6,
            cursor: loading ? "wait" : "pointer",
          }}
        >
          {loading ? "Ending..." : "End Experiment"}
        </button>
      )}
    </div>
  );
}