import React from "react";
import { scenarios } from "../../types/types";
type ScenarioTableProps = {
  scenarios: scenarios[];
  forceExecuteScenario: (scenario: scenarios) => void;
};
function ScenarioTable({
  scenarios,
  forceExecuteScenario,
}: ScenarioTableProps) {
  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="border-b">
          <th className="text-left py-2">Name</th>
          <th className="text-left py-2">Type</th>
          <th className="text-left py-2">Target</th>
          <th className="text-left py-2">Probability</th>
          <th className="text-left py-2">Enabled</th>
          <th className="text-left py-2">Action</th>
        </tr>
      </thead>
      <tbody>
        {scenarios.map((scenario: scenarios) => (
          <tr key={scenario.id} className="border-b">
            <td className="py-2">{scenario.name}</td>
            <td className="py-2">
              <code className="bg-gray-100 px-2 py-1 rounded text-xs">
                {scenario.type}
              </code>
            </td>
            <td className="py-2">{scenario.target_page}</td>
            <td className="py-2">
              <span className="font-mono">
                {(scenario.probability * 100).toFixed(0)}%
              </span>
            </td>
            <td className="py-2">
              <span
                className={`px-2 py-1 rounded text-xs ${
                  scenario.enabled
                    ? "bg-green-100 text-green-800"
                    : "bg-red-100 text-red-800"
                }`}
              >
                {scenario.enabled ? "Yes" : "No"}
              </span>
            </td>
            <td className="py-2">
              <button
                onClick={() => forceExecuteScenario(scenario)}
                disabled={!scenario.enabled}
                className={`px-3 py-1 rounded text-xs ${
                  scenario.enabled
                    ? "bg-blue-600 text-white hover:bg-blue-700"
                    : "bg-gray-300 text-gray-500 cursor-not-allowed"
                }`}
              >
                Force Execute
              </button>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default ScenarioTable;
