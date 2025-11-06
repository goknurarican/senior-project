// pages/test-scenarios.tsx
import Layout from '../components/Layout';
import { useEffect, useState } from 'react';

export default function TestScenarios() {
  const [scenarios, setScenarios] = useState([]);
  const [sessionInfo, setSessionInfo] = useState<any>(null);
  const [triggeredScenarios, setTriggeredScenarios] = useState<any[]>([]);
  
  useEffect(() => {
    fetchData();
    
    // Monitor triggered scenarios from SDK
    const interval = setInterval(() => {
      if (window.ExperimentSDK) {
        const triggered = Array.from(window.ExperimentSDK.triggeredScenarios.entries());
        setTriggeredScenarios(triggered);
      }
    }, 1000);
    
    return () => clearInterval(interval);
  }, []);
  
  const fetchData = async () => {
    // Get session info
    const sessionRes = await fetch('/api/session/info');
    const sessionData = await sessionRes.json();
    setSessionInfo(sessionData);
    
    // Get active scenarios
    const scenariosRes = await fetch(`/api/scenarios/active?page=/test-scenarios&group=${sessionData.experimentGroup}`);
    const scenariosData = await scenariosRes.json();
    setScenarios(scenariosData);
  };
  
  const forceExecuteScenario = (scenario: any) => {
    if (window.ExperimentSDK) {
      window.ExperimentSDK.executeScenario(scenario);
    }
  };
  
  return (
    <Layout>
      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">Scenario Testing Dashboard</h1>
        
        {/* Session Info */}
        <div className="bg-blue-50 border-l-4 border-blue-500 p-4 mb-8">
          <h2 className="font-bold mb-2">Session Information</h2>
          <p>Session ID: <code className="bg-gray-100 px-2 py-1 rounded">{sessionInfo?.sessionId?.substring(0, 16)}...</code></p>
          <p>Experiment Group: <span className={`px-3 py-1 rounded inline-block mt-1 ${
            sessionInfo?.experimentGroup === 'control' ? 'bg-gray-200' :
            sessionInfo?.experimentGroup === 'variant_a' ? 'bg-green-200' :
            sessionInfo?.experimentGroup === 'variant_b' ? 'bg-blue-200' :
            'bg-purple-200'
          }`}>{sessionInfo?.experimentGroup}</span></p>
          <p className="text-sm text-gray-600 mt-2">
            {sessionInfo?.experimentGroup === 'control' && 'No scenarios will trigger (control group)'}
            {sessionInfo?.experimentGroup === 'variant_a' && 'Low probability tier (30%)'}
            {sessionInfo?.experimentGroup === 'variant_b' && 'Medium probability tier (60%)'}
            {sessionInfo?.experimentGroup === 'variant_c' && 'Full probability tier (100%)'}
          </p>
        </div>
        
        {/* Triggered Scenarios */}
        <div className="bg-green-50 border-l-4 border-green-500 p-4 mb-8">
          <h2 className="font-bold mb-2">Triggered Scenarios ({triggeredScenarios.length})</h2>
          {triggeredScenarios.length > 0 ? (
            <ul className="space-y-1">
              {triggeredScenarios.map(([id, time]) => (
                <li key={id} className="text-sm">
                  Scenario #{id} - Triggered at {new Date(time).toLocaleTimeString()}
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-gray-600">No scenarios triggered yet (wait 3+ seconds or click Force Execute)</p>
          )}
        </div>
        
        {/* Test Elements */}
        <div className="grid grid-cols-2 gap-6 mb-8">
          {/* Product Card for visual scenarios */}
          <div className="bg-white rounded-lg shadow p-4 product-card">
            <img 
              src="https://picsum.photos/400/300?random=test" 
              alt="Test Product"
              className="w-full h-48 object-cover rounded product-image mb-4"
            />
            <h3 className="font-semibold mb-2">Test Product</h3>
            <p className="text-gray-600 mb-3">This is a test product for scenario testing</p>
            <button className="w-full bg-blue-600 text-white py-2 rounded hover:bg-blue-700 add-to-cart">
              Add to Cart
            </button>
          </div>
          
          {/* Form for interaction scenarios */}
          <div className="bg-white rounded-lg shadow p-4">
            <h3 className="font-semibold mb-4">Test Form</h3>
            <form onSubmit={(e) => { e.preventDefault(); alert('Form submitted!'); }}>
              <input 
                type="text" 
                placeholder="Coupon Code"
                className="w-full px-3 py-2 border rounded mb-3"
              />
              <button type="submit" className="w-full bg-green-600 text-white py-2 rounded hover:bg-green-700">
                Apply Coupon
              </button>
            </form>
          </div>
        </div>
        
        {/* Available Scenarios */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-bold mb-4">Available Scenarios</h2>
          <div className="overflow-x-auto">
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
                {scenarios.map((scenario: any) => (
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
                      <span className={`px-2 py-1 rounded text-xs ${
                        scenario.enabled ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                      }`}>
                        {scenario.enabled ? 'Yes' : 'No'}
                      </span>
                    </td>
                    <td className="py-2">
                      <button
                        onClick={() => forceExecuteScenario(scenario)}
                        disabled={!scenario.enabled}
                        className={`px-3 py-1 rounded text-xs ${
                          scenario.enabled 
                            ? 'bg-blue-600 text-white hover:bg-blue-700' 
                            : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                        }`}
                      >
                        Force Execute
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        
        {/* Instructions */}
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
