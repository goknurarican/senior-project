// pages/admin/index.tsx
import { useEffect, useState } from 'react';
import Link from 'next/link';

export default function Admin() {
  const [stats, setStats] = useState<any>({});
  const [recentEvents, setRecentEvents] = useState([]);
  const [scenarios, setScenarios] = useState([]);
  
  useEffect(() => {
    fetchDashboardData();
  }, []);
  
  const fetchDashboardData = async () => {
    // Fetch stats
    const statsRes = await fetch('/api/admin/stats');
    setStats(await statsRes.json());
    
    // Fetch recent events
    const eventsRes = await fetch('/api/admin/events/recent');
    setRecentEvents(await eventsRes.json());
    
    // Fetch scenarios
    const scenariosRes = await fetch('/api/admin/scenarios');
    setScenarios(await scenariosRes.json());
  };
  
  const toggleScenario = async (scenarioId: number, enabled: boolean) => {
    await fetch(`/api/admin/scenarios/${scenarioId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled: !enabled })
    });
    fetchDashboardData();
  };
  
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <h1 className="text-2xl font-bold">Admin Panel</h1>
            <Link href="/" className="text-blue-600 hover:text-blue-800">
              Back to Shop
            </Link>
          </div>
        </div>
      </div>
      
      {/* Navigation */}
      <div className="bg-gray-800 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <nav className="flex space-x-8 py-4">
            <Link href="/admin" className="hover:text-gray-300">Dashboard</Link>
            <Link href="/admin/scenarios" className="hover:text-gray-300">Scenarios</Link>
            <Link href="/admin/experiments" className="hover:text-gray-300">Experiments</Link>
            <Link href="/admin/sessions" className="hover:text-gray-300">Sessions</Link>
          </nav>
        </div>
      </div>
      
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Stats Cards */}
        <div className="grid grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-sm text-gray-500 mb-2">Total Sessions</h3>
            <p className="text-3xl font-bold">{stats.totalSessions || 0}</p>
          </div>
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-sm text-gray-500 mb-2">Total Events</h3>
            <p className="text-3xl font-bold">{stats.totalEvents || 0}</p>
          </div>
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-sm text-gray-500 mb-2">Active Scenarios</h3>
            <p className="text-3xl font-bold">{stats.activeScenarios || 0}</p>
          </div>
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-sm text-gray-500 mb-2">Triggered Today</h3>
            <p className="text-3xl font-bold">{stats.triggeredToday || 0}</p>
          </div>
        </div>
        
        <div className="grid grid-cols-2 gap-8">
          {/* Scenarios Management */}
          <div className="bg-white rounded-lg shadow">
            <div className="px-6 py-4 border-b">
              <h2 className="text-lg font-semibold">Scenario Management</h2>
            </div>
            <div className="p-6">
              <table className="w-full">
                <thead>
                  <tr className="text-left text-sm text-gray-500">
                    <th className="pb-2">Scenario</th>
                    <th className="pb-2">Type</th>
                    <th className="pb-2">Probability</th>
                    <th className="pb-2">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {scenarios.map((scenario: any) => (
                    <tr key={scenario.id} className="border-t">
                      <td className="py-2">{scenario.name}</td>
                      <td className="py-2 text-sm text-gray-600">{scenario.type}</td>
                      <td className="py-2 text-sm">{(scenario.probability * 100).toFixed(0)}%</td>
                      <td className="py-2">
                        <button
                          onClick={() => toggleScenario(scenario.id, scenario.enabled)}
                          className={`px-3 py-1 rounded text-sm ${
                            scenario.enabled 
                              ? 'bg-green-100 text-green-800' 
                              : 'bg-gray-100 text-gray-800'
                          }`}>
                          {scenario.enabled ? 'Enabled' : 'Disabled'}
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          
          {/* Recent Events */}
          <div className="bg-white rounded-lg shadow">
            <div className="px-6 py-4 border-b">
              <h2 className="text-lg font-semibold">Recent Events</h2>
            </div>
            <div className="p-6">
              <div className="space-y-2">
                {recentEvents.map((event: any) => (
                  <div key={event.id} className="flex justify-between text-sm py-2 border-b">
                    <span className="font-medium">{event.event_type}</span>
                    <span className="text-gray-500">
                      Session: {event.session_id?.substring(0, 8)}...
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
