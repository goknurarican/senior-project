// pages/admin/scenarios.tsx
import { useEffect, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/router';

export default function AdminScenarios() {
  const [scenarios, setScenarios] = useState([]);
  const [showAddForm, setShowAddForm] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    type: 'slow_image',
    target_page: '*',
    selector: '',
    params: '{}',
    probability: 0.1
  });
  const router = useRouter();

  useEffect(() => {
    checkAuth();
    fetchScenarios();
  }, []);

  const checkAuth = async () => {
    const res = await fetch('/api/auth/me');
    if (!res.ok) {
      router.push('/login');
    }
  };

  const fetchScenarios = async () => {
    const res = await fetch('/api/admin/scenarios');
    setScenarios(await res.json());
  };

  const toggleScenario = async (scenarioId: number, enabled: boolean) => {
    await fetch(`/api/admin/scenarios/${scenarioId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled: !enabled })
    });
    fetchScenarios();
  };

  const deleteScenario = async (scenarioId: number) => {
    if (!confirm('Are you sure you want to delete this scenario?')) return;
    
    await fetch(`/api/admin/scenarios/${scenarioId}`, {
      method: 'DELETE'
    });
    fetchScenarios();
  };

  const addScenario = async (e: React.FormEvent) => {
    e.preventDefault();
    
    await fetch('/api/admin/scenarios', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ...formData,
        probability: parseFloat(formData.probability.toString())
      })
    });
    
    setShowAddForm(false);
    setFormData({
      name: '',
      type: 'slow_image',
      target_page: '*',
      selector: '',
      params: '{}',
      probability: 0.1
    });
    fetchScenarios();
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <h1 className="text-2xl font-bold">Scenario Management</h1>
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
            <Link href="/admin/scenarios" className="text-yellow-400">Scenarios</Link>
            <Link href="/admin/experiments" className="hover:text-gray-300">Experiments</Link>
            <Link href="/admin/sessions" className="hover:text-gray-300">Sessions</Link>
          </nav>
        </div>
      </div>
      
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="bg-white rounded-lg shadow">
          <div className="px-6 py-4 border-b flex justify-between items-center">
            <h2 className="text-lg font-semibold">All Scenarios</h2>
            <button
              onClick={() => setShowAddForm(!showAddForm)}
              className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
            >
              + Add New Scenario
            </button>
          </div>
          
          {showAddForm && (
            <div className="p-6 bg-gray-50 border-b">
              <form onSubmit={addScenario} className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <input
                    placeholder="Scenario Name"
                    value={formData.name}
                    onChange={(e) => setFormData({...formData, name: e.target.value})}
                    className="px-3 py-2 border rounded"
                    required
                  />
                  <select
                    value={formData.type}
                    onChange={(e) => setFormData({...formData, type: e.target.value})}
                    className="px-3 py-2 border rounded"
                  >
                    <option value="slow_image">Slow Image</option>
                    <option value="button_delay">Button Delay</option>
                    <option value="search_irrelevant">Search Irrelevant</option>
                    <option value="3ds_soft_fail">3DS Soft Fail</option>
                    <option value="price_change">Price Change</option>
                  </select>
                  <input
                    placeholder="Target Page (e.g. /products)"
                    value={formData.target_page}
                    onChange={(e) => setFormData({...formData, target_page: e.target.value})}
                    className="px-3 py-2 border rounded"
                  />
                  <input
                    placeholder="CSS Selector (optional)"
                    value={formData.selector}
                    onChange={(e) => setFormData({...formData, selector: e.target.value})}
                    className="px-3 py-2 border rounded"
                  />
                  <input
                    placeholder="Params JSON"
                    value={formData.params}
                    onChange={(e) => setFormData({...formData, params: e.target.value})}
                    className="px-3 py-2 border rounded"
                  />
                  <input
                    type="number"
                    step="0.01"
                    min="0"
                    max="1"
                    placeholder="Probability (0-1)"
                    value={formData.probability}
                    onChange={(e) => setFormData({...formData, probability: parseFloat(e.target.value)})}
                    className="px-3 py-2 border rounded"
                  />
                </div>
                <div className="flex gap-2">
                  <button
                    type="submit"
                    className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700"
                  >
                    Save Scenario
                  </button>
                  <button
                    type="button"
                    onClick={() => setShowAddForm(false)}
                    className="bg-gray-600 text-white px-4 py-2 rounded hover:bg-gray-700"
                  >
                    Cancel
                  </button>
                </div>
              </form>
            </div>
          )}
          
          <div className="p-6">
            <table className="w-full">
              <thead>
                <tr className="text-left text-sm text-gray-500 border-b">
                  <th className="pb-2">ID</th>
                  <th className="pb-2">Name</th>
                  <th className="pb-2">Type</th>
                  <th className="pb-2">Target</th>
                  <th className="pb-2">Probability</th>
                  <th className="pb-2">Status</th>
                  <th className="pb-2">Actions</th>
                </tr>
              </thead>
              <tbody>
                {scenarios.map((scenario: any) => (
                  <tr key={scenario.id} className="border-t">
                    <td className="py-2 text-sm">{scenario.id}</td>
                    <td className="py-2">{scenario.name}</td>
                    <td className="py-2 text-sm text-gray-600">{scenario.type}</td>
                    <td className="py-2 text-sm">{scenario.target_page}</td>
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
                    <td className="py-2">
                      <button
                        onClick={() => deleteScenario(scenario.id)}
                        className="text-red-600 hover:text-red-800 text-sm"
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
