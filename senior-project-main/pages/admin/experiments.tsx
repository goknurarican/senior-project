// pages/admin/experiments.tsx
import { useEffect, useState } from 'react';
import Link from 'next/link';

export default function AdminExperiments() {
  const [stats, setStats] = useState<any>({});
  const [groupStats, setGroupStats] = useState<any>([]);

  useEffect(() => {
    fetchExperimentData();
  }, []);

  const fetchExperimentData = async () => {
    const res = await fetch('/api/admin/experiments/stats');
    const data = await res.json();
    setStats(data.overall);
    setGroupStats(data.groups);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <h1 className="text-2xl font-bold">Experiments</h1>
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
            <Link href="/admin/experiments" className="text-yellow-400">Experiments</Link>
            <Link href="/admin/sessions" className="hover:text-gray-300">Sessions</Link>
          </nav>
        </div>
      </div>
      
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Experiment Configuration */}
        <div className="bg-white rounded-lg shadow mb-8">
          <div className="px-6 py-4 border-b">
            <h2 className="text-lg font-semibold">Current Experiment Configuration</h2>
          </div>
          <div className="p-6">
            <div className="grid grid-cols-4 gap-6">
              <div className="border rounded-lg p-4">
                <h3 className="font-semibold mb-2">Control Group</h3>
                <p className="text-3xl font-bold text-gray-900">25%</p>
                <p className="text-sm text-gray-600 mt-1">No scenarios (0% probability)</p>
              </div>
              <div className="border rounded-lg p-4">
                <h3 className="font-semibold mb-2">Variant A (Low)</h3>
                <p className="text-3xl font-bold text-green-600">25%</p>
                <p className="text-sm text-gray-600 mt-1">30% probability tier</p>
              </div>
              <div className="border rounded-lg p-4">
                <h3 className="font-semibold mb-2">Variant B (Medium)</h3>
                <p className="text-3xl font-bold text-blue-600">25%</p>
                <p className="text-sm text-gray-600 mt-1">60% probability tier</p>
              </div>
              <div className="border rounded-lg p-4">
                <h3 className="font-semibold mb-2">Variant C (Full)</h3>
                <p className="text-3xl font-bold text-purple-600">25%</p>
                <p className="text-sm text-gray-600 mt-1">100% probability tier</p>
              </div>
            </div>
          </div>
        </div>

        {/* Group Statistics */}
        <div className="bg-white rounded-lg shadow">
          <div className="px-6 py-4 border-b">
            <h2 className="text-lg font-semibold">Group Performance Metrics</h2>
          </div>
          <div className="p-6">
            <table className="w-full">
              <thead>
                <tr className="text-left text-sm text-gray-500 border-b">
                  <th className="pb-2">Group</th>
                  <th className="pb-2">Sessions</th>
                  <th className="pb-2">Avg Events/Session</th>
                  <th className="pb-2">Cart Adds</th>
                  <th className="pb-2">Scenarios Triggered</th>
                </tr>
              </thead>
              <tbody>
                {groupStats.map((group: any) => (
                  <tr key={group.group} className="border-t">
                    <td className="py-3">
                      <span className={`px-2 py-1 rounded text-sm ${
                        group.group_name === 'control' ? 'bg-gray-100' :
                        group.group_name === 'variant_a' ? 'bg-green-100' :
                        group.group_name === 'variant_b' ? 'bg-blue-100' :
                        'bg-purple-100'
                      }`}>
                        {group.group_name}
                      </span>
                    </td>
                    <td className="py-3">{group.session_count || 0}</td>
                    <td className="py-3">{group.avg_events || 0}</td>
                    <td className="py-3">{group.cart_adds || 0}</td>
                    <td className="py-3">{group.scenarios_triggered || 0}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Overall Stats */}
        <div className="mt-8 grid grid-cols-4 gap-6">
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-sm text-gray-500 mb-2">Total Sessions</h3>
            <p className="text-3xl font-bold">{stats.totalSessions || 0}</p>
          </div>
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-sm text-gray-500 mb-2">Avg Session Duration</h3>
            <p className="text-3xl font-bold">{stats.avgDuration || '0'}s</p>
          </div>
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-sm text-gray-500 mb-2">Conversion Rate</h3>
            <p className="text-3xl font-bold">{stats.conversionRate || '0'}%</p>
          </div>
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-sm text-gray-500 mb-2">Total Revenue</h3>
            <p className="text-3xl font-bold">₺{stats.totalRevenue || '0'}</p>
          </div>
        </div>
      </div>
    </div>
  );
}
