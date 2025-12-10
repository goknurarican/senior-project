// pages/admin/sessions.tsx
import { useEffect, useState } from 'react';
import Link from 'next/link';

export default function AdminSessions() {
  const [sessions, setSessions] = useState([]);
  const [selectedSession, setSelectedSession] = useState<any>(null);
  const [sessionEvents, setSessionEvents] = useState([]);

  useEffect(() => {
    fetchSessions();
  }, []);

  const fetchSessions = async () => {
    const res = await fetch('/api/admin/sessions');
    setSessions(await res.json());
  };

  const viewSessionDetails = async (sessionId: string) => {
    setSelectedSession(sessionId);
    const res = await fetch(`/api/admin/sessions/${sessionId}/events`);
    setSessionEvents(await res.json());
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  const formatRelativeTime = (ms: number) => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    
    if (hours > 0) return `${hours}h ${minutes % 60}m`;
    if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
    return `${seconds}s`;
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <h1 className="text-2xl font-bold">Session Monitoring</h1>
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
            <Link href="/admin/sessions" className="text-yellow-400">Sessions</Link>
          </nav>
        </div>
      </div>
      
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-2 gap-8">
          {/* Sessions List */}
          <div className="bg-white rounded-lg shadow">
            <div className="px-6 py-4 border-b">
              <h2 className="text-lg font-semibold">Recent Sessions</h2>
            </div>
            <div className="p-6 max-h-96 overflow-y-auto">
              <table className="w-full">
                <thead>
                  <tr className="text-left text-sm text-gray-500 border-b">
                    <th className="pb-2">Session ID</th>
                    <th className="pb-2">Group</th>
                    <th className="pb-2">Created</th>
                    <th className="pb-2">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {sessions.map((session: any) => (
                    <tr key={session.id} className="border-t">
                      <td className="py-2 text-xs font-mono">
                        {session.id.substring(0, 8)}...
                      </td>
                      <td className="py-2">
                        <span className={`px-2 py-1 rounded text-xs ${
                          session.experiment_group === 'control' ? 'bg-gray-100' :
                          session.experiment_group === 'variant_a' ? 'bg-blue-100' :
                          'bg-green-100'
                        }`}>
                          {session.experiment_group}
                        </span>
                      </td>
                      <td className="py-2 text-xs">
                        {formatDate(session.created_at)}
                      </td>
                      <td className="py-2">
                        <button
                          onClick={() => viewSessionDetails(session.id)}
                          className="text-blue-600 hover:text-blue-800 text-sm"
                        >
                          View
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Session Events */}
          <div className="bg-white rounded-lg shadow">
            <div className="px-6 py-4 border-b">
              <h2 className="text-lg font-semibold">
                {selectedSession ? `Session Events: ${selectedSession.substring(0, 8)}...` : 'Session Events'}
              </h2>
            </div>
            <div className="p-6 max-h-96 overflow-y-auto">
              {selectedSession ? (
                <div className="space-y-2">
                  {sessionEvents.map((event: any) => (
                    <div key={event.id} className="border-b pb-2">
                      <div className="flex justify-between">
                        <span className="font-semibold text-sm">{event.event_type}</span>
                        <span className="text-xs text-gray-500">
                          +{formatRelativeTime(event.relative_t_ms || 0)}
                        </span>
                      </div>
                      <div className="text-xs text-gray-600 mt-1">
                        {event.page_url}
                      </div>
                      {event.event_type.includes('scenario') && (
                        <div className="text-xs bg-yellow-50 p-1 rounded mt-1">
                          Scenario Event
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-gray-500">Select a session to view events</p>
              )}
            </div>
          </div>
        </div>

        {/* Live Activity Feed */}
        <div className="mt-8 bg-white rounded-lg shadow">
          <div className="px-6 py-4 border-b">
            <h2 className="text-lg font-semibold">Live Activity Feed</h2>
          </div>
          <div className="p-6">
            <div className="space-y-2">
              {sessions.slice(0, 5).map((session: any) => (
                <div key={session.id} className="flex items-center justify-between py-2 border-b">
                  <div className="flex items-center">
                    <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
                    <span className="text-sm">
                      Session <span className="font-mono">{session.id.substring(0, 8)}</span>
                    </span>
                    <span className={`ml-2 px-2 py-1 rounded text-xs ${
                      session.experiment_group === 'control' ? 'bg-gray-100' :
                      session.experiment_group === 'variant_a' ? 'bg-blue-100' :
                      'bg-green-100'
                    }`}>
                      {session.experiment_group}
                    </span>
                  </div>
                  <span className="text-xs text-gray-500">
                    {formatDate(session.created_at)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
