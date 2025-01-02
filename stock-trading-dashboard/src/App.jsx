import { useState, useEffect } from 'react'
import Dashboard from './components/Dashboard'

const API_URL = 'http://localhost:5000';

function App() {
  const [modelResults, setModelResults] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(null);

  const fetchModelResults = async () => {
    try {
      const response = await fetch(`${API_URL}/api/model-results`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      setModelResults(data);
      setError(null);
      
      // Fetch cache status
      const statusResponse = await fetch(`${API_URL}/api/cache-status`);
      if (statusResponse.ok) {
        const statusData = await statusResponse.json();
        setLastUpdate(new Date(statusData.last_update));
      }
    } catch (err) {
      console.error('Fetch error:', err);
      setError('Failed to fetch model results. Please ensure the backend server is running.');
    } finally {
      setLoading(false);
    }
  };

  const forceUpdate = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_URL}/api/force-update`, {
        method: 'POST'
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      await fetchModelResults();
    } catch (err) {
      console.error('Update error:', err);
      setError('Failed to update model results.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchModelResults();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-900"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
          <strong className="font-bold">Error: </strong>
          <span className="block sm:inline">{error}</span>
        </div>
      </div>
    );
  }

  
  return (
    <div>
      <Dashboard modelResults={modelResults} />
    </div>
  );
}

export default App;