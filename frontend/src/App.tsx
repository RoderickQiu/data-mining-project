import { useState, useEffect } from 'react';
import { apiService } from './services/api';
import { Activity, CheckCircle, XCircle } from 'lucide-react';
import MainTab from './components/MainTab';

function App() {
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');

  useEffect(() => {
    const checkApiStatus = async () => {
      try {
        await apiService.healthCheck();
        setApiStatus('online');
      } catch (error) {
        console.error(error);
        setApiStatus('offline');
      }
    };

    checkApiStatus();
    const interval = setInterval(checkApiStatus, 30000); // Check every 30 seconds

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold text-foreground mb-2">
                Intelligent Learning Recommendation System
              </h1>
              <p className="text-muted-foreground text-lg">
                Personalized question recommendation and learning prediction platform based on deep learning
              </p>
            </div>
            <div className="flex items-center space-x-2">
              {apiStatus === 'checking' && (
                <>
                  <Activity className="h-5 w-5 text-yellow-500 animate-pulse" />
                  <span className="text-sm text-muted-foreground">Checking API status...</span>
                </>
              )}
              {apiStatus === 'online' && (
                <>
                  <CheckCircle className="h-5 w-5 text-green-500" />
                  <span className="text-sm text-green-600">API Online</span>
                </>
              )}
              {apiStatus === 'offline' && (
                <>
                  <XCircle className="h-5 w-5 text-red-500" />
                  <span className="text-sm text-red-600">API Offline</span>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Main Content */}
        <MainTab />

        {/* Footer */}
        <div className="mt-12 text-center text-sm text-muted-foreground">
          <p>
            Intelligent Learning Recommendation System - Personalized education solution based on deep learning
          </p>
          <p className="mt-1">
            Built with React + TypeScript + shadcn/ui + Recharts
          </p>
        </div>
      </div>
    </div>
  );
}

export default App;
