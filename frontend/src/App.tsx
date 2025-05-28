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
                智能学习推荐系统
              </h1>
              <p className="text-muted-foreground text-lg">
                基于深度学习的个性化题目推荐与学习预测平台
              </p>
            </div>
            <div className="flex items-center space-x-2">
              {apiStatus === 'checking' && (
                <>
                  <Activity className="h-5 w-5 text-yellow-500 animate-pulse" />
                  <span className="text-sm text-muted-foreground">检查API状态...</span>
                </>
              )}
              {apiStatus === 'online' && (
                <>
                  <CheckCircle className="h-5 w-5 text-green-500" />
                  <span className="text-sm text-green-600">API在线</span>
                </>
              )}
              {apiStatus === 'offline' && (
                <>
                  <XCircle className="h-5 w-5 text-red-500" />
                  <span className="text-sm text-red-600">API离线</span>
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
            智能学习推荐系统 - 基于深度学习的个性化教育解决方案
          </p>
          <p className="mt-1">
            使用 React + TypeScript + shadcn/ui + Recharts 构建
          </p>
        </div>
      </div>
    </div>
  );
}

export default App;
