import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Activity, Calendar, History, Play, TrendingUp, Database, BarChart3, Settings, Info, Zap, Clock, Target } from 'lucide-react';
import { useState, useEffect } from 'react';
import Dashboard from './pages/Dashboard';
import HistoryPage from './pages/HistoryPage';
import SchedulerPage from './pages/SchedulerPage';
import AnalyzePage from './pages/AnalyzePage';
import DataSyncPage from './pages/DataSyncPage';
import BacktestPage from './pages/BacktestPage';
import OptimizationPage from './pages/OptimizationPage';
import PaperTradingPage from './pages/PaperTradingPage';
import ExperimentsPage from './pages/ExperimentsPage';
import AboutPage from './pages/AboutPage';

function Navigation() {
  const location = useLocation();
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  const navItems = [
    { path: '/', label: 'Dashboard', icon: TrendingUp },
    { path: '/analyze', label: 'Analyze', icon: Play },
    { path: '/backtest', label: 'Backtest', icon: BarChart3 },
    { path: '/optimize', label: 'Optimize', icon: Settings },
    { path: '/paper', label: 'Paper', icon: Zap },
    { path: '/experiments', label: 'Experiments', icon: Target },
    { path: '/history', label: 'History', icon: History },
    { path: '/scheduler', label: 'Scheduler', icon: Calendar },
    { path: '/data-sync', label: 'Data', icon: Database },
    { path: '/about', label: 'About', icon: Info },
  ];

  const formatTime = (date) => {
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false
    });
  };

  const getTimezone = () => {
    const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
    const offset = -currentTime.getTimezoneOffset() / 60;
    const offsetStr = offset >= 0 ? `+${offset}` : offset;
    return `${timezone} (UTC${offsetStr})`;
  };

  return (
    <nav className="bg-gray-800 border-b border-gray-700 relative">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center space-x-2">
            <Activity className="w-8 h-8 text-brand-500" />
            <span className="text-xl font-bold text-gray-100">
              Temporal Trading Agents
            </span>
          </div>

          <div className="flex space-x-0.5">
            {navItems.map(({ path, label, icon: Icon }) => (
              <Link
                key={path}
                to={path}
                className={`flex items-center space-x-1.5 px-3 py-2 rounded-lg transition-colors text-sm ${
                  location.pathname === path
                    ? 'bg-brand-900 text-brand-300 border border-brand-700'
                    : 'text-gray-300 hover:bg-gray-700'
                }`}
              >
                <Icon className="w-4 h-4" />
                <span className="font-medium">{label}</span>
              </Link>
            ))}
          </div>
        </div>

        {/* Clock positioned below nav bar on the right */}
        <div className="absolute right-4 top-16 flex items-center space-x-1.5 px-2 py-1 bg-gray-700/50 rounded border border-gray-600 text-xs">
          <Clock className="w-3 h-3 text-brand-400" />
          <span className="font-mono text-gray-200">
            {formatTime(currentTime)}
          </span>
          <span className="text-gray-500">•</span>
          <span className="text-gray-500">
            {getTimezone()}
          </span>
        </div>
      </div>
    </nav>
  );
}

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-900">
        <Navigation />

        <main className="max-w-7xl mx-auto px-4 py-8">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/analyze" element={<AnalyzePage />} />
            <Route path="/backtest" element={<BacktestPage />} />
            <Route path="/optimize" element={<OptimizationPage />} />
            <Route path="/paper" element={<PaperTradingPage />} />
            <Route path="/experiments" element={<ExperimentsPage />} />
            <Route path="/history" element={<HistoryPage />} />
            <Route path="/scheduler" element={<SchedulerPage />} />
            <Route path="/data-sync" element={<DataSyncPage />} />
            <Route path="/about" element={<AboutPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
