import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Activity, Calendar, History, Play, TrendingUp, Database, BarChart3, Settings, Info } from 'lucide-react';
import Dashboard from './pages/Dashboard';
import HistoryPage from './pages/HistoryPage';
import SchedulerPage from './pages/SchedulerPage';
import AnalyzePage from './pages/AnalyzePage';
import DataSyncPage from './pages/DataSyncPage';
import BacktestPage from './pages/BacktestPage';
import OptimizationPage from './pages/OptimizationPage';
import AboutPage from './pages/AboutPage';

function Navigation() {
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Dashboard', icon: TrendingUp },
    { path: '/analyze', label: 'Analyze', icon: Play },
    { path: '/backtest', label: 'Backtest', icon: BarChart3 },
    { path: '/optimize', label: 'Optimize', icon: Settings },
    { path: '/history', label: 'History', icon: History },
    { path: '/scheduler', label: 'Scheduler', icon: Calendar },
    { path: '/data-sync', label: 'Data', icon: Database },
    { path: '/about', label: 'About', icon: Info },
  ];

  return (
    <nav className="bg-gray-800 border-b border-gray-700">
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
