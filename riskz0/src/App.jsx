import { useState, useEffect, useCallback } from "react";
import KpiDashboard from "./components/KpiDashboard";
import RiskScoring from "./components/RiskScoring";
import TrendAnalysis from "./components/TrendAnalysis";
import AlertSystem from "./components/AlertSystem";
import HistoricalReports from "./components/HistoricalReports";
import ManagerInsights from "./components/ManagerInsights";
import ProjectForm from "./components/ProjectForm";
import PredictionResult from "./components/PredictionResult";
import { Shield, RefreshCw, Trash2, Server } from "lucide-react";
import { fetchKpi, fetchProjects, fetchAlerts, fetchInsights, fetchTrends, clearProjects, healthCheck } from "./services/api";
import "./App.css";

function App() {
  const [kpiData, setKpiData] = useState(null);
  const [projects, setProjects] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [insights, setInsights] = useState([]);
  const [trendData, setTrendData] = useState([]);
  const [completionData, setCompletionData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [serverStatus, setServerStatus] = useState("unknown");
  const [lastPrediction, setLastPrediction] = useState(null);

  const loadAllData = useCallback(async () => {
    setLoading(true);
    try {
      const [kpi, proj, alertData, insightData, trendResult] = await Promise.all([
        fetchKpi(),
        fetchProjects(),
        fetchAlerts(),
        fetchInsights(),
        fetchTrends(),
      ]);

      setKpiData(kpi);
      setProjects(proj.projects || []);
      setAlerts(alertData.alerts || []);
      setInsights(insightData.insights || []);
      setTrendData(trendResult.trendData || []);
      setCompletionData(trendResult.completionData || []);
    } catch (err) {
      console.error("Failed to load data:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  const checkServerHealth = useCallback(async () => {
    try {
      const health = await healthCheck();
      setServerStatus(health.status === "healthy" ? "online" : "degraded");
    } catch (err) {
      setServerStatus("offline");
    }
  }, []);

  useEffect(() => {
    checkServerHealth();
    loadAllData();

    const interval = setInterval(checkServerHealth, 30000);
    return () => clearInterval(interval);
  }, [checkServerHealth, loadAllData]);

  const handlePredictionSuccess = (result) => {
    setLastPrediction(result.prediction);
    loadAllData();
  };

  const handleClearAll = async () => {
    if (window.confirm("Are you sure you want to clear all projects?")) {
      await clearProjects();
      loadAllData();
    }
  };

  return (
    <div className="app">
      <header className="navbar">
        <div className="navbar-brand">
          <Shield size={26} className="brand-logo-svg" />
          <h1>RiskZ0</h1>
          <span className="navbar-tag">Risk & Delay Prediction</span>
        </div>
        <div className="navbar-actions">
          <div className={`server-status status-${serverStatus}`}>
            <Server size={14} />
            <span>{serverStatus === "online" ? "Server Online" : serverStatus === "offline" ? "Server Offline" : "Checking..."}</span>
          </div>
          <button className="refresh-btn" onClick={loadAllData} disabled={loading}>
            <RefreshCw size={16} className={loading ? "spin" : ""} />
            Refresh
          </button>
        </div>
      </header>

      <main className="main-layout">
        <aside className="left-panel">
          <ProjectForm onSubmitSuccess={handlePredictionSuccess} />
          {lastPrediction && (
            <PredictionResult 
              prediction={lastPrediction} 
              onClose={() => setLastPrediction(null)} 
            />
          )}
        </aside>

        <section className="right-panel">
          {kpiData && <KpiDashboard data={kpiData} />}
          
          {projects.length > 0 ? (
            <RiskScoring projects={projects} />
          ) : (
            <div className="empty-state">
              <Shield size={48} />
              <h3>No Projects Yet</h3>
              <p>Add a project using the form to see risk predictions</p>
            </div>
          )}

          {trendData.length > 0 && (
            <TrendAnalysis trendData={trendData} completionData={completionData} />
          )}

          {alerts.length > 0 && (
            <AlertSystem alerts={alerts} />
          )}

          <div className="bottom-grid">
            <HistoricalReports />
            <ManagerInsights insights={insights} />
          </div>

          {projects.length > 0 && (
            <div className="clear-section">
              <button className="clear-btn" onClick={handleClearAll}>
                <Trash2 size={16} /> Clear All Projects
              </button>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
