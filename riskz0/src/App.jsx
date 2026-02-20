import PdfUpload from "./components/PdfUpload";
import KpiDashboard from "./components/KpiDashboard";
import RiskScoring from "./components/RiskScoring";
import TrendAnalysis from "./components/TrendAnalysis";
import AlertSystem from "./components/AlertSystem";
import HistoricalReports from "./components/HistoricalReports";
import ManagerInsights from "./components/ManagerInsights";
import { Shield } from "lucide-react";
import "./App.css";

function App() {
  return (
    <div className="app">
      {/* Top navbar */}
      <header className="navbar">
        <div className="navbar-brand">
          <Shield size={26} className="brand-logo-svg" />
          <h1>RiskZ0</h1>
          <span className="navbar-tag">Risk & Delay Prediction</span>
        </div>
        <div className="navbar-status">
          <span className="status-dot" />
          <span>System Active</span>
        </div>
      </header>

      {/* Main split layout */}
      <main className="main-layout">
        {/* Left panel – PDF Upload */}
        <aside className="left-panel">
          <PdfUpload />
        </aside>

        {/* Right panel – Metrics Dashboard */}
        <section className="right-panel">
          <KpiDashboard />
          <RiskScoring />
          <TrendAnalysis />
          <AlertSystem />
          <div className="bottom-grid">
            <HistoricalReports />
            <ManagerInsights />
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
