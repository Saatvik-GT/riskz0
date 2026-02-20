import {
  Activity,
  AlertTriangle,
  CheckCircle,
  Clock,
  Target,
  TrendingUp,
} from "lucide-react";

export default function KpiDashboard({ data }) {
  if (!data) return null;

  const cards = [
    {
      label: "Total Projects",
      value: data.totalProjects,
      icon: <Target size={22} />,
      color: "blue",
      change: "Analyzed projects",
    },
    {
      label: "At Risk",
      value: data.atRiskProjects,
      icon: <AlertTriangle size={22} />,
      color: "red",
      change: "Critical + High risk",
    },
    {
      label: "Critical",
      value: data.criticalCount,
      icon: <Clock size={22} />,
      color: "orange",
      change: "Immediate attention",
    },
    {
      label: "Avg Risk Score",
      value: data.avgRiskScore?.toFixed(2) || 0,
      icon: <Activity size={22} />,
      color: "yellow",
      change: "Scale: 0-10",
    },
    {
      label: "Confidence",
      value: `${data.avgConfidence || 0}%`,
      icon: <CheckCircle size={22} />,
      color: "green",
      change: "Model confidence",
    },
    {
      label: "Active Alerts",
      value: data.activeAlerts,
      icon: <TrendingUp size={22} />,
      color: "purple",
      change: "Requires attention",
    },
  ];

  return (
    <div className="kpi-dashboard">
      <h3 className="section-title">
        <Activity size={18} />
        KPI Dashboard
      </h3>
      <div className="kpi-grid">
        {cards.map((card) => (
          <div key={card.label} className={`kpi-card kpi-${card.color}`}>
            <div className="kpi-card-header">
              <span className="kpi-label">{card.label}</span>
              <div className={`kpi-icon kpi-icon-${card.color}`}>
                {card.icon}
              </div>
            </div>
            <div className="kpi-value">{card.value}</div>
            <div className="kpi-change">{card.change}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
