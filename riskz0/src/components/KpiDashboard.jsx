import {
  Activity,
  AlertTriangle,
  CheckCircle,
  Clock,
  Target,
  TrendingUp,
} from "lucide-react";
import { kpiData } from "../data/mockData";

export default function KpiDashboard() {
  const cards = [
    {
      label: "Total Projects",
      value: kpiData.totalProjects,
      icon: <Target size={22} />,
      color: "blue",
      change: "+3 this quarter",
    },
    {
      label: "At Risk",
      value: kpiData.atRiskProjects,
      icon: <AlertTriangle size={22} />,
      color: "red",
      change: "+2 from last week",
    },
    {
      label: "Delayed Tasks",
      value: kpiData.delayedTasks,
      icon: <Clock size={22} />,
      color: "orange",
      change: "+5 this sprint",
    },
    {
      label: "Avg Risk Score",
      value: kpiData.avgRiskScore,
      icon: <Activity size={22} />,
      color: "yellow",
      change: "↑ 0.8 from last month",
    },
    {
      label: "Completion Rate",
      value: `${kpiData.completionRate}%`,
      icon: <CheckCircle size={22} />,
      color: "green",
      change: "↓ 5% this sprint",
    },
    {
      label: "Active Alerts",
      value: kpiData.activeAlerts,
      icon: <TrendingUp size={22} />,
      color: "purple",
      change: "2 critical",
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
