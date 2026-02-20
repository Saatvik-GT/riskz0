import { Shield, TrendingUp, TrendingDown, Minus } from "lucide-react";
import { riskScores } from "../data/mockData";

export default function RiskScoring() {
  const getTrendIcon = (trend) => {
    switch (trend) {
      case "up":
        return <TrendingUp size={14} className="trend-up" />;
      case "down":
        return <TrendingDown size={14} className="trend-down" />;
      default:
        return <Minus size={14} className="trend-stable" />;
    }
  };

  const getStatusClass = (status) => {
    switch (status) {
      case "critical":
        return "status-critical";
      case "warning":
        return "status-warning";
      case "healthy":
        return "status-healthy";
      default:
        return "";
    }
  };

  return (
    <div className="risk-scoring">
      <h3 className="section-title">
        <Shield size={18} />
        Risk Scoring Engine
      </h3>
      <div className="risk-table-wrapper">
        <table className="risk-table">
          <thead>
            <tr>
              <th>Project</th>
              <th>Risk Score</th>
              <th>Status</th>
              <th>Trend</th>
              <th>Progress</th>
            </tr>
          </thead>
          <tbody>
            {riskScores.map((row) => (
              <tr key={row.project}>
                <td className="project-name">{row.project}</td>
                <td>
                  <div className="score-cell">
                    <div className="score-bar-bg">
                      <div
                        className={`score-bar ${getStatusClass(row.status)}`}
                        style={{ width: `${row.score * 10}%` }}
                      />
                    </div>
                    <span className="score-value">{row.score}</span>
                  </div>
                </td>
                <td>
                  <span className={`status-badge ${getStatusClass(row.status)}`}>
                    {row.status}
                  </span>
                </td>
                <td className="trend-cell">
                  {getTrendIcon(row.trend)}
                </td>
                <td>
                  <span className="progress-text">
                    {row.completed}/{row.tasks}
                  </span>
                  <span className="delayed-text">
                    ({row.delayed} delayed)
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
