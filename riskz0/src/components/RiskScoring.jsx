import { Shield, TrendingUp, TrendingDown, Minus } from "lucide-react";

export default function RiskScoring({ projects }) {
  if (!projects || projects.length === 0) return null;

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
              <th>Confidence</th>
            </tr>
          </thead>
          <tbody>
            {projects.map((row) => (
              <tr key={row.id || row.project}>
                <td className="project-name">{row.project}</td>
                <td>
                  <div className="score-cell">
                    <div className="score-bar-bg">
                      <div
                        className={`score-bar ${getStatusClass(row.status)}`}
                        style={{ width: `${row.score * 10}%` }}
                      />
                    </div>
                    <span className="score-value">{row.score.toFixed(1)}</span>
                  </div>
                </td>
                <td>
                  <span className={`status-badge ${getStatusClass(row.status)}`}>
                    {row.risk_level || row.status}
                  </span>
                </td>
                <td className="trend-cell">
                  {getTrendIcon(row.trend)}
                </td>
                <td>
                  <span className="confidence-text">
                    {(row.confidence * 100).toFixed(1)}%
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
