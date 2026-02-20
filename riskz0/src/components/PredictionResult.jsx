import { X, AlertTriangle, AlertCircle, CheckCircle, Shield } from "lucide-react";

export default function PredictionResult({ prediction, onClose }) {
  const getStatusIcon = (riskLevel) => {
    switch (riskLevel) {
      case "Critical":
        return <AlertTriangle size={24} className="status-icon-critical" />;
      case "High":
        return <AlertCircle size={24} className="status-icon-warning" />;
      case "Medium":
        return <Shield size={24} className="status-icon-medium" />;
      default:
        return <CheckCircle size={24} className="status-icon-healthy" />;
    }
  };

  const getStatusClass = (riskLevel) => {
    switch (riskLevel) {
      case "Critical":
        return "result-critical";
      case "High":
        return "result-warning";
      case "Medium":
        return "result-medium";
      default:
        return "result-healthy";
    }
  };

  return (
    <div className={`prediction-result ${getStatusClass(prediction.risk_level)}`}>
      <button className="close-btn" onClick={onClose}>
        <X size={18} />
      </button>

      <div className="result-header">
        {getStatusIcon(prediction.risk_level)}
        <h3>Risk Assessment Complete</h3>
      </div>

      <div className="result-body">
        <div className="result-main">
          <div className="risk-level-badge">{prediction.risk_level}</div>
          <div className="confidence">
            <span className="confidence-label">Confidence</span>
            <span className="confidence-value">{(prediction.confidence * 100).toFixed(1)}%</span>
          </div>
          <div className="risk-score">
            <span className="score-label">Risk Score</span>
            <span className="score-value">{prediction.risk_score.toFixed(2)}</span>
          </div>
        </div>

        <div className="probabilities">
          <h4>Class Probabilities</h4>
          {Object.entries(prediction.probabilities).map(([cls, prob]) => (
            <div key={cls} className="prob-bar-container">
              <span className="prob-label">{cls}</span>
              <div className="prob-bar-bg">
                <div
                  className={`prob-bar prob-${cls.toLowerCase()}`}
                  style={{ width: `${prob * 100}%` }}
                />
              </div>
              <span className="prob-value">{(prob * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
