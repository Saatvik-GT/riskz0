import { Bell, AlertTriangle, AlertCircle, Info, Check } from "lucide-react";
import { useState } from "react";
import { alerts as alertData } from "../data/mockData";

export default function AlertSystem() {
  const [alerts, setAlerts] = useState(alertData);
  const [filter, setFilter] = useState("all");

  const getIcon = (type) => {
    switch (type) {
      case "critical":
        return <AlertTriangle size={16} className="alert-icon-critical" />;
      case "warning":
        return <AlertCircle size={16} className="alert-icon-warning" />;
      case "info":
        return <Info size={16} className="alert-icon-info" />;
      default:
        return null;
    }
  };

  const dismissAlert = (id) => {
    setAlerts((prev) => prev.filter((a) => a.id !== id));
  };

  const filtered =
    filter === "all" ? alerts : alerts.filter((a) => a.type === filter);

  return (
    <div className="alert-system">
      <div className="section-title-row">
        <h3 className="section-title">
          <Bell size={18} />
          Alert System
          {alerts.length > 0 && (
            <span className="alert-count">{alerts.length}</span>
          )}
        </h3>
        <div className="alert-filters">
          {["all", "critical", "warning", "info"].map((f) => (
            <button
              key={f}
              className={`alert-filter-btn ${filter === f ? "active" : ""} filter-${f}`}
              onClick={() => setFilter(f)}
            >
              {f.charAt(0).toUpperCase() + f.slice(1)}
            </button>
          ))}
        </div>
      </div>
      <div className="alert-list">
        {filtered.length === 0 ? (
          <div className="alert-empty">
            <Check size={24} />
            <p>No alerts to show</p>
          </div>
        ) : (
          filtered.map((alert) => (
            <div key={alert.id} className={`alert-item alert-${alert.type}`}>
              <div className="alert-item-icon">{getIcon(alert.type)}</div>
              <div className="alert-item-content">
                <p className="alert-message">{alert.message}</p>
                <div className="alert-meta">
                  <span className="alert-project">{alert.project}</span>
                  <span className="alert-time">{alert.time}</span>
                </div>
              </div>
              <button
                className="alert-dismiss"
                onClick={() => dismissAlert(alert.id)}
                title="Dismiss"
              >
                ×
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
