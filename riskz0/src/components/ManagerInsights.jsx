import { useState } from "react";
import {
  Lightbulb,
  ArrowRight,
  Users,
  Calendar,
  Gauge,
  Star,
  X,
  CheckCircle,
  AlertTriangle,
  Shield,
} from "lucide-react";

const ACTION_RECOMMENDATIONS = {
  risk: [
    { action: "Schedule immediate risk review meeting", icon: "🔴", urgent: true },
    { action: "Assign dedicated risk mitigation leads per project", icon: "👤", urgent: true },
    { action: "Implement daily status check-ins for critical projects", icon: "📋", urgent: false },
    { action: "Create contingency plans with rollback strategies", icon: "🔄", urgent: false },
    { action: "Escalate to steering committee for resource reallocation", icon: "📢", urgent: true },
  ],
  performance: [
    { action: "Redistribute workload across less-loaded teams", icon: "⚖️", urgent: true },
    { action: "Bring in external consultants for bottleneck areas", icon: "🧑‍💼", urgent: false },
    { action: "Review and adjust project timelines with stakeholders", icon: "📅", urgent: true },
    { action: "Implement performance monitoring dashboards per team", icon: "📊", urgent: false },
  ],
  schedule: [
    { action: "Fast-track critical path activities immediately", icon: "⚡", urgent: true },
    { action: "Negotiate scope reduction with product owners", icon: "✂️", urgent: false },
    { action: "Add buffer time to upcoming milestone deadlines", icon: "🕐", urgent: false },
    { action: "Identify and eliminate dependencies blocking progress", icon: "🔗", urgent: true },
  ],
  quality: [
    { action: "Document best practices from successful projects", icon: "📝", urgent: false },
    { action: "Share learnings across teams in retrospective sessions", icon: "💡", urgent: false },
    { action: "Replicate successful patterns to at-risk projects", icon: "🔁", urgent: false },
  ],
};

export default function ManagerInsights({ insights }) {
  const [activeAction, setActiveAction] = useState(null);
  const [completedActions, setCompletedActions] = useState(new Set());

  if (!insights || insights.length === 0) {
    return (
      <div className="manager-insights">
        <h3 className="section-title">
          <Lightbulb size={18} />
          Manager Insights
        </h3>
        <div className="insights-empty">
          <p>Add projects to generate insights</p>
        </div>
      </div>
    );
  }

  const getPriorityClass = (priority) => {
    switch (priority) {
      case "high":
        return "priority-high";
      case "medium":
        return "priority-medium";
      case "low":
        return "priority-low";
      default:
        return "";
    }
  };

  const getCategoryIcon = (category) => {
    switch (category) {
      case "resource":
      case "risk":
        return <Users size={16} />;
      case "schedule":
        return <Calendar size={16} />;
      case "performance":
        return <Gauge size={16} />;
      case "quality":
        return <Star size={16} />;
      default:
        return <Lightbulb size={16} />;
    }
  };

  const handleTakeAction = (insight) => {
    setActiveAction(insight);
  };

  const toggleComplete = (insightId, actionIdx) => {
    const key = `${insightId}-${actionIdx}`;
    setCompletedActions((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  const getRecommendations = (category) => {
    return ACTION_RECOMMENDATIONS[category] || ACTION_RECOMMENDATIONS.risk;
  };

  return (
    <div className="manager-insights">
      <h3 className="section-title">
        <Lightbulb size={18} />
        Manager Insights
      </h3>
      <div className="insights-list">
        {insights.map((insight) => (
          <div key={insight.id} className="insight-card">
            <div className="insight-card-header">
              <div className="insight-category-icon">
                {getCategoryIcon(insight.category)}
              </div>
              <div className="insight-header-text">
                <h4>{insight.title}</h4>
                <span
                  className={`priority-badge ${getPriorityClass(insight.priority)}`}
                >
                  {insight.priority} priority
                </span>
              </div>
            </div>
            <p className="insight-description">{insight.description}</p>
            <button
              className="insight-action"
              onClick={() => handleTakeAction(insight)}
            >
              Take Action <ArrowRight size={14} />
            </button>
          </div>
        ))}
      </div>

      {activeAction && (
        <div
          className="report-modal-overlay"
          onClick={() => setActiveAction(null)}
        >
          <div className="report-modal action-modal" onClick={(e) => e.stopPropagation()}>
            <div className="report-modal-header">
              <div className="action-modal-title">
                <Shield size={20} />
                <h3>Recommended Actions</h3>
              </div>
              <button
                className="modal-close"
                onClick={() => setActiveAction(null)}
              >
                <X size={18} />
              </button>
            </div>
            <div className="report-modal-body">
              <div className="action-insight-summary">
                <span
                  className={`priority-badge ${getPriorityClass(activeAction.priority)}`}
                >
                  {activeAction.priority}
                </span>
                <h4>{activeAction.title}</h4>
                <p>{activeAction.description}</p>
              </div>

              <div className="action-list">
                {getRecommendations(activeAction.category).map((rec, idx) => {
                  const key = `${activeAction.id}-${idx}`;
                  const done = completedActions.has(key);
                  return (
                    <div
                      key={idx}
                      className={`action-item ${done ? "action-done" : ""} ${rec.urgent ? "action-urgent" : ""}`}
                      onClick={() => toggleComplete(activeAction.id, idx)}
                    >
                      <div className="action-check">
                        {done ? (
                          <CheckCircle size={18} className="check-done" />
                        ) : (
                          <div className="check-empty" />
                        )}
                      </div>
                      <span className="action-icon">{rec.icon}</span>
                      <span className={`action-text ${done ? "text-done" : ""}`}>
                        {rec.action}
                      </span>
                      {rec.urgent && !done && (
                        <span className="urgent-badge">
                          <AlertTriangle size={12} /> Urgent
                        </span>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
            <div className="report-modal-footer">
              <span className="action-progress">
                {completedActions.size} actions completed this session
              </span>
              <button
                className="modal-close-btn"
                onClick={() => setActiveAction(null)}
              >
                Done
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
