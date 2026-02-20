import { Lightbulb, ArrowRight, Users, Calendar, Gauge, Star } from "lucide-react";
import { managerInsights } from "../data/mockData";

export default function ManagerInsights() {
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

  return (
    <div className="manager-insights">
      <h3 className="section-title">
        <Lightbulb size={18} />
        Manager Insights
      </h3>
      <div className="insights-list">
        {managerInsights.map((insight) => (
          <div key={insight.id} className="insight-card">
            <div className="insight-card-header">
              <div className="insight-category-icon">
                {getCategoryIcon(insight.category)}
              </div>
              <div className="insight-header-text">
                <h4>{insight.title}</h4>
                <span className={`priority-badge ${getPriorityClass(insight.priority)}`}>
                  {insight.priority} priority
                </span>
              </div>
            </div>
            <p className="insight-description">{insight.description}</p>
            <button className="insight-action">
              Take Action <ArrowRight size={14} />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
