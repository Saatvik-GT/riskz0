import { FileBarChart, Download, Archive, Eye } from "lucide-react";
import { historicalReports } from "../data/mockData";

export default function HistoricalReports() {
  return (
    <div className="historical-reports">
      <h3 className="section-title">
        <FileBarChart size={18} />
        Historical Reports
      </h3>
      <div className="reports-list">
        {historicalReports.map((report) => (
          <div key={report.id} className="report-card">
            <div className="report-card-top">
              <div className="report-info">
                <h4>{report.name}</h4>
                <div className="report-meta">
                  <span>{report.date}</span>
                  <span>•</span>
                  <span>{report.projects} projects</span>
                  <span>•</span>
                  <span>Avg Risk: {report.avgRisk}</span>
                </div>
              </div>
              <span
                className={`report-status ${
                  report.status === "completed" ? "report-completed" : "report-archived"
                }`}
              >
                {report.status === "completed" ? (
                  <Archive size={12} />
                ) : (
                  <Archive size={12} />
                )}
                {report.status}
              </span>
            </div>
            <div className="report-actions">
              <button className="report-btn" title="View">
                <Eye size={14} /> View
              </button>
              <button className="report-btn" title="Download">
                <Download size={14} /> Download
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
