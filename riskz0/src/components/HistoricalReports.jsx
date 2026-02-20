import { useState } from "react";
import { FileBarChart, Download, Eye, X } from "lucide-react";

export default function HistoricalReports({ projects = [], kpiData }) {
  const [viewingReport, setViewingReport] = useState(false);

  const generateReportData = () => {
    if (!projects || projects.length === 0) return null;

    const critical = projects.filter((p) => p.risk_level === "Critical").length;
    const high = projects.filter((p) => p.risk_level === "High").length;
    const medium = projects.filter((p) => p.risk_level === "Medium").length;
    const low = projects.filter((p) => p.risk_level === "Low").length;
    const avgScore = (
      projects.reduce((sum, p) => sum + p.score, 0) / projects.length
    ).toFixed(2);
    const avgConfidence = (
      projects.reduce((sum, p) => sum + (p.confidence || 0), 0) /
      projects.length *
      100
    ).toFixed(1);

    return {
      date: new Date().toISOString().split("T")[0],
      time: new Date().toLocaleTimeString(),
      totalProjects: projects.length,
      distribution: { critical, high, medium, low },
      avgScore,
      avgConfidence,
      projects: projects.map((p) => ({
        name: p.project,
        risk: p.risk_level,
        score: p.score,
        status: p.status,
        confidence: ((p.confidence || 0) * 100).toFixed(1),
      })),
    };
  };

  const handleDownload = () => {
    const report = generateReportData();
    if (!report) return;

    let csv = "RiskZ0 — Risk Assessment Report\n";
    csv += `Generated: ${report.date} ${report.time}\n\n`;
    csv += `Total Projects,${report.totalProjects}\n`;
    csv += `Average Risk Score,${report.avgScore}\n`;
    csv += `Average Confidence,${report.avgConfidence}%\n\n`;
    csv += `Risk Distribution\n`;
    csv += `Critical,${report.distribution.critical}\n`;
    csv += `High,${report.distribution.high}\n`;
    csv += `Medium,${report.distribution.medium}\n`;
    csv += `Low,${report.distribution.low}\n\n`;
    csv += `Project,Risk Level,Risk Score,Status,Confidence\n`;
    report.projects.forEach((p) => {
      csv += `${p.name},${p.risk},${p.score},${p.status},${p.confidence}%\n`;
    });

    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `risk_report_${report.date}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const report = generateReportData();

  return (
    <div className="historical-reports">
      <h3 className="section-title">
        <FileBarChart size={18} />
        Historical Reports
      </h3>
      <div className="reports-list">
        <div className="report-card">
          <div className="report-card-top">
            <div className="report-info">
              <h4>Current Session Report</h4>
              <div className="report-meta">
                <span>{new Date().toISOString().split("T")[0]}</span>
                <span>•</span>
                <span>
                  {projects.length > 0
                    ? `${projects.length} projects`
                    : "Active"}
                </span>
                {report && (
                  <>
                    <span>•</span>
                    <span>Avg Risk: {report.avgScore}</span>
                  </>
                )}
              </div>
            </div>
            <span className="report-status report-live">
              <span className="live-dot" />
              live
            </span>
          </div>
          <div className="report-actions">
            <button
              className="report-btn"
              title="View"
              onClick={() => setViewingReport(true)}
              disabled={!report}
            >
              <Eye size={14} /> View
            </button>
            <button
              className="report-btn"
              title="Download"
              onClick={handleDownload}
              disabled={!report}
            >
              <Download size={14} /> Download
            </button>
          </div>
        </div>
      </div>

      {!report && (
        <p className="reports-note">
          Add projects to generate risk assessment reports
        </p>
      )}

      {viewingReport && report && (
        <div className="report-modal-overlay" onClick={() => setViewingReport(false)}>
          <div className="report-modal" onClick={(e) => e.stopPropagation()}>
            <div className="report-modal-header">
              <h3>Risk Assessment Report</h3>
              <button className="modal-close" onClick={() => setViewingReport(false)}>
                <X size={18} />
              </button>
            </div>
            <div className="report-modal-body">
              <div className="report-section">
                <p className="report-date">
                  {report.date} • {report.time}
                </p>
              </div>

              <div className="report-section">
                <h4>Summary</h4>
                <div className="report-summary-grid">
                  <div className="report-stat">
                    <span className="stat-number">{report.totalProjects}</span>
                    <span className="stat-label">Total Projects</span>
                  </div>
                  <div className="report-stat">
                    <span className="stat-number">{report.avgScore}</span>
                    <span className="stat-label">Avg Risk Score</span>
                  </div>
                  <div className="report-stat">
                    <span className="stat-number">{report.avgConfidence}%</span>
                    <span className="stat-label">Avg Confidence</span>
                  </div>
                </div>
              </div>

              <div className="report-section">
                <h4>Risk Distribution</h4>
                <div className="report-dist-bars">
                  {[
                    { label: "Critical", count: report.distribution.critical, color: "#ef4444" },
                    { label: "High", count: report.distribution.high, color: "#f97316" },
                    { label: "Medium", count: report.distribution.medium, color: "#f59e0b" },
                    { label: "Low", count: report.distribution.low, color: "#10b981" },
                  ].map((d) => (
                    <div key={d.label} className="dist-bar-row">
                      <span className="dist-label">{d.label}</span>
                      <div className="dist-bar-track">
                        <div
                          className="dist-bar-fill"
                          style={{
                            width: `${report.totalProjects > 0 ? (d.count / report.totalProjects) * 100 : 0}%`,
                            backgroundColor: d.color,
                          }}
                        />
                      </div>
                      <span className="dist-count">{d.count}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="report-section">
                <h4>Project Details</h4>
                <table className="report-table">
                  <thead>
                    <tr>
                      <th>Project</th>
                      <th>Risk</th>
                      <th>Score</th>
                      <th>Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {report.projects.map((p, i) => (
                      <tr key={i}>
                        <td>{p.name}</td>
                        <td>
                          <span className={`risk-badge risk-${p.risk.toLowerCase()}`}>
                            {p.risk}
                          </span>
                        </td>
                        <td>{p.score}</td>
                        <td>{p.confidence}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            <div className="report-modal-footer">
              <button className="report-btn" onClick={handleDownload}>
                <Download size={14} /> Download CSV
              </button>
              <button className="modal-close-btn" onClick={() => setViewingReport(false)}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
