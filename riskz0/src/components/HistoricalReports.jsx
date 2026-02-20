import { FileBarChart, Download, Archive, Eye } from "lucide-react";

export default function HistoricalReports() {
  const reports = [
    { 
      id: 1, 
      name: "Current Session Report", 
      date: new Date().toISOString().split('T')[0], 
      projects: "Active", 
      avgRisk: "-",
      status: "live" 
    },
  ];

  return (
    <div className="historical-reports">
      <h3 className="section-title">
        <FileBarChart size={18} />
        Historical Reports
      </h3>
      <div className="reports-list">
        {reports.map((report) => (
          <div key={report.id} className="report-card">
            <div className="report-card-top">
              <div className="report-info">
                <h4>{report.name}</h4>
                <div className="report-meta">
                  <span>{report.date}</span>
                  <span>•</span>
                  <span>{report.projects} projects</span>
                  {report.avgRisk !== "-" && (
                    <>
                      <span>•</span>
                      <span>Avg Risk: {report.avgRisk}</span>
                    </>
                  )}
                </div>
              </div>
              <span className={`report-status report-${report.status}`}>
                <Archive size={12} />
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
      <p className="reports-note">
        Add projects to generate risk assessment reports
      </p>
    </div>
  );
}
