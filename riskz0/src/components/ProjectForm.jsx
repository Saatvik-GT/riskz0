import { useState, useEffect } from "react";
import { Send, Loader, Info, ChevronDown, ChevronUp } from "lucide-react";
import { predictRisk, fetchFormFields } from "../services/api";

const DEFAULT_VALUES = {
  Project_Type: "IT",
  Methodology_Used: "Agile",
  Team_Experience_Level: "Senior",
  Project_Phase: "Planning",
  Requirement_Stability: "Moderate",
  Regulatory_Compliance_Level: "Medium",
  Technology_Familiarity: "Familiar",
  Stakeholder_Engagement_Level: "Medium",
  Executive_Sponsorship: "Moderate",
  Funding_Source: "Internal",
  Priority_Level: "Medium",
  Project_Manager_Experience: "Mid-level PM",
  Org_Process_Maturity: "Managed",
  Data_Security_Requirements: "Medium",
  Key_Stakeholder_Availability: "Good",
  Contract_Type: "Time & Materials",
  Resource_Contention_Level: "Medium",
  Industry_Volatility: "Moderate",
  Client_Experience_Level: "Regular",
  Change_Control_Maturity: "Formal",
  Risk_Management_Maturity: "Advanced",
  Team_Colocation: "Hybrid",
  Documentation_Quality: "Good",
  Complexity_Score: 5.0,
  Estimated_Timeline_Months: 12,
  External_Dependencies_Count: 2,
  Change_Request_Frequency: 1.0,
  Team_Turnover_Rate: 0.15,
  Vendor_Reliability_Score: 0.7,
  Historical_Risk_Incidents: 1,
  Communication_Frequency: 3.0,
  Geographical_Distribution: 2,
  Schedule_Pressure: 0.1,
  Budget_Utilization_Rate: 0.9,
  Market_Volatility: 0.3,
  Integration_Complexity: 3.0,
  Resource_Availability: 0.7,
  Organizational_Change_Frequency: 1.0,
  Cross_Functional_Dependencies: 3,
  Previous_Delivery_Success_Rate: 0.75,
  Technical_Debt_Level: 0.2,
  Project_Start_Month: 1,
  Current_Phase_Duration_Months: 3,
  Seasonal_Risk_Factor: 1.0,
  Past_Similar_Projects: 2,
};

export default function ProjectForm({ onSubmitSuccess }) {
  const [projectName, setProjectName] = useState("");
  const [formData, setFormData] = useState(DEFAULT_VALUES);
  const [formFields, setFormFields] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [expandedSections, setExpandedSections] = useState({
    basic: true,
    team: false,
    process: false,
    technical: false,
    advanced: false,
  });

  useEffect(() => {
    loadFormFields();
  }, []);

  const loadFormFields = async () => {
    try {
      const fields = await fetchFormFields();
      setFormFields(fields);
    } catch (err) {
      console.error("Failed to load form fields:", err);
    }
  };

  const toggleSection = (section) => {
    setExpandedSections((prev) => ({
      ...prev,
      [section]: !prev[section],
    }));
  };

  const handleChange = (field, value) => {
    setFormData((prev) => ({
      ...prev,
      [field]: parseFloat(value) || value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const projectData = {
        name: projectName || `Project ${Date.now()}`,
        ...formData,
      };

      const result = await predictRisk(projectData);

      if (result.success) {
        setProjectName("");
        setFormData(DEFAULT_VALUES);
        if (onSubmitSuccess) {
          onSubmitSuccess(result);
        }
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const renderSelect = (field, label) => (
    <div className="form-field" key={field}>
      <label>{label}</label>
      <select
        value={formData[field]}
        onChange={(e) => handleChange(field, e.target.value)}
      >
        {formFields?.categorical[field]?.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </div>
  );

  const renderNumber = (field, config) => (
    <div className="form-field" key={field}>
      <label>
        {field.replace(/_/g, " ")}
        {config.description && (
          <span className="field-hint" title={config.description}>
            <Info size={12} />
          </span>
        )}
      </label>
      <input
        type="number"
        step={config.max <= 1 ? "0.01" : "0.1"}
        min={config.min}
        max={config.max}
        value={formData[field]}
        onChange={(e) => handleChange(field, e.target.value)}
      />
    </div>
  );

  return (
    <form className="project-form" onSubmit={handleSubmit}>
      <div className="form-header">
        <h3>Add New Project</h3>
        <p>Enter project details to predict risk level</p>
      </div>

      <div className="form-field project-name-field">
        <label>Project Name</label>
        <input
          type="text"
          placeholder="Enter project name..."
          value={projectName}
          onChange={(e) => setProjectName(e.target.value)}
        />
      </div>

      {formFields && (
        <>
          <div className="form-section">
            <button
              type="button"
              className="section-toggle"
              onClick={() => toggleSection("basic")}
            >
              <span>Basic Information</span>
              {expandedSections.basic ? (
                <ChevronUp size={16} />
              ) : (
                <ChevronDown size={16} />
              )}
            </button>
            {expandedSections.basic && (
              <div className="section-content">
                {renderSelect("Project_Type", "Project Type")}
                {renderSelect("Project_Phase", "Project Phase")}
                {renderSelect("Priority_Level", "Priority Level")}
                {renderSelect("Funding_Source", "Funding Source")}
                {renderNumber(
                  "Estimated_Timeline_Months",
                  formFields.numerical.Estimated_Timeline_Months
                )}
                {renderNumber(
                  "Complexity_Score",
                  formFields.numerical.Complexity_Score
                )}
              </div>
            )}
          </div>

          <div className="form-section">
            <button
              type="button"
              className="section-toggle"
              onClick={() => toggleSection("team")}
            >
              <span>Team & Experience</span>
              {expandedSections.team ? (
                <ChevronUp size={16} />
              ) : (
                <ChevronDown size={16} />
              )}
            </button>
            {expandedSections.team && (
              <div className="section-content">
                {renderSelect("Team_Experience_Level", "Team Experience")}
                {renderSelect("Project_Manager_Experience", "PM Experience")}
                {renderSelect("Team_Colocation", "Team Location")}
                {renderNumber(
                  "Team_Turnover_Rate",
                  formFields.numerical.Team_Turnover_Rate
                )}
                {renderNumber(
                  "Past_Similar_Projects",
                  formFields.numerical.Past_Similar_Projects
                )}
                {renderNumber(
                  "Previous_Delivery_Success_Rate",
                  formFields.numerical.Previous_Delivery_Success_Rate
                )}
              </div>
            )}
          </div>

          <div className="form-section">
            <button
              type="button"
              className="section-toggle"
              onClick={() => toggleSection("process")}
            >
              <span>Process & Governance</span>
              {expandedSections.process ? (
                <ChevronUp size={16} />
              ) : (
                <ChevronDown size={16} />
              )}
            </button>
            {expandedSections.process && (
              <div className="section-content">
                {renderSelect("Methodology_Used", "Methodology")}
                {renderSelect("Org_Process_Maturity", "Process Maturity")}
                {renderSelect("Change_Control_Maturity", "Change Control")}
                {renderSelect("Risk_Management_Maturity", "Risk Management")}
                {renderSelect("Requirement_Stability", "Requirement Stability")}
                {renderSelect("Executive_Sponsorship", "Exec Sponsorship")}
              </div>
            )}
          </div>

          <div className="form-section">
            <button
              type="button"
              className="section-toggle"
              onClick={() => toggleSection("technical")}
            >
              <span>Technical & Resources</span>
              {expandedSections.technical ? (
                <ChevronUp size={16} />
              ) : (
                <ChevronDown size={16} />
              )}
            </button>
            {expandedSections.technical && (
              <div className="section-content">
                {renderSelect("Technology_Familiarity", "Tech Familiarity")}
                {renderNumber(
                  "External_Dependencies_Count",
                  formFields.numerical.External_Dependencies_Count
                )}
                {renderNumber(
                  "Integration_Complexity",
                  formFields.numerical.Integration_Complexity
                )}
                {renderNumber(
                  "Technical_Debt_Level",
                  formFields.numerical.Technical_Debt_Level
                )}
                {renderNumber(
                  "Resource_Availability",
                  formFields.numerical.Resource_Availability
                )}
                {renderSelect("Resource_Contention_Level", "Resource Contention")}
              </div>
            )}
          </div>

          <div className="form-section">
            <button
              type="button"
              className="section-toggle"
              onClick={() => toggleSection("advanced")}
            >
              <span>Advanced Settings</span>
              {expandedSections.advanced ? (
                <ChevronUp size={16} />
              ) : (
                <ChevronDown size={16} />
              )}
            </button>
            {expandedSections.advanced && (
              <div className="section-content">
                {renderSelect("Stakeholder_Engagement_Level", "Stakeholder Engagement")}
                {renderSelect("Key_Stakeholder_Availability", "Stakeholder Availability")}
                {renderSelect("Client_Experience_Level", "Client Experience")}
                {renderSelect("Contract_Type", "Contract Type")}
                {renderSelect("Industry_Volatility", "Industry Volatility")}
                {renderSelect("Regulatory_Compliance_Level", "Regulatory Level")}
                {renderSelect("Data_Security_Requirements", "Security Requirements")}
                {renderNumber("Budget_Utilization_Rate", formFields.numerical.Budget_Utilization_Rate)}
                {renderNumber("Schedule_Pressure", formFields.numerical.Schedule_Pressure)}
                {renderNumber("Market_Volatility", formFields.numerical.Market_Volatility)}
                {renderNumber("Vendor_Reliability_Score", formFields.numerical.Vendor_Reliability_Score)}
                {renderNumber("Historical_Risk_Incidents", formFields.numerical.Historical_Risk_Incidents)}
              </div>
            )}
          </div>
        </>
      )}

      {error && <div className="form-error">{error}</div>}

      <button type="submit" className="submit-btn" disabled={loading}>
        {loading ? (
          <>
            <Loader size={18} className="spin" /> Analyzing...
          </>
        ) : (
          <>
            <Send size={18} /> Predict Risk
          </>
        )}
      </button>
    </form>
  );
}
