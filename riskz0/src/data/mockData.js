// Mock data for the Risk & Delay Prediction System

export const kpiData = {
  totalProjects: 24,
  atRiskProjects: 7,
  delayedTasks: 13,
  avgRiskScore: 6.2,
  completionRate: 72,
  activeAlerts: 5,
};

export const riskScores = [
  { project: "Project Alpha", score: 8.5, status: "critical", trend: "up", tasks: 45, completed: 28, delayed: 8 },
  { project: "Project Beta", score: 6.2, status: "warning", trend: "stable", tasks: 32, completed: 22, delayed: 4 },
  { project: "Project Gamma", score: 3.1, status: "healthy", trend: "down", tasks: 28, completed: 25, delayed: 1 },
  { project: "Project Delta", score: 7.8, status: "critical", trend: "up", tasks: 55, completed: 30, delayed: 12 },
  { project: "Project Epsilon", score: 4.5, status: "warning", trend: "stable", tasks: 20, completed: 15, delayed: 2 },
  { project: "Project Zeta", score: 2.0, status: "healthy", trend: "down", tasks: 18, completed: 17, delayed: 0 },
];

export const trendData = [
  { week: "W1", riskScore: 4.2, tasksCompleted: 12, tasksDelayed: 2, velocity: 85 },
  { week: "W2", riskScore: 4.8, tasksCompleted: 10, tasksDelayed: 4, velocity: 78 },
  { week: "W3", riskScore: 5.5, tasksCompleted: 8, tasksDelayed: 5, velocity: 70 },
  { week: "W4", riskScore: 5.1, tasksCompleted: 11, tasksDelayed: 3, velocity: 75 },
  { week: "W5", riskScore: 6.2, tasksCompleted: 7, tasksDelayed: 6, velocity: 62 },
  { week: "W6", riskScore: 5.8, tasksCompleted: 9, tasksDelayed: 5, velocity: 67 },
  { week: "W7", riskScore: 6.5, tasksCompleted: 6, tasksDelayed: 7, velocity: 58 },
  { week: "W8", riskScore: 6.0, tasksCompleted: 10, tasksDelayed: 4, velocity: 72 },
];

export const alerts = [
  { id: 1, type: "critical", message: "Project Alpha risk score exceeded threshold (8.5)", time: "2 min ago", project: "Project Alpha" },
  { id: 2, type: "critical", message: "Project Delta has 12 delayed tasks", time: "15 min ago", project: "Project Delta" },
  { id: 3, type: "warning", message: "Project Beta velocity dropped 15% this sprint", time: "1 hr ago", project: "Project Beta" },
  { id: 4, type: "warning", message: "Project Epsilon approaching risk threshold (4.5)", time: "2 hrs ago", project: "Project Epsilon" },
  { id: 5, type: "info", message: "Project Zeta completed milestone ahead of schedule", time: "3 hrs ago", project: "Project Zeta" },
  { id: 6, type: "info", message: "Weekly risk report generated successfully", time: "5 hrs ago", project: "System" },
];

export const historicalReports = [
  { id: 1, name: "Q4 2025 Risk Assessment", date: "2025-12-31", projects: 20, avgRisk: 5.4, status: "completed" },
  { id: 2, name: "Q3 2025 Risk Assessment", date: "2025-09-30", projects: 18, avgRisk: 4.8, status: "completed" },
  { id: 3, name: "Q2 2025 Risk Assessment", date: "2025-06-30", projects: 16, avgRisk: 5.1, status: "completed" },
  { id: 4, name: "Q1 2025 Risk Assessment", date: "2025-03-31", projects: 15, avgRisk: 4.2, status: "completed" },
  { id: 5, name: "Q4 2024 Risk Assessment", date: "2024-12-31", projects: 14, avgRisk: 3.9, status: "archived" },
];

export const managerInsights = [
  {
    id: 1,
    title: "Resource Reallocation Needed",
    description: "Projects Alpha and Delta are critically under-resourced. Consider shifting 2-3 engineers from Project Zeta (ahead of schedule) to these projects.",
    priority: "high",
    category: "resource",
  },
  {
    id: 2,
    title: "Sprint Velocity Declining",
    description: "Overall team velocity has dropped 20% over the last 4 sprints. Root cause analysis suggests scope creep in 3 active projects.",
    priority: "high",
    category: "performance",
  },
  {
    id: 3,
    title: "Upcoming Deadline Risk",
    description: "Project Beta's milestone delivery is at risk. Current trajectory suggests a 2-week delay unless immediate intervention is taken.",
    priority: "medium",
    category: "schedule",
  },
  {
    id: 4,
    title: "Quality Metrics Improving",
    description: "Bug escape rate has decreased 30% this quarter. Automated testing coverage increased to 78% across all projects.",
    priority: "low",
    category: "quality",
  },
];

export const completionData = [
  { name: "On Track", value: 14, color: "#10b981" },
  { name: "At Risk", value: 5, color: "#f59e0b" },
  { name: "Delayed", value: 3, color: "#ef4444" },
  { name: "Completed", value: 2, color: "#6366f1" },
];
