const API_BASE_URL = 'http://localhost:5000/api';

export async function fetchKpi() {
  const response = await fetch(`${API_BASE_URL}/kpi`);
  if (!response.ok) throw new Error('Failed to fetch KPI data');
  return response.json();
}

export async function fetchProjects() {
  const response = await fetch(`${API_BASE_URL}/projects`);
  if (!response.ok) throw new Error('Failed to fetch projects');
  return response.json();
}

export async function fetchAlerts() {
  const response = await fetch(`${API_BASE_URL}/alerts`);
  if (!response.ok) throw new Error('Failed to fetch alerts');
  return response.json();
}

export async function fetchInsights() {
  const response = await fetch(`${API_BASE_URL}/insights`);
  if (!response.ok) throw new Error('Failed to fetch insights');
  return response.json();
}

export async function fetchTrends() {
  const response = await fetch(`${API_BASE_URL}/trends`);
  if (!response.ok) throw new Error('Failed to fetch trends');
  return response.json();
}

export async function predictRisk(projectData) {
  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ project_data: projectData }),
  });
  if (!response.ok) throw new Error('Failed to predict risk');
  return response.json();
}

export async function fetchFormFields() {
  const response = await fetch(`${API_BASE_URL}/form-fields`);
  if (!response.ok) throw new Error('Failed to fetch form fields');
  return response.json();
}

export async function clearProjects() {
  const response = await fetch(`${API_BASE_URL}/clear`, {
    method: 'POST',
  });
  if (!response.ok) throw new Error('Failed to clear projects');
  return response.json();
}

export async function healthCheck() {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) throw new Error('Health check failed');
  return response.json();
}
