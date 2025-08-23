const API_BASE = "http://127.0.0.1:8080";  // FastAPI backend
let currentJobId = null;
let visitedJobs = new Set();  // store job IDs already opened

// Utility: copy text by element ID
function copyText(elementId) {
  const el = document.getElementById(elementId);
  // If it's the job description, copy HTML (for bold)
  if (elementId === "job-description") {
    // Create a temporary element to select HTML
    const range = document.createRange();
    range.selectNodeContents(el);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    try {
      document.execCommand("copy");
      alert("Copied!");
    } catch (err) {
      console.error("Error copying HTML:", err);
    }
    selection.removeAllRanges();
    return;
  }
  // Otherwise, copy text (for URL)
  const textToCopy = el.tagName === "A" ? el.href : el.innerText;
  navigator.clipboard.writeText(textToCopy)
    .then(() => alert("Copied!"))
    .catch(err => console.error("Error copying text:", err));
}

// Render bold (**text**) as <b>text</b>
function renderMarkdownToHtml(markdownText) {
  if (!markdownText) return "";
  return markdownText.replace(/\*\*(.*?)\*\*/g, '<b>$1</b>');
}

function updateMeta() {
  document.getElementById("visited-count").innerText = `Visited: ${visitedJobs.size}`;
}

// Load the first job
async function loadFirstJob() {
  try {
    const res = await fetch(`${API_BASE}/jobs?page=1&limit=1`);
    if (!res.ok) throw new Error("Failed to load first job");
    const jobs = await res.json();
    if (jobs.length > 0) {
      showJob(jobs[0]);
    } else {
      document.getElementById("job-title").innerText = "No jobs found";
    }
  } catch (err) {
    console.error("Error loading first job:", err);
    document.getElementById("job-title").innerText = "Error loading job";
  }
}

// Show job details
function showJob(job) {
  currentJobId = job.entity_id;
  visitedJobs.add(job.entity_id); // mark as visited
  updateMeta();

  document.getElementById("job-title").innerText = job.title || "No title";
  document.getElementById("job-company").innerText = job.company || "N/A";
  document.getElementById("job-location").innerText = job.location || "N/A";

  // URL (IMPORTANT: your DB column is apply_url)
  const jobUrl = document.getElementById("job-url");
  const href = job.apply_url && job.apply_url.trim() ? job.apply_url.trim() : "";
  if (href) {
    jobUrl.href = href;
    jobUrl.innerText = href;
    jobUrl.removeAttribute("aria-disabled");
  } else {
    jobUrl.href = "#";
    jobUrl.innerText = "N/A";
    jobUrl.setAttribute("aria-disabled", "true");
  }

  // Render description as HTML with bold
  document.getElementById("job-description").innerHTML = renderMarkdownToHtml(job.description || "No description");
}

// Fetch next job, skipping visited ones
async function loadNextJob() {
  if (!currentJobId) return;
  try {
    let nextJob = null;
    let res = await fetch(`${API_BASE}/jobs/next/${currentJobId}`);
    if (res.ok) nextJob = await res.json();

    // keep fetching until we get a non-visited job
    while (nextJob && visitedJobs.has(nextJob.entity_id)) {
      res = await fetch(`${API_BASE}/jobs/next/${nextJob.entity_id}`);
      if (!res.ok) throw new Error("No more jobs available.");
      nextJob = await res.json();
    }

    if (nextJob) showJob(nextJob);
  } catch (err) {
    alert("No more jobs available.");
  }
}

// Fetch previous job, skipping visited ones
async function loadPrevJob() {
  if (!currentJobId) return;
  try {
    let prevJob = null;
    let res = await fetch(`${API_BASE}/jobs/prev/${currentJobId}`);
    if (res.ok) prevJob = await res.json();

    while (prevJob && visitedJobs.has(prevJob.entity_id)) {
      res = await fetch(`${API_BASE}/jobs/prev/${prevJob.entity_id}`);
      if (!res.ok) throw new Error("No previous jobs available.");
      prevJob = await res.json();
    }

    if (prevJob) showJob(prevJob);
  } catch (err) {
    alert("No previous job available.");
  }
}

// Event listeners
document.getElementById("next-btn").addEventListener("click", loadNextJob);
document.getElementById("prev-btn").addEventListener("click", loadPrevJob);

// Load first job on startup
loadFirstJob();
