const API_BASE = "http://127.0.0.1:8000";  // FastAPI backend
let currentJob = null;
let totalJobs = 0;
let currentJobIndex = 0;

// Utility: copy text by element ID with enhanced formatting preservation
function copyText(elementId) {
  const el = document.getElementById(elementId);
  
  if (elementId === "job-description") {
    // Enhanced HTML copying for job descriptions
    try {
      // Method 1: Try modern clipboard API with HTML
      if (navigator.clipboard && window.ClipboardItem) {
        const htmlBlob = new Blob([el.innerHTML], { type: 'text/html' });
        const textBlob = new Blob([el.innerText], { type: 'text/plain' });
        
        const clipboardItem = new ClipboardItem({
          'text/html': htmlBlob,
          'text/plain': textBlob
        });
        
        navigator.clipboard.write([clipboardItem]).then(() => {
          showCopySuccess("Description copied with formatting!");
        }).catch(() => {
          // Fallback to selection method
          fallbackCopyHTML(el);
        });
      } else {
        // Fallback for older browsers
        fallbackCopyHTML(el);
      }
    } catch (err) {
      console.error("Error copying HTML:", err);
      fallbackCopyHTML(el);
    }
    return;
  }
  
  // For other elements (URL, etc.), copy as plain text
  const textToCopy = el.tagName === "A" ? el.href : el.innerText;
  navigator.clipboard.writeText(textToCopy)
    .then(() => showCopySuccess("Copied!"))
    .catch(err => {
      console.error("Error copying text:", err);
      // Fallback for older browsers
      fallbackCopyText(textToCopy);
    });
}

// Fallback HTML copying method
function fallbackCopyHTML(element) {
  const range = document.createRange();
  range.selectNodeContents(element);
  const selection = window.getSelection();
  selection.removeAllRanges();
  selection.addRange(range);
  
  try {
    document.execCommand("copy");
    showCopySuccess("Description copied!");
  } catch (err) {
    console.error("Fallback copy failed:", err);
    showCopyError("Copy failed. Please select and copy manually.");
  }
  
  selection.removeAllRanges();
}

// Fallback text copying method
function fallbackCopyText(text) {
  const textArea = document.createElement("textarea");
  textArea.value = text;
  document.body.appendChild(textArea);
  textArea.select();
  
  try {
    document.execCommand("copy");
    showCopySuccess("Copied!");
  } catch (err) {
    console.error("Fallback copy failed:", err);
    showCopyError("Copy failed. Please select and copy manually.");
  }
  
  document.body.removeChild(textArea);
}

// Show copy success message
function showCopySuccess(message) {
  const notification = document.createElement("div");
  notification.className = "copy-notification success";
  notification.textContent = message;
  document.body.appendChild(notification);
  
  setTimeout(() => {
    notification.remove();
  }, 2000);
}

// Show copy error message
function showCopyError(message) {
  const notification = document.createElement("div");
  notification.className = "copy-notification error";
  notification.textContent = message;
  document.body.appendChild(notification);
  
  setTimeout(() => {
    notification.remove();
  }, 3000);
}

// Enhanced markdown to HTML rendering
function renderMarkdownToHtml(markdownText) {
  if (!markdownText) return "";
  
  let html = markdownText;
  
  // Convert **text** to <strong>text</strong> (better semantic than <b>)
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  
  // Convert *text* to <em>text</em> (italic)
  html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
  
  // Convert # Heading to <h3>Heading</h3>
  html = html.replace(/^# (.*$)/gm, '<h3>$1</h3>');
  
  // Convert ## Subheading to <h4>Subheading</h4>
  html = html.replace(/^## (.*$)/gm, '<h4>$1</h4>');
  
  return html;
}

// AI Service Status Monitoring
async function checkAIStatus() {
  try {
    const response = await fetch(`${API_BASE}/ai/status`);
    if (response.ok) {
      const status = await response.json();
      updateAIStatusDisplay(status);
    }
  } catch (error) {
    console.error("Failed to check AI status:", error);
  }
}

// Update AI status display
function updateAIStatusDisplay(status) {
  const statusContainer = document.getElementById("ai-status");
  if (!statusContainer) return;
  
  let statusHTML = '<div class="ai-status-grid">';
  
  for (const [modelName, info] of Object.entries(status)) {
    const isAvailable = info.client_available && info.is_active;
    const requestsLeft = info.requests_available;
    
    statusHTML += `
      <div class="ai-model-status ${isAvailable ? 'available' : 'unavailable'}">
        <div class="model-name">${modelName}</div>
        <div class="model-provider">${info.provider}</div>
        <div class="model-status">
          <span class="status-indicator ${isAvailable ? 'online' : 'offline'}"></span>
          ${isAvailable ? 'Online' : 'Offline'}
        </div>
        <div class="model-requests">Requests: ${requestsLeft}/${info.requests_per_minute}</div>
        <div class="model-cost">$${info.cost_per_1k_tokens}/1K tokens</div>
      </div>
    `;
  }
  
  statusHTML += '</div>';
  statusContainer.innerHTML = statusHTML;
}

// Enhanced job display with better formatting
function showJob(job) {
    currentJob = job;  // This was missing - causing navigation to fail
    updateMeta();

    // Add loading state
    document.body.classList.add('loading');
    
    // Update job details
    document.getElementById("job-title").innerText = job.title || "No title";
    document.getElementById("job-company").innerText = job.company || "N/A";
    document.getElementById("job-location").innerText = job.location || "N/A";

    // URL handling
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

    // Enhanced description rendering with better HTML
    const descriptionElement = document.getElementById("job-description");
    const formattedDescription = renderMarkdownToHtml(job.description || "No description");
    descriptionElement.innerHTML = formattedDescription;
    
    // Add copy button tooltip
    descriptionElement.title = "Click 'Copy Description' to copy with formatting preserved";
    
    // Remove loading state
    setTimeout(() => {
        document.body.classList.remove('loading');
    }, 300);
}

function updateJobCounter() {
    const counterElement = document.getElementById('jobCounter');
    if (counterElement) {
        counterElement.textContent = `${currentJobIndex + 1} / ${totalJobs}`;
    }
}

function updateMeta() {
    // Update job counter instead of visited count
    updateJobCounter();
}

// Load first job on page load
async function loadFirstJob() {
    try {
        console.log('🔄 Loading first job...');
        
        // First get total count - use valid limit of 200
        const countResponse = await fetch(`${API_BASE}/jobs?page=1&limit=200`);
        if (!countResponse.ok) {
            const errorText = await countResponse.text();
            console.error('❌ Count response error:', countResponse.status, errorText);
            throw new Error(`HTTP ${countResponse.status}: ${countResponse.statusText}`);
        }
        
        // Check if response is JSON
        const contentType = countResponse.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            const responseText = await countResponse.text();
            console.error('❌ Non-JSON response:', contentType, responseText);
            throw new Error('Server returned non-JSON response');
        }
        
        const countData = await countResponse.json();
        console.log('📊 Count response:', countData);
        
        // The API returns an array directly, not {jobs: [...]}
        totalJobs = countData.filter(job => !job.hidden).length;
        console.log(`📋 Total jobs found: ${totalJobs}`);
        
        // Then get first job
        const response = await fetch(`${API_BASE}/jobs?page=1&limit=1`);
        if (!response.ok) {
            const errorText = await response.text();
            console.error('❌ First job response error:', response.status, errorText);
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        // Check if response is JSON
        const responseContentType = response.headers.get('content-type');
        if (!responseContentType || !responseContentType.includes('application/json')) {
            const responseText = await response.text();
            console.error('❌ Non-JSON response:', responseContentType, responseText);
            throw new Error('Server returned non-JSON response');
        }
        
        const data = await response.json();
        console.log('📄 First job response:', data);
        
        // The API returns an array directly, not {jobs: [...]}
        if (data && data.length > 0) {
            currentJobIndex = 0;
            showJob(data[0]);
            updateJobCounter();
            console.log('✅ First job loaded successfully');
        } else {
            showCopyError('No jobs available');
            console.log('⚠️ No jobs found in response');
        }
    } catch (error) {
        console.error('❌ Error loading first job:', error);
        showCopyError(`Failed to load first job: ${error.message}`);
    }
}

// Fetch next job, skipping hidden ones
async function loadNextJob() {
    if (!currentJob) return;
    try {
        let nextJob = null;
        let res = await fetch(`${API_BASE}/jobs/next/${currentJob.entity_id}`);
        if (res.ok) nextJob = await res.json();

        // keep fetching until we get a non-hidden job
        while (nextJob && nextJob.hidden) {
            res = await fetch(`${API_BASE}/jobs/next/${nextJob.entity_id}`);
            if (!res.ok) throw new Error("No more jobs available.");
            nextJob = await res.json();
        }

        if (nextJob) {
            currentJobIndex++;
            showJob(nextJob);
            updateJobCounter();
        } else {
            showCopyError("No more jobs available.");
        }
    } catch (err) {
        console.error("Error loading next job:", err);
        showCopyError("Error loading next job.");
    }
}

// Fetch previous job, skipping hidden ones
async function loadPrevJob() {
    if (!currentJob) return;
    try {
        let prevJob = null;
        let res = await fetch(`${API_BASE}/jobs/prev/${currentJob.entity_id}`);
        if (res.ok) prevJob = await res.json();

        while (prevJob && prevJob.hidden) {
            res = await fetch(`${API_BASE}/jobs/prev/${prevJob.entity_id}`);
            if (!res.ok) throw new Error("No previous jobs available.");
            prevJob = await res.json();
        }

        if (prevJob) {
            currentJobIndex = Math.max(0, currentJobIndex - 1);
            showJob(prevJob);
            updateJobCounter();
        } else {
            showCopyError("No previous jobs available.");
        }
    } catch (err) {
        console.error("Error loading previous job:", err);
        showCopyError("Error loading previous job.");
    }
}

// Event listeners
document.getElementById("next-btn").addEventListener("click", loadNextJob);
document.getElementById("prev-btn").addEventListener("click", loadPrevJob);

// Load first job on startup
loadFirstJob();

// Add visited job navigation
function goToVisitedJob() {
    const visitedInput = document.getElementById('visitedInput');
    const jobNumber = parseInt(visitedInput.value);
    
    console.log(`🔄 Attempting to go to job number: ${jobNumber}`);
    
    if (isNaN(jobNumber) || jobNumber < 1) {
        showCopyError('Please enter a valid job number (1 or higher)');
        console.log('⚠️ Invalid job number input:', visitedInput.value);
        return;
    }
    
    // Load the specific job by index
    loadJobByIndex(jobNumber - 1);
}

async function loadJobByIndex(index) {
    try {
        console.log(`🔄 Loading job by index: ${index}`);
        
        const response = await fetch(`${API_BASE}/jobs?page=1&limit=200`);
        if (!response.ok) {
            const errorText = await response.text();
            console.error('❌ Load by index error:', response.status, errorText);
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        // Check if response is JSON
        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            const responseText = await response.text();
            console.error('❌ Non-JSON response:', contentType, responseText);
            throw new Error('Server returned non-JSON response');
        }
        
        const data = await response.json();
        console.log('📄 Load by index response:', data);
        
        // The API returns an array directly, not {jobs: [...]}
        const jobs = data.filter(job => !job.hidden);
        totalJobs = jobs.length;
        console.log(`📋 Total jobs found: ${totalJobs}`);
        
        if (index >= 0 && index < jobs.length) {
            currentJobIndex = index;
            showJob(jobs[index]);
            updateJobCounter();
            document.getElementById('visitedInput').value = ''; // Clear input after use
            console.log(`✅ Job ${index + 1} loaded successfully`);
        } else {
            showCopyError(`Job number ${index + 1} not found. Total jobs: ${totalJobs}`);
            console.log(`⚠️ Job index ${index} out of range. Total: ${totalJobs}`);
        }
    } catch (error) {
        console.error('❌ Error loading job by index:', error);
        showCopyError(`Failed to load job: ${error.message}`);
    }
}
