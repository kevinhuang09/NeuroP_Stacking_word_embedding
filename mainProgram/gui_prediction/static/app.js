const predictButton = document.getElementById('predictButton');
const statusLabel = document.getElementById('statusLabel');
const spinner = document.getElementById('spinner');
const sequenceInput = document.getElementById('sequenceInput');
const resultTableBody = document.querySelector('#resultTable tbody');
const logBox = document.getElementById('logBox');

let pollTimer = null;
let knownLogCount = 0;

function setStatus(text, cssClass) {
  statusLabel.textContent = text;
  statusLabel.className = 'status' + (cssClass ? ' ' + cssClass : '');
}

function appendLogLines(lines) {
  for (const line of lines) {
    logBox.textContent += line + '\n';
  }
  logBox.scrollTop = logBox.scrollHeight;
}

function clearResults() {
  resultTableBody.innerHTML = '';
}

function renderResults(rows) {
  clearResults();
  for (const row of rows) {
    const tr = document.createElement('tr');
    const prob = parseFloat(row.probability).toFixed(4);
    tr.innerHTML = `<td>${row.id}</td><td>${row.sequence}</td><td>${prob}</td><td>${row.prediction}</td>`;
    resultTableBody.appendChild(tr);
  }
}

async function startPrediction() {
  const sequences = sequenceInput.value;
  if (!sequences.trim()) {
    alert('Please enter at least one amino acid sequence.');
    return;
  }

  predictButton.disabled = true;
  spinner.hidden = false;
  knownLogCount = 0;
  logBox.textContent = '';
  clearResults();
  setStatus('Predicting... (new sequences may take a few minutes, please wait)', 'running');

  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sequences }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || 'Request failed');
    }
    pollTimer = setInterval(() => pollStatus(data.job_id), 1500);
  } catch (err) {
    setStatus('Failed to start: ' + err.message, 'error');
    spinner.hidden = true;
    predictButton.disabled = false;
  }
}

async function pollStatus(jobId) {
  try {
    const response = await fetch(`/api/status/${jobId}`);
    const job = await response.json();
    if (!response.ok) {
      throw new Error(job.error || 'Status request failed');
    }

    if (job.log && job.log.length > knownLogCount) {
      appendLogLines(job.log.slice(knownLogCount));
      knownLogCount = job.log.length;
    }

    if (job.state === 'done') {
      clearInterval(pollTimer);
      renderResults(job.results);
      setStatus('Done', 'done');
      spinner.hidden = true;
      predictButton.disabled = false;
    } else if (job.state === 'error') {
      clearInterval(pollTimer);
      setStatus('Failed, see log below', 'error');
      spinner.hidden = true;
      predictButton.disabled = false;
    }
  } catch (err) {
    clearInterval(pollTimer);
    setStatus('Failed: ' + err.message, 'error');
    spinner.hidden = true;
    predictButton.disabled = false;
  }
}

predictButton.addEventListener('click', startPrediction);
