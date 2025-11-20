// API endpoint
const API_URL = 'http://localhost:8000';

// DOM elements
const queryInput = document.getElementById('queryInput');
const topKInput = document.getElementById('topKInput');
const searchBtn = document.getElementById('searchBtn');
const statusDiv = document.getElementById('status');
const resultsDiv = document.getElementById('results');

// Search function
async function search() {
    const query = queryInput.value.trim();
    const topK = parseInt(topKInput.value) || 10;

    if (!query) {
        showStatus('Please enter a query', 'error');
        return;
    }

    showStatus('Searching...', 'info');
    resultsDiv.innerHTML = '';

    try {
        const response = await fetch(`${API_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query, top_k: topK }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        displayResults(data);
        showStatus(`Found ${data.results.length} results`, 'success');
    } catch (error) {
        showStatus(`Error: ${error.message}`, 'error');
        console.error('Search error:', error);
    }
}

// Display results
function displayResults(data) {
    resultsDiv.innerHTML = '';

    const queryHeader = document.createElement('h2');
    queryHeader.textContent = `Query: "${data.query}"`;
    resultsDiv.appendChild(queryHeader);

    if (data.results.length === 0) {
        const noResults = document.createElement('p');
        noResults.textContent = 'No results found';
        noResults.className = 'no-results';
        resultsDiv.appendChild(noResults);
        return;
    }

    const resultsList = document.createElement('div');
    resultsList.className = 'results-list';

    data.results.forEach((result, index) => {
        const resultItem = document.createElement('div');
        resultItem.className = 'result-item';

        const rank = document.createElement('span');
        rank.className = 'rank';
        rank.textContent = `#${index + 1}`;

        const score = document.createElement('span');
        score.className = 'score';
        score.textContent = `Score: ${result.score.toFixed(4)}`;

        const docId = document.createElement('span');
        docId.className = 'doc-id';
        docId.textContent = `ID: ${result.document_id}`;

        resultItem.appendChild(rank);
        resultItem.appendChild(docId);
        resultItem.appendChild(score);

        if (result.text) {
            const text = document.createElement('p');
            text.className = 'result-text';
            text.textContent = result.text;
            resultItem.appendChild(text);
        }

        resultsList.appendChild(resultItem);
    });

    resultsDiv.appendChild(resultsList);
}

// Show status message
function showStatus(message, type) {
    statusDiv.textContent = message;
    statusDiv.className = `status ${type}`;
}

// Event listeners
searchBtn.addEventListener('click', search);
queryInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        search();
    }
});

// Check API health on load
async function checkHealth() {
    try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
            showStatus('API connected', 'success');
        } else {
            showStatus('API not available', 'error');
        }
    } catch (error) {
        showStatus('API not available', 'error');
    }
}

checkHealth();

