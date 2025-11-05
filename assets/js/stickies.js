(function ($) {
    var $window = $(window), $body = $('body');
    // Play initial animations on page load.
    $window.on('load', function () {
        window.setTimeout(function () {
            $body.removeClass('is-preload');
        }, 100);
    });
})(jQuery);


const notes = document.querySelectorAll('.frosted-note');
const tabs = document.querySelectorAll('.note-tab');
let maxZIndex = 2;
console.log(notes)

// ===========================================
// --- 1. Utility Functions ---
// ===========================================

// Helper function to format a date object to YYYY-MM-DD
function formatDate(date) {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
}

// Generates an array of file paths for the last 'days' count
function getDateRangeFilePaths(days) {
    const filePaths = [];
    let date = new Date();
    date.setDate(date.getDate()); // Start from yesterday to include today's data (if available)

    for (let i = 0; i < days; i++) {
        const dateString = formatDate(date);
        // Use the requested file name format
        filePaths.push(`../scripts/gold_prices/gold_prices_${dateString}.json`);
        date.setDate(date.getDate() - 1);
    }
    return filePaths.reverse(); // Display trend oldest to newest
}

// ===========================================
// --- 2. Trending Data Logic (Note 1) ---
// ===========================================

// (Previous function to load trending data remains here)
function getCurrentDateFilePath() {
    const today = new Date();
    const dateString = formatDate(today);
    // Use .json extension for trending data
    return `../scripts/hot_words/all_${dateString}.json`;
}

async function loadTrendingData() {
    const container = document.getElementById('trending-data-container');
    const latestFilePath = getCurrentDateFilePath();

    try {
        // Fetch the latest file
        const response = await fetch(latestFilePath);
        // Check if the file exists and is OK (e.g., handles 404 for a missing file)
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}. File may not exist for today's date.`);
        }

        const text = await response.text();

        // Parse line-delimited JSON and get the last (latest) record
        const records = text.trim().split('\n').filter(line => line.length > 0);
        const latestRecord = JSON.parse(records[records.length - 1]);

        let html = `<h4>Last updated: ${new Date(latestRecord.time).toLocaleTimeString()}</h4>`;

        latestRecord.results.forEach(result => {
            if (result.success && result.data && result.data.length > 0) {
                const platformTime = result.timestamp ? new Date(result.timestamp).toLocaleTimeString() : '';
                html += `<div class="trending-platform">`;
                html += `<h4>${result.platform} Hot Search ${platformTime ? `(${platformTime})` : ''}</h4>`;
                html += `<ol>`;
                result.data.slice(0, 5).forEach(item => { // Show top 5
                    const hotValue = item.hot_value ? ` (${item.hot_value})` : '';
                    const keyword = item.keyword || 'N/A';
                    html += `<li><a href="${item.link}" target="_blank">${keyword}${hotValue}</a></li>`;
                });
                html += `</ol></div>`;
            }
        });

        container.innerHTML = html;

    } catch (error) {
        console.error('Error loading trending data:', error);
        container.innerHTML = `<p style="color: red;">Failed to load trending data.</p><p>Please ensure the latest file, ${latestFilePath}, exists at the specified relative path.</p>`;
    }
}

// ===========================================
// --- 3. Gold Price Chart Logic (Note 2) ---
// ===========================================

// Global variable to hold the Chart.js instance for proper updates/destruction
let goldPriceChart = null;

/**
 * Renders the chart using Chart.js
 * @param {Array<Object>} data - Array of { date, price } objects.
 */
function renderChart(data) {
    const chartArea = document.getElementById('chart-area');
    const debugData = document.getElementById('chart-debug-data');

    // 1. Clear previous content and Canvas
    chartArea.innerHTML = '';
    debugData.innerHTML = '';

    if (data.length === 0) {
        chartArea.innerHTML = '<p style="text-align: center; margin-top: 50px;">No data files found for this period.</p>';
        debugData.innerHTML = '<p>No data found.</p>';
        return;
    }

    // 2. Create the Canvas element
    const canvas = document.createElement('canvas');
    canvas.id = 'goldPriceCanvas';
    chartArea.appendChild(canvas);

    // 3. Prepare Chart Data
    const labels = data.map(item => item.date);
    const prices = data.map(item => item.price);

    // 4. Destroy existing chart instance if it exists
    if (goldPriceChart) {
        goldPriceChart.destroy();
    }

    // 5. Render the Chart using Chart.js
    goldPriceChart = new Chart(canvas, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: '999.9 Gold Price (HKD/g)',
                data: prices,
                borderColor: 'rgb(255, 193, 7)',
                backgroundColor: 'rgba(255, 193, 7, 0.2)',
                tension: 0.2, // Makes the line slightly curved
                fill: true, // Fill area under the line
                pointRadius: 3,
                pointHoverRadius: 5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    display: true,
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45,
                        // Only show a few labels for longer periods
                        callback: function (value, index, values) {
                            if (data.length > 30) {
                                return index % 5 === 0 ? labels[index] : '';
                            }
                            return labels[index];
                        }
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Price (HKD/g)'
                    },
                    beginAtZero: false
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += context.parsed.y.toFixed(2) + ' HKD';
                            }
                            return label;
                        }
                    }
                }
            }
        }
    });

    // 6. Generate debug list (for reference)
    let html = '<ol>';
    data.forEach(item => {
        html += `<li>${item.date}: ${item.price.toFixed(2)} HKD</li>`;
    });
    html += '</ol>';
    debugData.innerHTML = html;
}

async function fetchAndAggregateData(days) {
    const filePaths = getDateRangeFilePaths(days);
    const aggregatedData = [];
    const promises = filePaths.map(async filePath => {
        const date = filePath.split('_').pop().split('.')[0]; // e.g., '2025-10-29'
        try {
            const response = await fetch(filePath);
            if (!response.ok) {
                throw new Error(`Status: ${response.status}`);
            }

            const text = await response.text();
            const records = text.trim().split('\n').filter(line => line.length > 0);
            // Use the latest record in the file, if multiple are present
            const latestRecord = JSON.parse(records[records.length - 1]);

            const zhubaohuiResult = latestRecord.results.find(r => r.platform === '周大福');

            if (zhubaohuiResult && zhubaohuiResult.success && zhubaohuiResult.data) {
                const sellingPriceItem = zhubaohuiResult.data.find(item => item.type === '999.9饰金卖出价');

                if (sellingPriceItem) {
                    // Extract numerical price per gram (e.g., "1176.63 HKD" -> 1176.63)
                    const priceString = sellingPriceItem.price_per_gram;
                    const price = parseFloat(priceString.split(' ')[0]);

                    if (!isNaN(price)) {
                        aggregatedData.push({ date, price });
                    }
                }
            }

        } catch (error) {
            // console.warn(`Could not load data for ${date}: ${error.message}`);
            // Silently skip missing/failing days, as this is expected for historical range
        }
    });

    await Promise.all(promises);

    // Sort by date before rendering
    aggregatedData.sort((a, b) => new Date(a.date) - new Date(b.date));
    console.log(aggregatedData)
    renderChart(aggregatedData);
}

// Initial load and event setup for buttons
function setupGoldPriceTracker() {
    const controls = document.getElementById('chart-controls');

    // Default load 7 days
    fetchAndAggregateData(7);

    controls.addEventListener('click', function (event) {
        const button = event.target.closest('button');
        if (button && button.dataset.window) {
            // Update active button state
            controls.querySelectorAll('button').forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');

            const days = parseInt(button.dataset.window, 10);
            fetchAndAggregateData(days);
        }
    });
}

// ===========================================
// --- 4. Tab Activation Logic (Existing) ---
// ===========================================

function activateNote(noteId) {
    notes.forEach(n => n.classList.remove('active'));
    tabs.forEach(t => t.classList.remove('active-tab'));

    const targetNote = document.querySelector(`.frosted-note[data-note="${noteId}"]`);
    if (targetNote) {
        maxZIndex++;
        if (maxZIndex > 99) maxZIndex = 100;

        targetNote.style.zIndex = maxZIndex;
        targetNote.classList.add('active');
    }

    const targetTab = document.querySelector(`.note-tab[data-note-id="${noteId}"]`);
    if (targetTab) {
        targetTab.classList.add('active-tab');
    }
}

// --- Initialize: Set the default active note/tab ---
const initialActiveNote = document.querySelector('.frosted-note.active');
if (initialActiveNote) {
    const initialNoteId = initialActiveNote.dataset.note;
    const initialActiveTab = document.querySelector(`.note-tab[data-note-id="${initialNoteId}"]`);
    if (initialActiveTab) {
        initialActiveTab.classList.add('active-tab');
    }
}

// --- Event Listener for Tab/Note Clicks ---
tabs.forEach(tab => {
    tab.addEventListener('click', function (event) {
        event.stopPropagation();
        const noteId = this.dataset.noteId;
        activateNote(noteId);
    });
});

notes.forEach(note => {
    note.addEventListener('click', function () {
        const noteId = this.dataset.note;
        activateNote(noteId);
    });
});

// ===========================================
// --- 5. Initialization ---
// ===========================================

document.addEventListener('DOMContentLoaded', () => {
    loadTrendingData();
    setupGoldPriceTracker();
});