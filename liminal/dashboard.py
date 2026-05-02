"""
dashboard.py — Live web dashboard for monitoring training progress.

Runs a local Flask server that shows real-time:
- Accuracy graph over loops
- Per-category heatmap
- Loss curve
- NPC conversation viewer
- Current loop status, ETA, cost tracker
- Regression warnings

Launch with: python -m liminal.dashboard --metrics-dir ./metrics
Or automatically via train.py --dashboard
"""

import json
import time
import threading
from pathlib import Path
from flask import Flask, render_template_string, jsonify

app = Flask(__name__)

METRICS_DIR = Path("./metrics")
CONVERSATIONS_DIR = Path("./metrics/conversations")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Liminal AI Training — Live Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Courier New', monospace;
            background: #0a0a0a;
            color: #e0e0e0;
            min-height: 100vh;
        }
        .header {
            background: #111;
            border-bottom: 1px solid #333;
            padding: 20px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 {
            font-size: 18px;
            color: #fff;
            letter-spacing: 2px;
        }
        .header .status {
            font-size: 13px;
            color: #888;
        }
        .header .status .live {
            color: #4caf50;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            padding: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }
        .card {
            background: #141414;
            border: 1px solid #222;
            border-radius: 8px;
            padding: 20px;
        }
        .card h2 {
            font-size: 13px;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 16px;
        }
        .card.full { grid-column: 1 / -1; }
        .stat-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #1a1a1a;
        }
        .stat-row:last-child { border-bottom: none; }
        .stat-label { color: #888; font-size: 13px; }
        .stat-value { color: #fff; font-size: 14px; font-weight: bold; }
        .stat-value.green { color: #4caf50; }
        .stat-value.red { color: #f44336; }
        .stat-value.yellow { color: #ff9800; }
        .big-number {
            font-size: 48px;
            font-weight: bold;
            text-align: center;
            padding: 20px 0;
        }
        .big-number .label {
            font-size: 13px;
            color: #888;
            display: block;
            margin-top: 8px;
        }
        .chart-container {
            position: relative;
            height: 250px;
        }
        .category-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
            gap: 8px;
        }
        .category-cell {
            background: #1a1a1a;
            border-radius: 4px;
            padding: 12px;
            text-align: center;
        }
        .category-cell .name {
            font-size: 11px;
            color: #888;
            text-transform: uppercase;
            margin-bottom: 4px;
        }
        .category-cell .accuracy {
            font-size: 24px;
            font-weight: bold;
        }
        .category-cell .change {
            font-size: 12px;
            margin-top: 4px;
        }
        .conversation {
            background: #1a1a1a;
            border-radius: 4px;
            padding: 12px;
            margin-bottom: 8px;
            font-size: 13px;
            line-height: 1.6;
            max-height: 300px;
            overflow-y: auto;
        }
        .conversation .npc { color: #ff9800; }
        .conversation .model { color: #64b5f6; }
        .conversation .correct { color: #4caf50; }
        .conversation .wrong { color: #f44336; }
        .regression-warning {
            background: #2d1b1b;
            border: 1px solid #f44336;
            border-radius: 4px;
            padding: 12px;
            margin-bottom: 8px;
            font-size: 13px;
            color: #f44336;
        }
        .no-data {
            color: #444;
            text-align: center;
            padding: 40px;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>LIMINAL AI TRAINING</h1>
        <div class="status">
            <span class="live">●</span> LIVE — refreshing every 3s
        </div>
    </div>

    <div class="grid">
        <!-- Overview stats -->
        <div class="card">
            <h2>Current Status</h2>
            <div id="status-content">
                <div class="no-data">Waiting for training data...</div>
            </div>
        </div>

        <div class="card">
            <h2>Improvement</h2>
            <div id="improvement-content">
                <div class="no-data">Waiting for baseline...</div>
            </div>
        </div>

        <!-- Accuracy chart -->
        <div class="card full">
            <h2>Accuracy Over Loops</h2>
            <div class="chart-container">
                <canvas id="accuracyChart"></canvas>
            </div>
        </div>

        <!-- Loss chart -->
        <div class="card">
            <h2>Training Loss</h2>
            <div class="chart-container">
                <canvas id="lossChart"></canvas>
            </div>
        </div>

        <!-- Pairs chart -->
        <div class="card">
            <h2>Preference Pairs Generated</h2>
            <div class="chart-container">
                <canvas id="pairsChart"></canvas>
            </div>
        </div>

        <!-- Category heatmap -->
        <div class="card full">
            <h2>Per-Category Accuracy</h2>
            <div id="category-content" class="category-grid">
                <div class="no-data">Waiting for category data...</div>
            </div>
        </div>

        <!-- Regression warnings -->
        <div class="card full" id="regression-card" style="display:none;">
            <h2>Regression Warnings</h2>
            <div id="regression-content"></div>
        </div>

        <!-- Recent NPC conversations -->
        <div class="card full">
            <h2>Recent NPC Conversations</h2>
            <div id="conversations-content">
                <div class="no-data">No conversations yet...</div>
            </div>
        </div>
    </div>

    <script>
        let accuracyChart, lossChart, pairsChart;

        function initCharts() {
            const chartDefaults = {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { color: '#222' }, ticks: { color: '#666' } },
                    y: { grid: { color: '#222' }, ticks: { color: '#666' } }
                }
            };

            accuracyChart = new Chart(document.getElementById('accuracyChart'), {
                type: 'line',
                data: { labels: [], datasets: [{
                    data: [],
                    borderColor: '#4caf50',
                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 2,
                }]},
                options: {
                    ...chartDefaults,
                    scales: {
                        ...chartDefaults.scales,
                        y: { ...chartDefaults.scales.y, min: 0, max: 1,
                             ticks: { ...chartDefaults.scales.y.ticks, callback: v => (v*100)+'%' }}
                    }
                }
            });

            lossChart = new Chart(document.getElementById('lossChart'), {
                type: 'line',
                data: { labels: [], datasets: [{
                    data: [],
                    borderColor: '#ff9800',
                    backgroundColor: 'rgba(255, 152, 0, 0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 2,
                }]},
                options: chartDefaults
            });

            pairsChart = new Chart(document.getElementById('pairsChart'), {
                type: 'bar',
                data: { labels: [], datasets: [{
                    data: [],
                    backgroundColor: '#64b5f6',
                }]},
                options: chartDefaults
            });
        }

        function colorForAccuracy(acc) {
            if (acc >= 0.8) return '#4caf50';
            if (acc >= 0.6) return '#ff9800';
            return '#f44336';
        }

        async function refresh() {
            try {
                const res = await fetch('/api/metrics');
                const data = await res.json();

                if (!data.loops || data.loops.length === 0) return;

                // Status card
                const latest = data.loops[data.loops.length - 1];
                const elapsed = data.elapsed_minutes || 0;
                const totalPairs = data.loops.reduce((s, l) => s + l.pairs_generated, 0);
                document.getElementById('status-content').innerHTML = `
                    <div class="stat-row"><span class="stat-label">Loop</span><span class="stat-value">${latest.loop} / ${data.total_loops_target || '?'}</span></div>
                    <div class="stat-row"><span class="stat-label">Current Accuracy</span><span class="stat-value">${(latest.accuracy * 100).toFixed(1)}%</span></div>
                    <div class="stat-row"><span class="stat-label">Tasks This Loop</span><span class="stat-value">${latest.tasks_attempted}</span></div>
                    <div class="stat-row"><span class="stat-label">Mistakes This Loop</span><span class="stat-value ${latest.tasks_wrong > 0 ? 'red' : 'green'}">${latest.tasks_wrong}</span></div>
                    <div class="stat-row"><span class="stat-label">Total Pairs</span><span class="stat-value">${totalPairs}</span></div>
                    <div class="stat-row"><span class="stat-label">Elapsed</span><span class="stat-value">${elapsed.toFixed(1)} min</span></div>
                    <div class="stat-row"><span class="stat-label">Loss</span><span class="stat-value">${latest.training_loss != null ? latest.training_loss.toFixed(4) : 'n/a'}</span></div>
                `;

                // Improvement card
                if (data.baseline_accuracy != null) {
                    const improvement = latest.accuracy - data.baseline_accuracy;
                    const color = improvement > 0 ? 'green' : (improvement < 0 ? 'red' : '');
                    document.getElementById('improvement-content').innerHTML = `
                        <div class="big-number ${color}">${improvement >= 0 ? '+' : ''}${(improvement * 100).toFixed(1)}%
                            <span class="label">vs baseline (${(data.baseline_accuracy * 100).toFixed(1)}%)</span>
                        </div>
                    `;
                }

                // Accuracy chart
                const labels = data.loops.map(l => l.loop);
                const accuracies = data.loops.map(l => l.accuracy);
                accuracyChart.data.labels = labels;
                accuracyChart.data.datasets[0].data = accuracies;
                accuracyChart.update('none');

                // Loss chart
                const losses = data.loops.filter(l => l.training_loss != null);
                lossChart.data.labels = losses.map(l => l.loop);
                lossChart.data.datasets[0].data = losses.map(l => l.training_loss);
                lossChart.update('none');

                // Pairs chart
                pairsChart.data.labels = labels;
                pairsChart.data.datasets[0].data = data.loops.map(l => l.pairs_generated);
                pairsChart.update('none');

                // Category heatmap
                if (data.category_improvement && Object.keys(data.category_improvement).length > 0) {
                    let catHtml = '';
                    for (const [cat, info] of Object.entries(data.category_improvement)) {
                        const color = colorForAccuracy(info.latest);
                        const changeColor = info.change > 0 ? '#4caf50' : (info.change < 0 ? '#f44336' : '#666');
                        const changeSign = info.change >= 0 ? '+' : '';
                        catHtml += `
                            <div class="category-cell">
                                <div class="name">${cat}</div>
                                <div class="accuracy" style="color:${color}">${(info.latest * 100).toFixed(0)}%</div>
                                <div class="change" style="color:${changeColor}">${changeSign}${(info.change * 100).toFixed(1)}%</div>
                            </div>
                        `;
                    }
                    document.getElementById('category-content').innerHTML = catHtml;
                }

                // Regressions
                if (data.regressions && data.regressions.length > 0) {
                    document.getElementById('regression-card').style.display = 'block';
                    document.getElementById('regression-content').innerHTML =
                        data.regressions.map(r =>
                            `<div class="regression-warning">⚠ ${r.category}: dropped from ${(r.peak * 100).toFixed(0)}% to ${(r.current * 100).toFixed(0)}% (${(r.drop * 100).toFixed(1)}% regression)</div>`
                        ).join('');
                } else {
                    document.getElementById('regression-card').style.display = 'none';
                }

                // Conversations
                const convRes = await fetch('/api/conversations');
                const convData = await convRes.json();
                if (convData.conversations && convData.conversations.length > 0) {
                    document.getElementById('conversations-content').innerHTML =
                        convData.conversations.slice(-5).reverse().map(c => `
                            <div class="conversation">
                                <div><strong>Task:</strong> ${c.task}</div>
                                <div><span class="wrong">Model said:</span> ${c.model_response}</div>
                                <div><span class="correct">Correct:</span> ${c.correct}</div>
                                <div style="margin-top:8px"><strong>Strategy:</strong> ${c.strategy}</div>
                                ${c.exchanges.map(e =>
                                    `<div><span class="npc">NPC:</span> ${e.npc}</div>
                                     <div><span class="model">Model:</span> ${e.model}</div>`
                                ).join('')}
                                <div style="margin-top:4px"><strong>Understood:</strong> ${c.understood ? '✓ Yes' : '✗ No'}</div>
                            </div>
                        `).join('');
                }
            } catch (e) {
                // silently retry on next interval
            }
        }

        initCharts();
        refresh();
        setInterval(refresh, 3000);
    </script>
</body>
</html>
"""


class DashboardServer:
    """Serves the live training dashboard."""

    def __init__(self, metrics_dir: str = "./metrics", port: int = 8420):
        self.metrics_dir = Path(metrics_dir)
        self.conversations_dir = self.metrics_dir / "conversations"
        self.port = port
        self.total_loops_target = None
        self._thread = None

    def start(self, total_loops: int | None = None):
        """Start dashboard server in background thread."""
        self.total_loops_target = total_loops
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"Dashboard running at http://localhost:{self.port}")

    def _run(self):
        metrics_dir = self.metrics_dir
        conversations_dir = self.conversations_dir
        total_target = self.total_loops_target

        @app.route("/")
        def index():
            return render_template_string(HTML_TEMPLATE)

        @app.route("/api/metrics")
        def api_metrics():
            metrics_file = metrics_dir / "training_metrics.json"
            if not metrics_file.exists():
                return jsonify({"loops": []})

            with open(metrics_file) as f:
                data = json.load(f)

            data["total_loops_target"] = total_target

            # Calculate elapsed
            if data.get("loops"):
                first_ts = data["loops"][0].get("timestamp", "")
                last_ts = data["loops"][-1].get("timestamp", "")
                if first_ts and last_ts:
                    try:
                        t0 = time.mktime(time.strptime(first_ts, "%Y-%m-%dT%H:%M:%SZ"))
                        t1 = time.mktime(time.strptime(last_ts, "%Y-%m-%dT%H:%M:%SZ"))
                        data["elapsed_minutes"] = (t1 - t0) / 60
                    except Exception:
                        data["elapsed_minutes"] = 0

            # Detect regressions
            data["regressions"] = _detect_regressions(data.get("loops", []))

            return jsonify(data)

        @app.route("/api/conversations")
        def api_conversations():
            if not conversations_dir.exists():
                return jsonify({"conversations": []})

            convos = []
            for f in sorted(conversations_dir.glob("*.json"))[-20:]:
                try:
                    with open(f) as fh:
                        convos.append(json.load(fh))
                except Exception:
                    pass

            return jsonify({"conversations": convos})

        app.run(host="0.0.0.0", port=self.port, debug=False, use_reloader=False)


def _detect_regressions(loops: list[dict], threshold: float = 0.1) -> list[dict]:
    """Detect categories where accuracy has dropped from peak."""
    if len(loops) < 5:
        return []

    # Track peak accuracy per category
    peaks = {}
    for loop_data in loops:
        for cat, results in loop_data.get("category_results", {}).items():
            total = results.get("total", 0)
            if total == 0:
                continue
            acc = results["correct"] / total
            if cat not in peaks or acc > peaks[cat]:
                peaks[cat] = acc

    # Check latest against peaks
    latest = loops[-1]
    regressions = []
    for cat, results in latest.get("category_results", {}).items():
        total = results.get("total", 0)
        if total == 0:
            continue
        current = results["correct"] / total
        peak = peaks.get(cat, current)
        drop = peak - current
        if drop >= threshold:
            regressions.append({
                "category": cat,
                "peak": round(peak, 3),
                "current": round(current, 3),
                "drop": round(drop, 3),
            })

    return regressions


def save_conversation(metrics_dir: str, task: str, model_response: str, correct: str,
                      strategy: str, exchanges: list[dict], understood: bool):
    """Save an NPC conversation for the dashboard viewer."""
    conv_dir = Path(metrics_dir) / "conversations"
    conv_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    filename = f"{ts}_{strategy}.json"

    data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "task": task[:200],
        "model_response": model_response[:200],
        "correct": correct[:200],
        "strategy": strategy,
        "exchanges": exchanges[-6:],  # last 6 exchanges
        "understood": understood,
    }

    with open(conv_dir / filename, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Liminal AI Training Dashboard")
    parser.add_argument("--metrics-dir", default="./metrics", help="Metrics directory")
    parser.add_argument("--port", type=int, default=8420, help="Dashboard port")
    args = parser.parse_args()

    METRICS_DIR = Path(args.metrics_dir)
    server = DashboardServer(metrics_dir=args.metrics_dir, port=args.port)
    server._run()  # run in foreground when called directly
