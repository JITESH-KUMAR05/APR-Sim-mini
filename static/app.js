/**
 * PaintPilot - Automated Painting Rover (APR) Web Console Controller
 * Coordinates OpenCV segmentation, 3D WebGL digital twin, 2D geometry map, and CAD exports.
 * Traced to MSME Grant INC25ETS084848 / WBS 1.2 §3.
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const $ = (id) => document.getElementById(id);

    const fileInput = $('fileInput');
    const dropZone = $('dropZone');
    const fileNameDisplay = $('fileName');
    const generateBtn = $('generate');
    const logContainer = $('log');

    // Preset sample buttons
    const sampleResidentialBtn = $('sampleResidential');
    const sampleCommercialBtn = $('sampleCommercial');

    // View tabs
    const tab3dBtn = $('tab3d');
    const tab2dBtn = $('tab2d');
    const tabTelemetryBtn = $('tabTelemetry');
    const viewport3d = $('viewport3d');
    const viewport2d = $('viewport2d');
    const telemetryView = $('telemetryView');

    // 2D Canvas
    const canvas2d = $('canvas2d');
    const ctx2d = canvas2d.getContext('2d');

    // Parameter inputs
    const widthInput = $('width');
    const heightInput = $('height');
    const sprayInput = $('sprayWidth');
    const overlapInput = $('overlap');
    const bufferInput = $('safetyBuffer');
    const sensitivityInput = $('sensitivity');

    // Value display spans
    const widthVal = $('widthValue');
    const heightVal = $('heightValue');
    const sprayVal = $('sprayValue');
    const overlapVal = $('overlapValue');
    const bufferVal = $('bufferValue');
    const sensitivityVal = $('sensitivityValue');

    // Metric elements
    const layoutValue = $('layoutValue');
    const cellsValue = $('cellsValue');
    const areaValue = $('areaValue');
    const distanceValue = $('distanceValue');
    const paintVolValue = $('paintVolValue');
    const cycleTimeValue = $('cycleTimeValue');
    const throughputValue = $('throughputValue');
    const confidenceValue = $('confidenceValue');
    const confidenceBar = $('confidenceBar');
    const activeObstacleCount = $('activeObstacleCount');

    // Overlay 3D Controls
    const resetViewBtn = $('resetViewBtn');
    const isoViewBtn = $('isoViewBtn');
    const topDownViewBtn = $('topDownViewBtn');
    const wireframeBtn = $('wireframeBtn');
    const togglePathBtn = $('togglePathBtn');
    const toggleObstaclesBtn = $('toggleObstaclesBtn');
    const simPlayBtn = $('simPlayBtn');

    // State
    let viewer3D = null;
    let roverSim = null;
    let currentData = null;
    let selectedFile = null;
    let activeSampleId = 'residential';
    let loadedImage2D = null;

    // Initialize 3D Viewer
    try {
        if (window.THREE && window.WallViewer3D) {
            viewer3D = new window.WallViewer3D('viewport3d');
            appendLog('3D WebGL engine initialized with OrbitControls', 'info');
        } else {
            appendLog('Warning: 3D engine script loading...', 'warn');
        }
    } catch (e) {
        console.error('Error initializing 3D viewer:', e);
        appendLog('3D engine fallback enabled', 'warn');
    }

    // Initialize Autonomous Rover Simulation Engine
    try {
        if (window.RoverSimulation) {
            roverSim = new window.RoverSimulation('simCanvas');
            appendLog('Autonomous Rover Simulation SITL engine ready', 'info');
        }
    } catch (e) {
        console.error('Error initializing RoverSimulation:', e);
    }

    // Bind slider labels
    widthInput.addEventListener('input', () => { widthVal.textContent = widthInput.value; });
    heightInput.addEventListener('input', () => { heightVal.textContent = heightInput.value; });
    sprayInput.addEventListener('input', () => { sprayVal.textContent = sprayInput.value; });
    overlapInput.addEventListener('input', () => { overlapVal.textContent = overlapInput.value; });
    bufferInput.addEventListener('input', () => { bufferVal.textContent = bufferInput.value; });
    sensitivityInput.addEventListener('input', () => { sensitivityVal.textContent = Number(sensitivityInput.value).toFixed(2); });

    // Preset buttons
    sampleResidentialBtn.addEventListener('click', () => {
        selectedFile = null;
        activeSampleId = 'residential';
        fileNameDisplay.textContent = 'Benchmark: Residential Exterior Facade';
        sampleResidentialBtn.classList.add('active');
        sampleCommercialBtn.classList.remove('active');
        widthInput.value = 4000; widthVal.textContent = '4000';
        heightInput.value = 2800; heightVal.textContent = '2800';
        runAnalysis();
    });

    sampleCommercialBtn.addEventListener('click', () => {
        selectedFile = null;
        activeSampleId = 'commercial';
        fileNameDisplay.textContent = 'Benchmark: Commercial Building Wall';
        sampleCommercialBtn.classList.add('active');
        sampleResidentialBtn.classList.remove('active');
        widthInput.value = 4500; widthVal.textContent = '4500';
        heightInput.value = 3000; heightVal.textContent = '3000';
        runAnalysis();
    });

    // File Upload handling
    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            selectedFile = e.target.files[0];
            activeSampleId = null;
            sampleResidentialBtn.classList.remove('active');
            sampleCommercialBtn.classList.remove('active');
            fileNameDisplay.textContent = selectedFile.name + ` (${(selectedFile.size / 1024).toFixed(0)} KB)`;
            appendLog(`Loaded custom image: ${selectedFile.name}`, 'info');
            runAnalysis();
        }
    });

    // Drag and Drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--cyan)';
    });
    dropZone.addEventListener('dragleave', () => {
        dropZone.style.borderColor = '#34636b';
    });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#34636b';
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            selectedFile = e.dataTransfer.files[0];
            activeSampleId = null;
            sampleResidentialBtn.classList.remove('active');
            sampleCommercialBtn.classList.remove('active');
            fileNameDisplay.textContent = selectedFile.name;
            runAnalysis();
        }
    });

    // View Tabs Switching
    tab3dBtn.addEventListener('click', () => switchTab('3d'));
    tab2dBtn.addEventListener('click', () => switchTab('2d'));
    tabTelemetryBtn.addEventListener('click', () => switchTab('telemetry'));

    function switchTab(tab) {
        tab3dBtn.classList.remove('active');
        tab2dBtn.classList.remove('active');
        tabTelemetryBtn.classList.remove('active');

        viewport3d.style.display = 'none';
        viewport2d.style.display = 'none';
        telemetryView.style.display = 'none';

        if (tab === '3d') {
            tab3dBtn.classList.add('active');
            viewport3d.style.display = 'block';
            if (viewer3D) viewer3D.onResize();
        } else if (tab === '2d') {
            tab2dBtn.classList.add('active');
            viewport2d.style.display = 'block';
            render2DCanvas();
        } else if (tab === 'telemetry') {
            tabTelemetryBtn.classList.add('active');
            telemetryView.style.display = 'block';
        }
    }

    // 3D Controls
    if (resetViewBtn) resetViewBtn.addEventListener('click', () => viewer3D && viewer3D.resetView());
    if (isoViewBtn) isoViewBtn.addEventListener('click', () => viewer3D && viewer3D.setIsometricView());
    if (topDownViewBtn) topDownViewBtn.addEventListener('click', () => viewer3D && viewer3D.setTopDownView());
    if (wireframeBtn) wireframeBtn.addEventListener('click', () => viewer3D && viewer3D.toggleWireframe());
    if (togglePathBtn) togglePathBtn.addEventListener('click', () => viewer3D && viewer3D.togglePath());
    if (toggleObstaclesBtn) toggleObstaclesBtn.addEventListener('click', () => viewer3D && viewer3D.toggleObstacles());
    if (simPlayBtn) simPlayBtn.addEventListener('click', () => viewer3D && viewer3D.toggleSimulation());

    // Generate Button
    generateBtn.addEventListener('click', runAnalysis);

    // Export buttons
    document.querySelectorAll('[data-export]').forEach((btn) => {
        btn.addEventListener('click', () => {
            const format = btn.dataset.export;
            downloadExport(format);
        });
    });

    function downloadExport(format) {
        appendLog(`Initiating download for ${format.toUpperCase()} export...`, 'info');
        const url = `/api/download/${format}`;
        const a = document.createElement('a');
        a.href = url;
        a.download = '';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }

    // Autonomous Simulation Modal Controls
    const simModalOverlay = $('simModalOverlay');
    const openSimBtn = $('openSimBtn');
    const closeSimBtn = $('closeSimBtn');
    const simStartBtn = $('simStartBtn');
    const simPauseBtn = $('simPauseBtn');
    const simResetBtn = $('simResetBtn');

    if (openSimBtn) {
        openSimBtn.addEventListener('click', () => {
            if (!currentData) {
                appendLog('Notice: Generating wall data before launching simulation...', 'info');
                runAnalysis();
            }
            if (simModalOverlay) {
                simModalOverlay.style.display = 'flex';
                if (roverSim && currentData) {
                    roverSim.init(currentData);
                    if (loadedImage2D && loadedImage2D.complete) {
                        roverSim.sourceImage = loadedImage2D;
                    }
                    roverSim.resize();
                    roverSim.render();
                }
            }
        });
    }

    if (closeSimBtn) {
        closeSimBtn.addEventListener('click', () => {
            if (roverSim) roverSim.stop();
            if (simModalOverlay) simModalOverlay.style.display = 'none';
        });
    }

    if (simStartBtn) {
        simStartBtn.addEventListener('click', () => {
            if (roverSim) roverSim.start();
        });
    }

    if (simPauseBtn) {
        simPauseBtn.addEventListener('click', () => {
            if (roverSim) roverSim.pause();
        });
    }

    if (simResetBtn) {
        simResetBtn.addEventListener('click', () => {
            if (roverSim) roverSim.reset();
        });
    }

    // Speed buttons
    document.querySelectorAll('.sim-speed-btn').forEach((btn) => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.sim-speed-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const speed = Number(btn.dataset.speed) || 1.0;
            if (roverSim) roverSim.setSpeed(speed);
        });
    });

    // View mode buttons
    document.querySelectorAll('.sim-mode-btn').forEach((btn) => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.sim-mode-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const mode = btn.dataset.mode || 'hybrid';
            if (roverSim) roverSim.setViewMode(mode);
        });
    });

    // Core Analysis Execution
    function runAnalysis() {
        appendLog('Starting OpenCV segmentation & CAD toolpath generation...', 'info');

        const formData = new FormData();
        formData.append('width_mm', widthInput.value);
        formData.append('height_mm', heightInput.value);
        formData.append('spray_width_mm', sprayInput.value);
        formData.append('overlap_pct', overlapInput.value);
        formData.append('safety_buffer_mm', bufferInput.value);
        formData.append('sensitivity', sensitivityInput.value);

        if (selectedFile) {
            formData.append('file', selectedFile);
        } else if (activeSampleId) {
            formData.append('sample_id', activeSampleId);
        }

        generateBtn.disabled = true;
        generateBtn.textContent = 'Processing Geometry...';

        fetch('/api/analyze', {
            method: 'POST',
            body: formData
        })
        .then((res) => {
            if (!res.ok) throw new Error(`HTTP Error ${res.status}`);
            return res.json();
        })
        .then((data) => {
            generateBtn.disabled = false;
            generateBtn.innerHTML = 'Generate Wall Map & CAD &nbsp;→';

            if (!data.success) {
                appendLog(`Error: ${data.error}`, 'warn');
                return;
            }

            currentData = data;
            updateMetrics(data);
            updateTelemetry(data);

            // Update 3D Viewer
            if (viewer3D) {
                viewer3D.loadWallData(data);
            }

            // Update Autonomous Simulation Engine
            if (roverSim) {
                roverSim.init(data);
            }

            // Pre-load image for 2D Canvas
            if (data.image_data_url) {
                loadedImage2D = new Image();
                window.loadedImage2D = loadedImage2D;
                loadedImage2D.onload = () => {
                    render2DCanvas();
                    if (roverSim) {
                        roverSim.sourceImage = loadedImage2D;
                        roverSim.render();
                    }
                };
                loadedImage2D.src = data.image_data_url;
            } else {
                render2DCanvas();
            }

            appendLog(`Model generated: ${data.obstacles.length} keep-out zones detected, ${data.waypoints.length} waypoints planned`, 'success');
            appendLog(`AutoCAD DXF & Wavefront 3D OBJ ready for export (DFT: ${data.metrics.target_dft_um} µm)`, 'info');
        })
        .catch((err) => {
            generateBtn.disabled = false;
            generateBtn.innerHTML = 'Generate Wall Map & CAD &nbsp;→';
            console.error('Analysis error:', err);
            appendLog(`Execution error: ${err.message}`, 'warn');
        });
    }

    function updateMetrics(data) {
        const { wall, obstacles, stats, metrics, confidence } = data;

        layoutValue.textContent = `${stats.num_rows} passes`;
        cellsValue.textContent = stats.waypoint_count.toLocaleString();
        areaValue.textContent = `${metrics.net_paintable_area_m2} m²`;
        distanceValue.textContent = `${stats.total_distance_m} m`;

        if (paintVolValue) paintVolValue.textContent = `${metrics.paint_volume_liters} L`;
        if (cycleTimeValue) cycleTimeValue.textContent = metrics.cycle_time_formatted;
        if (throughputValue) throughputValue.textContent = `${metrics.throughput_m2_per_hr} m²/h`;

        confidenceValue.textContent = `${confidence}%`;
        confidenceBar.style.width = `${Math.min(100, Math.max(10, confidence))}%`;

        if (activeObstacleCount) {
            activeObstacleCount.textContent = `${obstacles.length} zones`;
        }

        $('statusWallBoundary').textContent = `${wall.width_mm} × ${wall.height_mm} mm`;
        $('statusSprayDist').textContent = `${stats.spray_distance_m} m (Active)`;
    }

    function updateTelemetry(data) {
        const tbody = document.getElementById('telemetryBody');
        if (!tbody) return;

        tbody.innerHTML = '';
        const waypoints = data.waypoints || [];

        // Show up to 120 sample waypoints in telemetry view
        const sampleRate = Math.max(1, Math.floor(waypoints.length / 120));
        waypoints.forEach((wp, idx) => {
            if (idx % sampleRate !== 0 && idx !== waypoints.length - 1) return;

            const tr = document.createElement('tr');
            const sprayClass = wp.spray_active ? 'badge-spray-on' : 'badge-spray-off';
            const sprayText = wp.spray_active ? 'SPRAY ON' : 'SPRAY OFF (TRANSIT)';

            tr.innerHTML = `
                <td>#${wp.seq}</td>
                <td>${wp.x.toFixed(1)}</td>
                <td>${wp.y.toFixed(1)}</td>
                <td>${wp.z.toFixed(1)}</td>
                <td><span class="${sprayClass}">${sprayText}</span></td>
                <td>${wp.type}</td>
                <td>Row ${wp.row}</td>
            `;
            tbody.appendChild(tr);
        });
    }

    // 2D High-Resolution Canvas Rendering
    function render2DCanvas() {
        if (!currentData) return;

        const { wall, obstacles, waypoints } = currentData;
        const width = canvas2d.width;
        const height = canvas2d.height;

        ctx2d.clearRect(0, 0, width, height);

        // Dark industrial background
        ctx2d.fillStyle = '#071217';
        ctx2d.fillRect(0, 0, width, height);

        const padX = 75;
        const padY = 55;
        const drawW = width - (padX * 2);
        const drawH = height - (padY * 2);

        // Draw source image if available
        if (loadedImage2D) {
            ctx2d.save();
            ctx2d.globalAlpha = 0.28;
            ctx2d.drawImage(loadedImage2D, padX, padY, drawW, drawH);
            ctx2d.restore();
        }

        // Active paintable fill
        ctx2d.fillStyle = 'rgba(92, 225, 210, 0.06)';
        ctx2d.fillRect(padX, padY, drawW, drawH);

        // Outer wall boundary (Cyan)
        ctx2d.strokeStyle = '#5ce1d2';
        ctx2d.lineWidth = 2;
        ctx2d.strokeRect(padX, padY, drawW, drawH);

        // Scale factors: mm to 2D canvas pixels
        const scaleX = drawW / wall.width_mm;
        const scaleY = drawH / wall.height_mm;

        // Draw Obstacles (Keep-out zones)
        obstacles.forEach((obs) => {
            const ox = padX + (obs.x * scaleX);
            const oy = padY + drawH - ((obs.y + obs.h) * scaleY); // Flip Y (origin at bottom-left)
            const ow = obs.w * scaleX;
            const oh = obs.h * scaleY;

            // Obstacle fill & border (Red/Coral)
            ctx2d.fillStyle = 'rgba(250, 116, 110, 0.25)';
            ctx2d.fillRect(ox, oy, ow, oh);

            ctx2d.strokeStyle = '#fa746e';
            ctx2d.lineWidth = 2;
            ctx2d.strokeRect(ox, oy, ow, oh);

            // Safety standoff buffer outline (Dashed Yellow/Orange)
            const bufMm = wall.safety_buffer_mm || 60;
            const bx = padX + (Math.max(0, obs.x - bufMm) * scaleX);
            const by = padY + drawH - ((Math.min(wall.height_mm, obs.y + obs.h + bufMm)) * scaleY);
            const bw = (obs.w + (bufMm * 2)) * scaleX;
            const bh = (obs.h + (bufMm * 2)) * scaleY;

            ctx2d.save();
            ctx2d.setLineDash([4, 4]);
            ctx2d.strokeStyle = 'rgba(244, 184, 96, 0.5)';
            ctx2d.strokeRect(bx, by, bw, bh);
            ctx2d.restore();

            // Label
            ctx2d.fillStyle = '#fa746e';
            ctx2d.font = '10px "DM Mono", monospace';
            ctx2d.fillText(obs.label, ox + 6, oy + 16);
            ctx2d.fillStyle = '#a1b6b9';
            ctx2d.font = '9px "DM Mono", monospace';
            ctx2d.fillText(`${obs.w}×${obs.h}mm`, ox + 6, oy + 28);
        });

        // Draw Boustrophedon Toolpath
        if (waypoints && waypoints.length > 1) {
            ctx2d.lineWidth = 2;

            for (let i = 1; i < waypoints.length; i++) {
                const p1 = waypoints[i - 1];
                const p2 = waypoints[i];

                const x1 = padX + (p1.x * scaleX);
                const y1 = padY + drawH - (p1.y * scaleY);
                const x2 = padX + (p2.x * scaleX);
                const y2 = padY + drawH - (p2.y * scaleY);

                const isSpray = p1.spray_active && p2.spray_active && (p1.row === p2.row);

                ctx2d.beginPath();
                ctx2d.moveTo(x1, y1);
                ctx2d.lineTo(x2, y2);

                if (isSpray) {
                    ctx2d.strokeStyle = '#f4b860'; // Golden Amber for active spray
                    ctx2d.setLineDash([]);
                } else {
                    ctx2d.strokeStyle = 'rgba(157, 78, 221, 0.6)'; // Magenta for transit
                    ctx2d.setLineDash([3, 3]);
                }
                ctx2d.stroke();
            }
            ctx2d.setLineDash([]);

            // Draw Start / End indicators
            const first = waypoints[0];
            const last = waypoints[waypoints.length - 1];

            const fx = padX + (first.x * scaleX);
            const fy = padY + drawH - (first.y * scaleY);
            ctx2d.fillStyle = '#f4b860';
            ctx2d.beginPath();
            ctx2d.arc(fx, fy, 6, 0, Math.PI * 2);
            ctx2d.fill();
            ctx2d.font = '10px "DM Mono", monospace';
            ctx2d.fillText('START (0,0)', fx + 10, fy - 5);

            const lx = padX + (last.x * scaleX);
            const ly = padY + drawH - (last.y * scaleY);
            ctx2d.fillStyle = '#5ce1d2';
            ctx2d.beginPath();
            ctx2d.arc(lx, ly, 6, 0, Math.PI * 2);
            ctx2d.fill();
            ctx2d.fillText('END', lx + 10, ly + 14);
        }

        // Coordinate Corner Marks
        ctx2d.font = '10px "DM Mono", monospace';
        ctx2d.fillStyle = '#7d969d';
        ctx2d.fillText('X:0, Y:0 (Origin)', padX - 10, padY + drawH + 20);
        ctx2d.fillText(`X:${wall.width_mm}, Y:0`, padX + drawW - 75, padY + drawH + 20);
        ctx2d.fillText(`X:0, Y:${wall.height_mm}`, padX - 10, padY - 14);
        ctx2d.fillText(`X:${wall.width_mm}, Y:${wall.height_mm}`, padX + drawW - 85, padY - 14);
    }

    function appendLog(message, type = 'info') {
        const time = new Date().toTimeString().split(' ')[0];
        const div = document.createElement('div');
        div.className = type;
        div.innerHTML = `<b>${time}</b> &nbsp;${message}`;

        // Keep cursor at bottom
        const cursor = logContainer.querySelector('.cursor');
        if (cursor) cursor.remove();

        logContainer.appendChild(div);

        const cursorSpan = document.createElement('span');
        cursorSpan.className = 'cursor';
        cursorSpan.textContent = ' ▌';
        div.appendChild(cursorSpan);

        logContainer.scrollTop = logContainer.scrollHeight;
    }

    // Initial load
    runAnalysis();
});