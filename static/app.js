/**
 * APR Console controller: wall settings, the dimensioned wall plan, the 3D view, the sweep preview,
 * the rover simulator replay and the exports. Colours come from the theme tokens (static/theme.js).
 */

document.addEventListener('DOMContentLoaded', () => {
    const $ = (id) => document.getElementById(id);
    const color = (name) => window.APRTheme.color(name);

    // Source and settings
    const fileInput = $('fileInput');
    const dropZone = $('dropZone');
    const fileNameDisplay = $('fileName');
    const generateBtn = $('generate');
    const planError = $('planError');
    const sampleResidentialBtn = $('sampleResidential');
    const sampleCommercialBtn = $('sampleCommercial');
    const randomWallBtn = $('randomWallBtn');
    const widthInput = $('width');
    const heightInput = $('height');
    const sprayInput = $('sprayWidth');
    const overlapInput = $('overlap');
    const bufferInput = $('safetyBuffer');
    const sensitivityInput = $('sensitivity');
    const startCornerInput = $('startCorner');
    const directionInput = $('stripeDirection');
    const keepCorrectionsInput = $('keepCorrections');

    // Views
    const TABS = [
        { name: 'plan', tab: $('tabPlan'), panel: $('panelPlan') },
        { name: '3d', tab: $('tab3d'), panel: $('panel3d') },
        { name: 'simulate', tab: $('tabSimulate'), panel: $('panelSimulate') },
        { name: 'export', tab: $('tabExport'), panel: $('panelExport') }
    ];
    const viewport2d = $('viewport2d');
    const canvas2d = $('canvas2d');
    const ctx2d = canvas2d.getContext('2d');
    const layoutWhy = $('layoutWhy');
    const layoutAdvice = $('layoutAdvice');
    const simRoverBtn = $('simRoverBtn');
    const simReport = $('simReport');
    const simUnavailable = $('simUnavailable');
    const exportCaution = $('exportCaution');
    const exportButtons = document.querySelectorAll('[data-export]');

    // Inspector
    const inspector = $('inspector');
    const propsPanel = $('propsPanel');
    const scrim = $('scrim');
    const layoutValue = $('layoutValue');
    const cellsValue = $('cellsValue');
    const areaValue = $('areaValue');
    const distanceValue = $('distanceValue');
    const paintVolValue = $('paintVolValue');
    const cycleTimeValue = $('cycleTimeValue');
    const activeObstacleCount = $('activeObstacleCount');
    const obstacleList = $('obstacleList');
    const detectorBanner = $('detectorBanner');
    const reviewPanel = $('reviewPanel');
    const reviewCount = $('reviewCount');
    const reviewList = $('reviewList');

    // Status bar and activity
    const statusDetector = $('statusDetector');
    const statusSimulator = $('statusSimulator');
    const statusWall = $('statusWall');
    const statusLayout = $('statusLayout');
    const statusLast = $('statusLast');
    const logToggle = $('logToggle');
    const logDrawer = $('logDrawer');
    const logContainer = $('log');

    // Wall corner picker
    const cornersRow = $('cornersRow');
    const cornersBtn = $('cornersBtn');
    const cornersStatus = $('cornersStatus');
    const cornerModal = $('cornerModal');
    const cornerCanvas = $('cornerCanvas');
    const cornerHint = $('cornerHint');
    const cornerUndoBtn = $('cornerUndoBtn');
    const cornerClearBtn = $('cornerClearBtn');
    const cornerCancelBtn = $('cornerCancelBtn');
    const cornerApplyBtn = $('cornerApplyBtn');

    // State
    let viewer3D = null;
    let roverSim = null;
    let currentData = null;
    let selectedFile = null;
    let activeSampleId = 'residential';
    let loadedImage2D = null;
    let wallCorners = null;
    let cornerPreviewUrl = null;
    let replanInFlight = false;
    let simResult = null; // the last rover simulator replay; cleared whenever the plan changes
    let simGeneration = 0; // bumped whenever the plan changes, so a replay that was still running is dropped
    let activeTab = 'plan';
    let planView = null; // zoom and pan of the wall plan; null means fit to the panel
    let planGeom = null; // where the wall was last drawn, for zoom, pan and hover
    let highlightId = null;
    const layers = { photo: true, stripes: true, transit: true, keepouts: true };

    const CLASS_OPTIONS = [
        ['window', 'Window'],
        ['door', 'Door'],
        ['ac_unit', 'AC unit'],
        ['meter_panel', 'Meter panel'],
        ['pipe', 'Pipe'],
        ['grill', 'Grill']
    ];

    const CORNER_NAMES = {
        bottom_left: 'bottom left',
        bottom_right: 'bottom right',
        top_left: 'top left',
        top_right: 'top right'
    };

    // ----- Activity log -----

    function appendLog(text, kind = 'info') {
        const item = document.createElement('li');
        item.className = kind;
        const time = document.createElement('time');
        time.textContent = new Date().toTimeString().slice(0, 8);
        const message = document.createElement('span');
        message.textContent = text;
        item.append(time, message);
        logContainer.appendChild(item);
        logContainer.scrollTop = logContainer.scrollHeight;
        if (kind === 'warn' || kind === 'error') {
            statusLast.textContent = text;
            logToggle.classList.add('error');
        }
    }

    function setLast(text) {
        statusLast.textContent = text;
        logToggle.classList.remove('error');
    }

    function setLogOpen(open) {
        logDrawer.hidden = !open;
        logToggle.setAttribute('aria-expanded', String(open));
        if (open) logContainer.scrollTop = logContainer.scrollHeight;
    }

    logToggle.addEventListener('click', () => setLogOpen(logDrawer.hidden));
    $('logClose').addEventListener('click', () => setLogOpen(false));

    // ----- Engines -----

    try {
        if (window.THREE && window.WallViewer3D) viewer3D = new window.WallViewer3D('viewport3d');
        else appendLog('The 3D view could not load; the plan and exports still work', 'warn');
    } catch (e) {
        console.error('Error initializing 3D viewer:', e);
        appendLog('The 3D view could not start; the plan and exports still work', 'warn');
    }

    try {
        if (window.RoverSimulation) roverSim = new window.RoverSimulation('simCanvas');
    } catch (e) {
        console.error('Error initializing the sweep preview:', e);
    }

    // ----- What this app can do -----

    function setDetectorStatus(detector) {
        const trained = detector === 'yolo11n-onnx';
        const dot = document.createElement('span');
        dot.className = trained ? 'dot ok' : 'dot warn';
        dot.setAttribute('aria-hidden', 'true');
        statusDetector.replaceChildren(dot, trained ? 'Detector: trained model' : 'Detector: basic edge detection');
        statusDetector.title = trained
            ? 'Obstacles are found by the trained YOLO11n model'
            : 'The trained model is not loaded. Basic edge detection is less reliable on real walls.';
    }

    function setSimulatorStatus(available) {
        statusSimulator.textContent = available ? 'Physics replay: available' : 'Physics replay: local app only';
        simUnavailable.hidden = available;
        simRoverBtn.disabled = !available;
    }

    fetch('/api/capabilities')
        .then((res) => res.json())
        .then((caps) => {
            setDetectorStatus(caps.detector);
            setSimulatorStatus(Boolean(caps.simulator));
        })
        .catch(() => {
            statusDetector.textContent = 'Detector: unknown';
            statusSimulator.textContent = 'Physics replay: unknown';
        });

    // ----- Theme -----

    function updateThemeLabel() {
        const next = window.APRTheme.get() === 'dark' ? 'light' : 'dark';
        $('themeToggle').setAttribute('aria-label', `Switch to ${next} theme`);
        $('themeToggle').title = `Switch to ${next} theme`;
    }

    $('themeToggle').addEventListener('click', () => window.APRTheme.toggle());
    document.addEventListener('apr-theme-change', () => {
        updateThemeLabel();
        render2DCanvas();
        if (viewer3D && viewer3D.applyTheme) viewer3D.applyTheme();
        if (roverSim) roverSim.render();
    });
    updateThemeLabel();

    // ----- Tabs -----

    function switchTab(name) {
        activeTab = name;
        TABS.forEach((entry) => {
            const selected = entry.name === name;
            entry.tab.setAttribute('aria-selected', String(selected));
            entry.tab.tabIndex = selected ? 0 : -1;
            entry.panel.hidden = !selected;
        });
        if (name === 'plan') render2DCanvas();
        if (name === '3d' && viewer3D) viewer3D.onResize();
        if (name === 'simulate' && roverSim) {
            roverSim.resize();
            roverSim.render();
        }
        if (name !== 'plan') setAdviceOpen(false);
    }

    TABS.forEach((entry, index) => {
        entry.tab.addEventListener('click', () => switchTab(entry.name));
        entry.tab.addEventListener('keydown', (e) => {
            let next = null;
            if (e.key === 'ArrowRight') next = (index + 1) % TABS.length;
            if (e.key === 'ArrowLeft') next = (index - 1 + TABS.length) % TABS.length;
            if (e.key === 'Home') next = 0;
            if (e.key === 'End') next = TABS.length - 1;
            if (next === null) return;
            e.preventDefault();
            switchTab(TABS[next].name);
            TABS[next].tab.focus();
        });
    });

    // ----- 3D view controls -----

    $('resetViewBtn').addEventListener('click', () => viewer3D && viewer3D.resetView());
    $('isoViewBtn').addEventListener('click', () => viewer3D && viewer3D.setIsometricView());
    $('topDownViewBtn').addEventListener('click', () => viewer3D && viewer3D.setTopDownView());
    [['wireframeBtn', 'toggleWireframe'], ['togglePathBtn', 'togglePath'], ['toggleObstaclesBtn', 'toggleObstacles']].forEach(([id, method]) => {
        const btn = $(id);
        btn.addEventListener('click', () => {
            if (!viewer3D) return;
            const on = viewer3D[method]();
            btn.classList.toggle('on', on);
            btn.setAttribute('aria-pressed', String(on));
        });
    });

    // ----- Drawers on narrow screens -----

    function closeDrawers() {
        propsPanel.classList.remove('open');
        inspector.classList.remove('open');
        scrim.hidden = true;
    }

    function openDrawer(panel) {
        const wasOpen = panel.classList.contains('open');
        closeDrawers();
        if (wasOpen) return;
        panel.classList.add('open');
        scrim.hidden = false;
    }

    $('openProps').addEventListener('click', () => openDrawer(propsPanel));
    $('openInspector').addEventListener('click', () => openDrawer(inspector));
    scrim.addEventListener('click', closeDrawers);

    document.addEventListener('keydown', (e) => {
        if (e.key !== 'Escape') return;
        if (cornerModal.style.display !== 'none') { closeCornerModal(); return; }
        if (!layoutAdvice.hidden) { setAdviceOpen(false); return; }
        if (!logDrawer.hidden) { setLogOpen(false); return; }
        closeDrawers();
    });

    // ----- Settings -----

    const sliderOutputs = [
        [sprayInput, $('sprayValue'), (v) => `${v} mm`],
        [overlapInput, $('overlapValue'), (v) => `${v} %`],
        [bufferInput, $('bufferValue'), (v) => `${v} mm`],
        [sensitivityInput, $('sensitivityValue'), (v) => Number(v).toFixed(2)]
    ];
    sliderOutputs.forEach(([input, output, format]) => {
        const update = () => { output.textContent = format(input.value); };
        input.addEventListener('input', update);
        update();
    });

    function chooseSample(id, button, label, width, height) {
        selectedFile = null;
        activeSampleId = id;
        resetCorners(false);
        fileNameDisplay.textContent = label;
        [sampleResidentialBtn, sampleCommercialBtn].forEach((b) => b.classList.toggle('active', b === button));
        widthInput.value = width;
        heightInput.value = height;
        runAnalysis();
    }

    sampleResidentialBtn.addEventListener('click', () => chooseSample('residential', sampleResidentialBtn, 'Sample: residential facade', 4000, 2800));
    sampleCommercialBtn.addEventListener('click', () => chooseSample('commercial', sampleCommercialBtn, 'Sample: commercial wall', 4500, 3000));

    function useFile(file, label) {
        selectedFile = file;
        activeSampleId = null;
        resetCorners(true);
        sampleResidentialBtn.classList.remove('active');
        sampleCommercialBtn.classList.remove('active');
        fileNameDisplay.textContent = label;
    }

    // Random wall: the server draws a wall with known obstacles, then it goes through the normal upload path
    randomWallBtn.addEventListener('click', () => {
        randomWallBtn.disabled = true;
        fetch('/api/random_wall', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        })
        .then((res) => res.json())
        .then((wall) => {
            if (!wall.success) throw new Error(wall.error || 'Could not make a random wall');
            return fetch(wall.image_data_url).then((res) => res.blob()).then((blob) => ({ wall, blob }));
        })
        .then(({ wall, blob }) => {
            useFile(new File([blob], `random_wall_${wall.seed}.jpg`, { type: 'image/jpeg' }), `Random wall #${wall.seed}`);
            widthInput.value = wall.width_mm;
            heightInput.value = wall.height_mm;
            const planted = Object.entries(wall.planted).filter(([, n]) => n > 0).map(([name, n]) => `${n} ${name}`).join(', ');
            appendLog(`Random wall #${wall.seed}, ${wall.width_mm} × ${wall.height_mm} mm. Hidden in it: ${planted || 'nothing'}`);
            return runAnalysis(); // the button stays off until the analysis is done, so two walls cannot race
        })
        .catch((err) => {
            appendLog(`Random wall: ${err.message}`, 'error');
        })
        .finally(() => {
            randomWallBtn.disabled = false;
        });
    });

    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
    });
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files && e.target.files[0];
        if (!file) return;
        useFile(file, `${file.name} (${(file.size / 1024).toFixed(0)} KB)`);
        appendLog(`Photo loaded: ${file.name}`);
        runAnalysis();
    });
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        const file = e.dataTransfer.files && e.dataTransfer.files[0];
        if (!file) return;
        useFile(file, file.name);
        appendLog(`Photo loaded: ${file.name}`);
        runAnalysis();
    });

    generateBtn.addEventListener('click', runAnalysis);

    // Changing the start corner after a plan exists re-plans so the exports match the dropdown
    startCornerInput.addEventListener('change', () => {
        if (!currentData) return;
        if (replanInFlight) {
            startCornerInput.value = currentData.wall.start_corner;
            return;
        }
        replan(currentData.obstacles, startCornerInput.value);
    });

    // Changing the stripe direction after a plan exists re-plans in the same way
    directionInput.addEventListener('change', () => {
        if (!currentData || replanInFlight) return;
        replan(currentData.obstacles, startCornerInput.value);
    });

    // ----- Wall corner picker -----

    const CORNER_PROMPTS = [
        "Click the wall's top-left corner.",
        'Click the top-right corner.',
        'Click the bottom-right corner.',
        'Click the bottom-left corner.',
        'All four corners set. Apply to straighten the photo.'
    ];

    let cornerPicker = null;
    try {
        if (window.CornerPicker) cornerPicker = new window.CornerPicker(cornerCanvas, updateCornerHint);
    } catch (e) {
        console.error('Error initializing CornerPicker:', e);
        appendLog('The corner picker could not start; the whole photo is used as the wall', 'warn');
    }

    function updateCornerHint() {
        if (!cornerPicker) return;
        const count = cornerPicker.points.length;
        cornerHint.textContent = CORNER_PROMPTS[count];
        cornerApplyBtn.disabled = count !== 4;
    }

    function updateCornersStatus() {
        cornersStatus.textContent = wallCorners
            ? 'Wall corners set. The photo is straightened before detection.'
            : 'The whole photo is treated as the wall.';
    }

    function resetCorners(hasUpload) {
        wallCorners = null;
        cornersRow.hidden = !hasUpload;
        updateCornersStatus();
    }

    function closeCornerModal() {
        cornerModal.style.display = 'none';
    }

    cornersBtn.addEventListener('click', () => {
        if (!selectedFile || !cornerPicker) return;
        if (cornerPreviewUrl) URL.revokeObjectURL(cornerPreviewUrl);
        cornerPreviewUrl = URL.createObjectURL(selectedFile);
        const image = new Image();
        image.onload = () => {
            cornerPicker.setImage(image);
            cornerModal.style.display = 'flex';
        };
        image.src = cornerPreviewUrl;
    });

    cornerUndoBtn.addEventListener('click', () => cornerPicker && cornerPicker.undo());
    cornerCancelBtn.addEventListener('click', closeCornerModal);
    cornerClearBtn.addEventListener('click', () => {
        wallCorners = null;
        updateCornersStatus();
        closeCornerModal();
        runAnalysis();
    });
    cornerApplyBtn.addEventListener('click', () => {
        wallCorners = cornerPicker ? cornerPicker.getCorners() : null;
        updateCornersStatus();
        closeCornerModal();
        runAnalysis();
    });

    // ----- Plan the wall -----

    function showPlanError(text) {
        planError.textContent = text;
        planError.hidden = !text;
    }

    function runAnalysis() {
        appendLog('Planning: finding obstacles and laying out the stripes');
        showPlanError('');

        const formData = new FormData();
        formData.append('width_mm', widthInput.value);
        formData.append('height_mm', heightInput.value);
        formData.append('spray_width_mm', sprayInput.value);
        formData.append('overlap_pct', overlapInput.value);
        formData.append('safety_buffer_mm', bufferInput.value);
        formData.append('sensitivity', sensitivityInput.value);
        formData.append('start_corner', startCornerInput.value);
        formData.append('direction', directionInput.value);

        if (selectedFile) {
            formData.append('file', selectedFile);
            if (wallCorners) formData.append('corners', JSON.stringify(wallCorners));
        } else if (activeSampleId) {
            formData.append('sample_id', activeSampleId);
        }

        generateBtn.disabled = true;
        generateBtn.textContent = 'Planning…';

        return fetch('/api/analyze', { method: 'POST', body: formData })
        .then((res) => res.json().catch(() => ({})).then((body) => ({ res, body })))
        .then(({ res, body: data }) => {
            generateBtn.disabled = false;
            generateBtn.textContent = 'Plan wall';

            if (!res.ok || !data.success) {
                const reason = data.error || `The server answered ${res.status}`;
                appendLog(`Plan failed: ${reason}`, 'error');
                showPlanError(`${reason}. Check the settings and try again.`);
                // A rejected request (for example bad corners) must not keep sending the same corners
                wallCorners = null;
                updateCornersStatus();
                return;
            }

            currentData = data;
            planView = null;
            clearSimResult();
            refreshViews(data);
            showLayout(data.layout);

            loadedImage2D = null;
            if (data.image_data_url) {
                const image = new Image();
                image.onload = () => {
                    if (currentData !== data) return;
                    loadedImage2D = image;
                    render2DCanvas();
                    if (roverSim) {
                        roverSim.sourceImage = image;
                        roverSim.render();
                    }
                };
                image.src = data.image_data_url;
            }
            render2DCanvas();

            const reviewNote = data.needs_review ? `, ${data.needs_review} to check` : '';
            appendLog(`Planned: ${data.obstacles.length} keep-outs${reviewNote}, ${data.stats.num_rows} stripes, ${data.waypoints.length} waypoints`, 'success');
            markPlanned(data);
        })
        .catch((err) => {
            generateBtn.disabled = false;
            generateBtn.textContent = 'Plan wall';
            console.error('Analysis error:', err);
            appendLog(`Plan failed: ${err.message}`, 'error');
            showPlanError('The server could not be reached. Check that the app is running and try again.');
            wallCorners = null;
            updateCornersStatus();
        });
    }

    function markPlanned(data) {
        const now = new Date().toTimeString().slice(0, 5);
        setLast(`Planned ${now} · ${data.waypoints.length} waypoints`);
    }

    function replan(obstacles, startCorner = currentData.wall.start_corner) {
        const wall = currentData.wall;
        appendLog('Re-planning with your changes');
        replanInFlight = true;
        reviewList.querySelectorAll('button, select').forEach((el) => { el.disabled = true; });

        fetch('/api/replan', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                width_mm: wall.width_mm,
                height_mm: wall.height_mm,
                spray_width_mm: wall.spray_width_mm,
                overlap_pct: wall.overlap_pct,
                safety_buffer_mm: wall.safety_buffer_mm,
                start_corner: startCorner,
                direction: directionInput.value,
                review_token: currentData.review_token,
                save_corrections: keepCorrectionsInput.checked,
                obstacles
            })
        })
        .then((res) => res.json().then((body) => ({ ok: res.ok, body })))
        .then(({ ok, body }) => {
            if (!ok || !body.success) throw new Error(body.error || 'Re-plan failed');
            currentData = {
                ...currentData,
                wall: { ...currentData.wall, start_corner: body.layout.start_corner, direction: body.layout.direction },
                layout: body.layout,
                obstacles: body.obstacles,
                waypoints: body.waypoints,
                stats: body.stats,
                metrics: body.metrics,
                needs_review: body.needs_review
            };
            clearSimResult();
            refreshViews(currentData);
            showLayout(body.layout);
            if (body.correction_saved) appendLog('Your review was saved as a labelled photo in exports/corrections on this computer, for the next training round');
            render2DCanvas();
            appendLog(`Re-planned: ${body.obstacles.length} keep-outs, ${body.stats.num_rows} stripes, ${body.waypoints.length} waypoints`, 'success');
            markPlanned(currentData);
            replanInFlight = false;
        })
        .catch((err) => {
            appendLog(`Re-plan failed: ${err.message}`, 'error');
            startCornerInput.value = currentData.wall.start_corner;
            replanInFlight = false;
            reviewList.querySelectorAll('button, select').forEach((el) => { el.disabled = false; });
        });
    }

    function resolveObstacle(id, newType) {
        if (!currentData || replanInFlight) return;
        const obstacles = currentData.obstacles
            .map((obs) => (obs.id === id ? (newType ? { ...obs, type: newType } : null) : obs))
            .filter(Boolean);
        replan(obstacles, startCornerInput.value);
    }

    function refreshViews(data) {
        updateMetrics(data);
        updateObstacles(data);
        updateReviewUI(data);
        updateWaypoints(data);
        exportButtons.forEach((btn) => { btn.disabled = false; });
        if (viewer3D) viewer3D.loadWallData(data);
        if (roverSim) roverSim.init(data);
    }

    // ----- Inspector -----

    function layoutText(layout) {
        if (!layout) return '';
        return `${layout.direction === 'vertical' ? 'Vertical' : 'Horizontal'} from ${CORNER_NAMES[layout.start_corner] || layout.start_corner}`;
    }

    function updateMetrics(data) {
        const { wall, stats, metrics } = data;
        layoutValue.textContent = String(stats.num_rows);
        cellsValue.textContent = stats.waypoint_count.toLocaleString();
        areaValue.textContent = `${metrics.net_paintable_area_m2} m²`;
        distanceValue.textContent = `${stats.total_distance_m} m`;
        paintVolValue.textContent = `${metrics.paint_volume_liters} L`;
        cycleTimeValue.textContent = metrics.cycle_time_formatted;
        cycleTimeValue.title = `At the nominal speed of ${metrics.nominal_speed_mps} m/s`;
        $('statusWallBoundary').textContent = `${wall.width_mm} × ${wall.height_mm} mm`;
        $('statusSprayDist').textContent = `${stats.spray_distance_m} m`;
        statusWall.textContent = `${wall.width_mm} × ${wall.height_mm} mm`;
        statusLayout.textContent = data.layout ? `${layoutText(data.layout)} · ${stats.num_rows} stripes` : '';
    }

    function obstacleName(obs) {
        return obs.type === 'unverified' ? `${obs.label} (unsure)` : obs.label;
    }

    function updateObstacles(data) {
        activeObstacleCount.textContent = String(data.obstacles.length);
        if (!data.obstacles.length) {
            const empty = document.createElement('li');
            empty.className = 'empty';
            empty.textContent = 'No keep-outs. The whole wall is painted.';
            obstacleList.replaceChildren(empty);
            return;
        }
        obstacleList.replaceChildren(...data.obstacles.map((obs) => {
            const item = document.createElement('li');
            item.className = 'obstacle-item';
            item.dataset.id = obs.id;
            const swatch = document.createElement('span');
            swatch.className = obs.type === 'unverified' ? 'swatch unsure' : 'swatch';
            const name = document.createElement('span');
            name.className = 'obstacle-name';
            name.textContent = obstacleName(obs);
            const size = document.createElement('span');
            size.className = 'obstacle-size';
            size.textContent = `${Math.round(obs.w)} × ${Math.round(obs.h)}`;
            item.append(swatch, name, size);
            item.addEventListener('mouseenter', () => setHighlight(obs.id));
            item.addEventListener('mouseleave', () => setHighlight(null));
            return item;
        }));
    }

    function updateReviewUI(data) {
        const pending = data.obstacles.filter((obs) => obs.type === 'unverified');

        detectorBanner.hidden = data.detector !== 'classical-fallback';
        reviewPanel.hidden = pending.length === 0;
        exportCaution.hidden = pending.length === 0;
        reviewCount.textContent = String(pending.length);
        exportCaution.textContent =
            `${pending.length} obstacle${pending.length === 1 ? '' : 's'} not checked yet. ` +
            'The plan keeps clear of them, so the exports are safe to use but may leave some wall unpainted.';

        reviewList.replaceChildren(...pending.map((obs) => {
            const row = document.createElement('div');
            row.className = 'review-item';
            row.addEventListener('mouseenter', () => setHighlight(obs.id));
            row.addEventListener('mouseleave', () => setHighlight(null));

            const head = document.createElement('div');
            head.className = 'review-head';
            const name = document.createElement('span');
            name.textContent = obs.label;
            const meta = document.createElement('span');
            meta.className = 'mono';
            meta.textContent = `${Math.round((obs.confidence || 0) * 100)}% sure · ${Math.round(obs.w)} × ${Math.round(obs.h)}`;
            head.append(name, meta);

            const select = document.createElement('select');
            select.className = 'field';
            select.setAttribute('aria-label', `What ${obs.label} is`);
            CLASS_OPTIONS.forEach(([value, text]) => {
                const option = document.createElement('option');
                option.value = value;
                option.textContent = text;
                if (value === obs.guess) option.selected = true;
                select.appendChild(option);
            });

            const keepBtn = document.createElement('button');
            keepBtn.type = 'button';
            keepBtn.className = 'btn btn-small';
            keepBtn.textContent = 'Keep clear';
            keepBtn.title = 'It is an obstacle of this type; paint stays clear of it';
            keepBtn.addEventListener('click', () => resolveObstacle(obs.id, select.value));

            const removeBtn = document.createElement('button');
            removeBtn.type = 'button';
            removeBtn.className = 'btn btn-small';
            removeBtn.textContent = 'Not an obstacle';
            removeBtn.title = 'Paint over this area';
            removeBtn.addEventListener('click', () => resolveObstacle(obs.id, null));

            const actions = document.createElement('div');
            actions.className = 'review-actions';
            actions.append(select, keepBtn, removeBtn);

            row.append(head, actions);
            return row;
        }));
    }

    function setHighlight(id) {
        if (highlightId === id) return;
        highlightId = id;
        obstacleList.querySelectorAll('.obstacle-item').forEach((item) => {
            item.classList.toggle('hot', id !== null && item.dataset.id === String(id));
        });
        render2DCanvas();
    }

    // ----- Why this layout -----

    function setAdviceOpen(open) {
        layoutAdvice.hidden = !open;
        layoutWhy.setAttribute('aria-expanded', String(open));
    }

    layoutWhy.addEventListener('click', () => setAdviceOpen(layoutAdvice.hidden));

    // Why the planner chose this corner and direction, with every layout it scored
    function showLayout(layout) {
        layoutAdvice.replaceChildren();
        layoutWhy.disabled = !layout;
        if (!layout) {
            setAdviceOpen(false);
            return;
        }
        const reason = document.createElement('p');
        reason.textContent = layout.reason;
        const table = document.createElement('table');
        const head = table.createTHead().insertRow();
        ['#', 'Layout', 'Crossings', 'Moves, m', 'Turns'].forEach((text) => {
            const th = document.createElement('th');
            th.textContent = text;
            head.appendChild(th);
        });
        const tbody = table.createTBody();
        layout.options.forEach((option) => {
            const row = tbody.insertRow();
            if (option.start_corner === layout.start_corner && option.direction === layout.direction) row.className = 'chosen';
            [option.rank, layoutText(option), option.crossings, option.transit_m.toFixed(1), option.turns].forEach((value) => {
                row.insertCell().textContent = String(value);
            });
        });
        layoutAdvice.append(reason, table);
    }

    // ----- Export -----

    exportButtons.forEach((btn) => {
        btn.disabled = true;
        btn.addEventListener('click', () => downloadExport(btn.dataset.export));
    });

    function downloadExport(format) {
        appendLog(`Downloading ${format.toUpperCase()}`);
        const a = document.createElement('a');
        a.href = `/api/download/${format}`;
        a.download = '';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }

    function updateWaypoints(data) {
        const tbody = $('telemetryBody');
        const waypoints = data.waypoints || [];
        const step = Math.max(1, Math.ceil(waypoints.length / 400));
        const rows = [];
        waypoints.forEach((wp, idx) => {
            if (idx % step !== 0 && idx !== waypoints.length - 1) return;
            const tr = document.createElement('tr');
            [wp.seq, wp.x.toFixed(0), wp.y.toFixed(0), wp.spray_active ? 'On' : 'Off',
                String(wp.type).toLowerCase().replace(/_/g, ' '), wp.row].forEach((value, col) => {
                const td = tr.insertCell();
                td.textContent = String(value);
                if (col === 3 && wp.spray_active) td.className = 'spray-on';
            });
            rows.push(tr);
        });
        tbody.replaceChildren(...rows);
        $('waypointNote').textContent = step > 1
            ? `· every ${step}th of ${waypoints.length}; the CSV has all of them`
            : `· ${waypoints.length}`;
    }

    // ----- Sweep preview -----

    const simStartBtn = $('simStartBtn');
    const simPauseBtn = $('simPauseBtn');
    const simResetBtn = $('simResetBtn');
    simStartBtn.addEventListener('click', () => roverSim && roverSim.start());
    simPauseBtn.addEventListener('click', () => roverSim && roverSim.pause());
    simResetBtn.addEventListener('click', () => roverSim && roverSim.reset());
    document.querySelectorAll('.sim-speed-btn').forEach((btn) => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.sim-speed-btn').forEach((b) => b.classList.toggle('active', b === btn));
            if (roverSim) roverSim.setSpeed(Number(btn.dataset.speed) || 1);
        });
    });

    // ----- Physics replay -----

    function clearSimResult() {
        simGeneration += 1;
        simResult = null;
        simReport.hidden = true;
        simReport.replaceChildren();
    }

    // Replay the exported rover mission in the C++ rover simulator and draw what it did
    simRoverBtn.addEventListener('click', () => {
        if (!currentData || replanInFlight) return;
        simRoverBtn.disabled = true;
        const generation = simGeneration;
        const plain = (text) => String(text).replace(/[<>&]/g, ' '); // server text, kept plain in the log
        appendLog('Replaying the mission in the rover simulator');
        fetch('/api/simulate_rover', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ slip: 0.03, gyro_bias: 0.01, seed: 1 })
        })
        .then((res) => res.json().then((body) => ({ status: res.status, body })))
        .then(({ status, body }) => {
            simRoverBtn.disabled = false;
            if (generation !== simGeneration) {
                appendLog('Replay result dropped: the plan changed while it ran');
                return;
            }
            if (!body.success) {
                if (status === 503) setSimulatorStatus(false);
                appendLog(`Rover simulator: ${plain(body.error)}`, 'warn');
                return;
            }
            simResult = body;
            showSimReport(body);
            render2DCanvas();
            appendLog(body.ok ? 'Replay finished; the true path is drawn on the plan' : `Replay stopped: ${plain(body.error)}`, body.ok ? 'success' : 'warn');
        })
        .catch((err) => {
            simRoverBtn.disabled = false;
            appendLog(`Rover simulator: ${plain(err.message)}`, 'error');
        });
    });

    function showSimReport(result) {
        const summary = result.summary;
        const verdict = document.createElement('p');
        verdict.className = result.ok ? 'verdict good' : 'verdict bad';
        verdict.textContent = result.ok ? 'Run finished' : `Stopped: ${result.error}`;

        const facts = [
            ['Painted', `${(summary.painted_mm / 1000).toFixed(1)} of ${(summary.planned_spray_mm / 1000).toFixed(1)} m`],
            ['Worst true error', `x ${summary.max_truth_error_mm.x.toFixed(0)}, y ${summary.max_truth_error_mm.y.toFixed(0)} mm`],
            ['Worst believed error', `x ${summary.max_reported_error_mm.x.toFixed(0)}, y ${summary.max_reported_error_mm.y.toFixed(0)} mm`],
            ['Simulated time', `${summary.duration_s.toFixed(0)} s`],
            ['Wheel slip', `${(summary.slip * 100).toFixed(0)}%`],
            ['Gyro drift', `${summary.gyro_bias_deg_s} °/s`]
        ];
        const list = document.createElement('dl');
        list.className = 'readouts';
        facts.forEach(([label, value]) => {
            const row = document.createElement('div');
            const dt = document.createElement('dt');
            dt.textContent = label;
            const dd = document.createElement('dd');
            dd.className = 'mono';
            dd.textContent = value;
            row.append(dt, dd);
            list.appendChild(row);
        });

        const show = document.createElement('button');
        show.type = 'button';
        show.className = 'btn btn-quiet';
        show.textContent = 'Show on the plan';
        show.addEventListener('click', () => switchTab('plan'));

        const note = document.createElement('p');
        note.className = 'hint';
        note.textContent = 'On the plan, the solid blue line is where the rover truly went; the dashed one is where it believed it was.';

        simReport.replaceChildren(verdict, list, note, show);
        simReport.hidden = false;
    }

    // ----- Plan layers, zoom and pan -----

    [['layerPhoto', 'photo'], ['layerStripes', 'stripes'], ['layerTransit', 'transit'], ['layerKeepouts', 'keepouts']].forEach(([id, key]) => {
        const btn = $(id);
        btn.addEventListener('click', () => {
            layers[key] = !layers[key];
            btn.classList.toggle('on', layers[key]);
            btn.setAttribute('aria-pressed', String(layers[key]));
            render2DCanvas();
        });
    });

    $('fitBtn').addEventListener('click', () => {
        planView = null;
        render2DCanvas();
    });

    function currentView() {
        return planView || (planGeom ? { s: planGeom.s, x0: planGeom.x0, y0: planGeom.y0 } : null);
    }

    canvas2d.addEventListener('wheel', (e) => {
        const view = currentView();
        if (!view || !planGeom) return;
        e.preventDefault();
        const rect = canvas2d.getBoundingClientRect();
        const cx = e.clientX - rect.left;
        const cy = e.clientY - rect.top;
        const factor = Math.exp(-e.deltaY * 0.0015);
        const s = Math.min(planGeom.fit * 20, Math.max(planGeom.fit * 0.5, view.s * factor));
        const k = s / view.s;
        planView = { s, x0: cx - (cx - view.x0) * k, y0: cy - (cy - view.y0) * k };
        render2DCanvas();
    }, { passive: false });

    let drag = null;
    canvas2d.addEventListener('pointerdown', (e) => {
        const view = currentView();
        if (!view) return;
        drag = { x: e.clientX, y: e.clientY, view };
        canvas2d.setPointerCapture(e.pointerId);
        canvas2d.style.cursor = 'grabbing';
    });
    canvas2d.addEventListener('pointermove', (e) => {
        if (drag) {
            planView = { s: drag.view.s, x0: drag.view.x0 + e.clientX - drag.x, y0: drag.view.y0 + e.clientY - drag.y };
            render2DCanvas();
            return;
        }
        if (!planGeom || !currentData || !layers.keepouts) return;
        const rect = canvas2d.getBoundingClientRect();
        const x = (e.clientX - rect.left - planGeom.x0) / planGeom.s;
        const y = currentData.wall.height_mm - (e.clientY - rect.top - planGeom.y0) / planGeom.s;
        const hit = currentData.obstacles.find((obs) => x >= obs.x && x <= obs.x + obs.w && y >= obs.y && y <= obs.y + obs.h);
        setHighlight(hit ? hit.id : null);
    });
    const endDrag = () => {
        drag = null;
        canvas2d.style.cursor = '';
    };
    canvas2d.addEventListener('pointerup', endDrag);
    canvas2d.addEventListener('pointercancel', endDrag);
    canvas2d.addEventListener('pointerleave', () => { if (!drag) setHighlight(null); });

    // ----- The wall plan drawing -----

    const MONO = '"IBM Plex Mono", ui-monospace, monospace';
    const SANS = '"IBM Plex Sans", system-ui, sans-serif';

    function labelBox(ctx, text, x, y, align, colours) {
        ctx.font = `500 10px ${MONO}`;
        const w = ctx.measureText(text).width + 8;
        const left = align === 'right' ? x - w : align === 'center' ? x - w / 2 : x;
        ctx.fillStyle = colours.bg;
        ctx.fillRect(left, y - 8, w, 16);
        if (colours.border) {
            ctx.strokeStyle = colours.border;
            ctx.lineWidth = 1;
            ctx.strokeRect(left + 0.5, y - 7.5, w - 1, 15);
        }
        ctx.fillStyle = colours.text;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, left + 4, y + 0.5);
    }

    // A dimension line with extension lines and architectural ticks, the length written on it
    function dimension(ctx, ax, ay, bx, by, text, vertical, c) {
        ctx.save();
        ctx.strokeStyle = c.muted;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
        [[ax, ay], [bx, by]].forEach(([x, y]) => {
            ctx.moveTo(x - 4, y + 4);
            ctx.lineTo(x + 4, y - 4);
        });
        ctx.stroke();
        ctx.font = `500 11px ${MONO}`;
        const w = ctx.measureText(text).width + 10;
        ctx.translate((ax + bx) / 2, (ay + by) / 2);
        if (vertical) ctx.rotate(-Math.PI / 2);
        ctx.fillStyle = c.canvas;
        ctx.fillRect(-w / 2, -8, w, 16);
        ctx.fillStyle = c.text;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, 0, 0.5);
        ctx.restore();
    }

    function niceScaleLength(s) {
        const options = [100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000];
        let best = options[0];
        options.forEach((mm) => { if (mm * s <= 140) best = mm; });
        return best;
    }

    function render2DCanvas() {
        const cssW = viewport2d.clientWidth;
        const cssH = viewport2d.clientHeight;
        if (!cssW || !cssH) return;
        const dpr = window.devicePixelRatio || 1;
        if (canvas2d.width !== Math.round(cssW * dpr) || canvas2d.height !== Math.round(cssH * dpr)) {
            canvas2d.width = Math.round(cssW * dpr);
            canvas2d.height = Math.round(cssH * dpr);
        }
        const ctx = ctx2d;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        const c = {
            canvas: color('canvas'), wall: color('wall'), strong: color('strong'), line: color('line'),
            text: color('text'), muted: color('muted'), accent: color('accent'), keep: color('keep'),
            review: color('review'), trace: color('trace'), panel: color('panel')
        };
        ctx.fillStyle = c.canvas;
        ctx.fillRect(0, 0, cssW, cssH);

        if (!currentData) {
            ctx.fillStyle = c.muted;
            ctx.font = `400 13px ${SANS}`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('Planning the sample wall…', cssW / 2, cssH / 2);
            return;
        }

        const { wall, obstacles, waypoints } = currentData;
        const W = wall.width_mm;
        const H = wall.height_mm;
        const narrow = cssW < 560;
        const margin = narrow ? { top: 56, right: 16, bottom: 56, left: 48 } : { top: 64, right: 40, bottom: 84, left: 76 };
        const availW = Math.max(40, cssW - margin.left - margin.right);
        const availH = Math.max(40, cssH - margin.top - margin.bottom);
        const fit = Math.min(availW / W, availH / H);
        const view = planView || { s: fit, x0: margin.left + (availW - W * fit) / 2, y0: margin.top + (availH - H * fit) / 2 };
        const { s, x0, y0 } = view;
        planGeom = { s, x0, y0, fit };
        const X = (mm) => x0 + mm * s;
        const Y = (mm) => y0 + (H - mm) * s;
        const wallW = W * s;
        const wallH = H * s;

        // Wall surface and photo
        ctx.fillStyle = c.wall;
        ctx.fillRect(x0, y0, wallW, wallH);
        if (layers.photo && loadedImage2D) {
            ctx.save();
            ctx.globalAlpha = window.APRTheme.get() === 'dark' ? 0.34 : 0.42;
            ctx.drawImage(loadedImage2D, x0, y0, wallW, wallH);
            ctx.restore();
        }

        // Stripes: the band the nozzle covers, and its centre line
        const segments = [];
        for (let i = 1; i < waypoints.length; i++) {
            const a = waypoints[i - 1];
            const b = waypoints[i];
            segments.push({ a, b, spray: a.spray_active && b.spray_active && a.row === b.row });
        }
        ctx.lineCap = 'butt';
        if (layers.stripes) {
            ctx.save();
            ctx.beginPath();
            ctx.rect(x0, y0, wallW, wallH);
            ctx.clip();
            ctx.strokeStyle = c.accent;
            ctx.globalAlpha = window.APRTheme.get() === 'dark' ? 0.1 : 0.14;
            ctx.lineWidth = Math.max(1, wall.spray_width_mm * s);
            segments.filter((seg) => seg.spray).forEach(({ a, b }) => {
                ctx.beginPath();
                ctx.moveTo(X(a.x), Y(a.y));
                ctx.lineTo(X(b.x), Y(b.y));
                ctx.stroke();
            });
            ctx.restore();
            ctx.strokeStyle = c.accent;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            segments.filter((seg) => seg.spray).forEach(({ a, b }) => {
                ctx.moveTo(X(a.x), Y(a.y));
                ctx.lineTo(X(b.x), Y(b.y));
            });
            ctx.stroke();
        }

        // Moves with the spray off
        if (layers.transit) {
            ctx.save();
            ctx.strokeStyle = c.muted;
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            segments.filter((seg) => !seg.spray).forEach(({ a, b }) => {
                ctx.moveTo(X(a.x), Y(a.y));
                ctx.lineTo(X(b.x), Y(b.y));
            });
            ctx.stroke();
            ctx.restore();
        }

        // Keep-outs: the obstacle, its margin, its name and size
        if (layers.keepouts) {
            const margin_mm = wall.safety_buffer_mm || 0;
            const placedLabels = [];
            const overlaps = (r) => placedLabels.some((p) => r.x < p.x + p.w && r.x + r.w > p.x && r.y < p.y + p.h && r.y + r.h > p.y);
            obstacles.forEach((obs) => {
                const unsure = obs.type === 'unverified';
                const hot = highlightId !== null && obs.id === highlightId;
                const tone = unsure ? c.review : c.keep;
                const ox = X(obs.x);
                const oy = Y(obs.y + obs.h);
                const ow = obs.w * s;
                const oh = obs.h * s;

                ctx.save();
                ctx.globalAlpha = 0.5;
                ctx.strokeStyle = tone;
                ctx.lineWidth = 1;
                ctx.setLineDash([2, 3]);
                const bx = X(Math.max(0, obs.x - margin_mm));
                const by = Y(Math.min(H, obs.y + obs.h + margin_mm));
                const bw = X(Math.min(W, obs.x + obs.w + margin_mm)) - bx;
                const bh = Y(Math.max(0, obs.y - margin_mm)) - by;
                ctx.strokeRect(bx + 0.5, by + 0.5, bw, bh);
                ctx.restore();

                ctx.save();
                ctx.globalAlpha = hot ? 0.3 : (unsure ? 0.08 : 0.16);
                ctx.fillStyle = tone;
                ctx.fillRect(ox, oy, ow, oh);
                ctx.globalAlpha = 1;
                ctx.strokeStyle = tone;
                ctx.lineWidth = hot ? 2.5 : 1.25;
                if (unsure) ctx.setLineDash([5, 3]);
                ctx.strokeRect(ox, oy, ow, oh);
                ctx.restore();

                const labelRect = { x: ox, y: oy, w: Math.min(ow, 150), h: 32 };
                if (ow >= 78 && oh >= 34 && !overlaps(labelRect)) {
                    placedLabels.push(labelRect);
                    ctx.textAlign = 'left';
                    ctx.textBaseline = 'alphabetic';
                    ctx.fillStyle = tone;
                    ctx.font = `500 11px ${SANS}`;
                    ctx.fillText(obs.label, ox + 6, oy + 15, ow - 12);
                    ctx.fillStyle = c.text;
                    ctx.font = `400 10px ${MONO}`;
                    ctx.fillText(`${Math.round(obs.w)} × ${Math.round(obs.h)}`, ox + 6, oy + 28, ow - 12);
                }
            });
        }

        // Stripe numbers in painting order, at the start of each stripe
        if (layers.stripes) {
            const firsts = [];
            const seen = new Set();
            waypoints.forEach((wp) => {
                if (wp.spray_active && !seen.has(wp.row)) {
                    seen.add(wp.row);
                    firsts.push(wp);
                }
            });
            const pitch = wall.spray_width_mm * (1 - (wall.overlap_pct || 0) / 100) * s;
            const every = Math.max(1, Math.ceil(18 / Math.max(1, pitch)));
            const vertical = (currentData.wall.direction || (currentData.layout && currentData.layout.direction)) === 'vertical';
            firsts.forEach((wp, index) => {
                if (index % every !== 0 && index !== firsts.length - 1) return;
                const x = X(wp.x);
                const y = Y(wp.y);
                const inward = vertical ? 0 : (x < x0 + wallW / 2 ? 10 : -10);
                const dy = vertical ? (y > y0 + wallH / 2 ? -12 : 12) : 0;
                labelBox(ctx, String(index + 1), x + inward, y + dy, vertical ? 'center' : (inward >= 0 ? 'left' : 'right'),
                    { bg: c.canvas, border: c.accent, text: c.text });
            });
        }

        // Start and end of the path
        if (waypoints.length > 1 && (layers.stripes || layers.transit)) {
            const first = waypoints[0];
            const last = waypoints[waypoints.length - 1];
            ctx.fillStyle = c.accent;
            ctx.beginPath();
            ctx.arc(X(first.x), Y(first.y), 4.5, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = c.text;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(X(last.x), Y(last.y), 4.5, 0, Math.PI * 2);
            ctx.stroke();
        }

        // Rover simulator replay: where the rover truly went (solid) and where it believed it was (dashed)
        if (simResult && simResult.steps && simResult.steps.length) {
            const trace = (poses, dash, alpha) => {
                ctx.save();
                ctx.globalAlpha = alpha;
                ctx.beginPath();
                poses.forEach((pose, index) => {
                    if (index === 0) ctx.moveTo(X(pose.x), Y(pose.y)); else ctx.lineTo(X(pose.x), Y(pose.y));
                });
                ctx.setLineDash(dash);
                ctx.strokeStyle = c.trace;
                ctx.lineWidth = 1.75;
                ctx.stroke();
                ctx.restore();
            };
            trace([simResult.start, ...simResult.steps.map((step) => step.reported)], [5, 4], 0.7);
            trace([simResult.start, ...simResult.steps.map((step) => step.truth)], [], 1);
        }

        // Wall outline
        ctx.strokeStyle = c.strong;
        ctx.lineWidth = 1;
        ctx.strokeRect(x0 + 0.5, y0 + 0.5, wallW - 1, wallH - 1);

        // Dimensions in mm
        const bottom = y0 + wallH;
        const dimY = bottom + (narrow ? 18 : 26);
        const dimX = x0 - (narrow ? 18 : 26);
        ctx.strokeStyle = c.muted;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x0, bottom + 5); ctx.lineTo(x0, dimY + 6);
        ctx.moveTo(x0 + wallW, bottom + 5); ctx.lineTo(x0 + wallW, dimY + 6);
        ctx.moveTo(x0 - 5, y0); ctx.lineTo(dimX - 6, y0);
        ctx.moveTo(x0 - 5, bottom); ctx.lineTo(dimX - 6, bottom);
        ctx.stroke();
        dimension(ctx, x0, dimY, x0 + wallW, dimY, `${W} mm`, false, c);
        dimension(ctx, dimX, bottom, dimX, y0, `${H} mm`, true, c);

        // Scale bar
        const barMm = niceScaleLength(s);
        const barPx = barMm * s;
        const barX = cssW - (narrow ? 16 : 24) - barPx;
        const barY = cssH - (narrow ? 18 : 22);
        ctx.fillStyle = c.muted;
        ctx.strokeStyle = c.muted;
        ctx.lineWidth = 1;
        for (let i = 0; i < 4; i++) {
            const segX = barX + (barPx / 4) * i;
            if (i % 2 === 0) ctx.fillRect(segX, barY, barPx / 4, 4);
        }
        ctx.strokeRect(barX + 0.5, barY + 0.5, barPx - 1, 4);
        ctx.font = `400 10px ${MONO}`;
        ctx.textBaseline = 'alphabetic';
        ctx.textAlign = 'left';
        ctx.fillText('0', barX, barY - 4);
        ctx.textAlign = 'right';
        ctx.fillText(barMm >= 1000 ? `${barMm / 1000} m` : `${barMm} mm`, barX + barPx, barY - 4);
    }

    // Redraw when the plan panel changes size, and once the fonts have arrived
    if (window.ResizeObserver) {
        new ResizeObserver(() => { if (activeTab === 'plan') render2DCanvas(); }).observe(viewport2d);
        new ResizeObserver(() => {
            if (activeTab === '3d' && viewer3D) viewer3D.onResize();
            if (activeTab === 'simulate' && roverSim) { roverSim.resize(); roverSim.render(); }
        }).observe($('stage'));
    } else {
        window.addEventListener('resize', () => render2DCanvas());
    }
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => render2DCanvas());

    // First plan
    runAnalysis();
});
