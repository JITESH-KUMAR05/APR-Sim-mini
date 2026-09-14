/**
 * Automated Painting Rover (APR) - Autonomous Simulation Engine
 * Reference: MSME Grant INC25ETS084848 / WBS 1.2 & 1.3 Subsystems
 * High-precision 2D/3D kinematic simulation of wall rover with:
 * - Dynamic cell grid (G01...Gnn) with red keep-out obstacle avoidance
 * - Realistic vehicle chassis with vacuum adhesion seal & airless spray nozzle
 * - Solenoid gate simulation (SPRAY ON vs. RAPID BYPASS TRANSIT)
 * - Real-time robotics telemetry & mission terminal log
 */

class RoverSimulation {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        this.ctx = this.canvas.getContext('2d');

        // Mission data
        this.data = null;
        this.waypoints = [];
        this.obstacles = [];
        this.wall = { width_mm: 4000, height_mm: 2800, spray_width_mm: 250 };

        // Grid cells (G01...Gnn)
        this.cells = [];
        this.cols = 16;
        this.rows = 12;

        // Kinematics & Simulation State
        this.isRunning = false;
        this.isPaused = false;
        this.animId = null;
        this.speedMultiplier = 1.0;
        this.viewMode = 'hybrid'; // 'hybrid', 'cells', 'realistic'

        // Rover state
        this.rover = {
            x: 0,
            y: 0,
            heading: 0, // radians
            targetHeading: 0,
            sprayActive: false,
            currentWpIndex: 0,
            progressOnSegment: 0,
            segmentLength: 0,
            speed_mm_per_sec: 250, // 0.25 m/s nominal speed (WBS 1.2 §2)
            vacuumPressure_kpa: 48.5,
            adhesionForce_n: 320,
            paintDeposited_liters: 0.0,
            areaCovered_m2: 0.0,
            elapsedSeconds: 0
        };

        // Paint deposition canvas for realistic continuous painting
        this.paintCanvas = document.createElement('canvas');
        this.paintCtx = this.paintCanvas.getContext('2d');

        // Bind resize
        this.resize = this.resize.bind(this);
        window.addEventListener('resize', this.resize);

        // Particle system for airless spray effect
        this.particles = [];

        // Source wall background image overlay
        this.sourceImage = null;

        this.lastFrameTime = performance.now();
    }

    init(missionData) {
        if (!missionData || !missionData.wall) return;
        this.data = missionData;
        this.wall = missionData.wall;
        this.obstacles = missionData.obstacles || [];
        this.waypoints = missionData.waypoints || [];

        // Preload dimmed original wall image overlay
        if (missionData.image_data_url) {
            this.sourceImage = new Image();
            this.sourceImage.onload = () => {
                this.render();
            };
            this.sourceImage.src = missionData.image_data_url;
        } else if (window.loadedImage2D) {
            this.sourceImage = window.loadedImage2D;
        }

        // Calculate grid cells based on spray width
        const cellW_mm = Math.max(150, this.wall.spray_width_mm || 250);
        const cellH_mm = Math.max(150, this.wall.spray_width_mm || 250);
        this.cols = Math.max(8, Math.round(this.wall.width_mm / cellW_mm));
        this.rows = Math.max(6, Math.round(this.wall.height_mm / cellH_mm));

        this.generateGridCells();
        this.resize();
        this.reset();
    }

    generateGridCells() {
        this.cells = [];
        const cellW_mm = this.wall.width_mm / this.cols;
        const cellH_mm = this.wall.height_mm / this.rows;
        let count = 1;

        for (let r = 0; r < this.rows; r++) {
            for (let c = 0; c < this.cols; c++) {
                const x1 = c * cellW_mm;
                const y1 = r * cellH_mm;
                const x2 = (c + 1) * cellW_mm;
                const y2 = (r + 1) * cellH_mm;

                // Check intersection with any obstacle (Keep-Out)
                let isKeepOut = false;
                let obstacleRef = null;

                for (const obs of this.obstacles) {
                    const ox1 = obs.x;
                    const oy1 = obs.y;
                    const ox2 = obs.x + obs.w;
                    const oy2 = obs.y + obs.h;

                    // Overlap check in wall coordinates
                    if (x1 < ox2 && x2 > ox1 && y1 < oy2 && y2 > oy1) {
                        isKeepOut = true;
                        obstacleRef = obs;
                        break;
                    }
                }

                this.cells.push({
                    id: count,
                    label: `G${count < 10 ? '0' + count : count}`,
                    col: c,
                    row: r,
                    x_mm: x1,
                    y_mm: y1,
                    w_mm: cellW_mm,
                    h_mm: cellH_mm,
                    isKeepOut: isKeepOut,
                    obstacleRef: obstacleRef,
                    isPainted: false,
                    paintCoverage: 0.0 // 0 to 1.0
                });

                count++;
            }
        }
    }

    resize() {
        if (!this.canvas) return;
        const rect = this.canvas.parentElement.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const w = rect.width;
        const h = rect.height;

        this.canvas.width = w * dpr;
        this.canvas.height = h * dpr;
        this.canvas.style.width = w + 'px';
        this.canvas.style.height = h + 'px';
        this.ctx.scale(dpr, dpr);

        this.paintCanvas.width = w * dpr;
        this.paintCanvas.height = h * dpr;
        this.paintCtx.scale(dpr, dpr);

        this.viewW = w;
        this.viewH = h;
    }

    reset() {
        this.stop();
        if (this.waypoints.length === 0) return;

        const first = this.waypoints[0];
        this.rover.x = first.x;
        this.rover.y = first.y;
        this.rover.heading = 0;
        this.rover.targetHeading = 0;
        this.rover.sprayActive = false;
        this.rover.currentWpIndex = 0;
        this.rover.progressOnSegment = 0;
        this.rover.paintDeposited_liters = 0.0;
        this.rover.areaCovered_m2 = 0.0;
        this.rover.elapsedSeconds = 0;
        this.particles = [];

        // Clear paint canvas
        this.paintCtx.clearRect(0, 0, this.viewW, this.viewH);

        // Reset cell painted status
        this.cells.forEach(c => {
            c.isPainted = false;
            c.paintCoverage = 0.0;
        });

        this.updateTelemetryUI();
        this.render();
    }

    start() {
        if (this.isRunning) return;
        this.isRunning = true;
        this.isPaused = false;
        this.lastFrameTime = performance.now();
        this.logMessage(`Mission execution started from Origin (X: ${this.rover.x.toFixed(0)}, Y: ${this.rover.y.toFixed(0)})`, 'success');
        this.loop = this.loop.bind(this);
        this.animId = requestAnimationFrame(this.loop);
    }

    pause() {
        this.isPaused = !this.isPaused;
        if (!this.isPaused) {
            this.lastFrameTime = performance.now();
            this.animId = requestAnimationFrame(this.loop);
            this.logMessage(`Mission resumed at Waypoint #${this.rover.currentWpIndex + 1}`, 'info');
        } else {
            this.logMessage(`Mission paused by operator`, 'warn');
        }
    }

    stop() {
        this.isRunning = false;
        this.isPaused = false;
        if (this.animId) {
            cancelAnimationFrame(this.animId);
            this.animId = null;
        }
    }

    setSpeed(multiplier) {
        this.speedMultiplier = multiplier;
        this.logMessage(`Simulation speed set to ${multiplier}x`, 'info');
    }

    setViewMode(mode) {
        this.viewMode = mode;
        this.render();
    }

    loop(timestamp) {
        if (!this.isRunning || this.isPaused) return;

        const dtSec = Math.min(0.08, (timestamp - this.lastFrameTime) / 1000.0) * this.speedMultiplier;
        this.lastFrameTime = timestamp;

        this.update(dtSec);
        this.render();

        if (this.isRunning) {
            this.animId = requestAnimationFrame(this.loop);
        }
    }

    update(dt) {
        if (this.rover.currentWpIndex >= this.waypoints.length - 1) {
            this.isRunning = false;
            this.rover.sprayActive = false;
            this.updateTelemetryUI();
            this.logMessage(`MISSION COMPLETE: Full surface covered. Total paint: ${this.rover.paintDeposited_liters.toFixed(2)} L`, 'success');
            return;
        }

        this.rover.elapsedSeconds += dt;

        const p1 = this.waypoints[this.rover.currentWpIndex];
        const p2 = this.waypoints[this.rover.currentWpIndex + 1];

        const dx_mm = p2.x - p1.x;
        const dy_mm = p2.y - p1.y;
        const segDist_mm = Math.hypot(dx_mm, dy_mm);

        if (segDist_mm < 1e-3) {
            this.advanceWaypoint();
            return;
        }

        // Target angle in radians (canvas Y is inverted relative to wall Y)
        this.rover.targetHeading = Math.atan2(-dy_mm, dx_mm);

        // Smooth heading turn
        let angleDiff = this.rover.targetHeading - this.rover.heading;
        while (angleDiff < -Math.PI) angleDiff += Math.PI * 2;
        while (angleDiff > Math.PI) angleDiff -= Math.PI * 2;
        this.rover.heading += angleDiff * Math.min(1.0, dt * 10.0);

        // Speed: faster during non-spray transit (0.45 m/s) vs spraying (0.25 m/s)
        const isSpray = p1.spray_active && p2.spray_active && (p1.row === p2.row);
        const speed_mm = (isSpray ? 250.0 : 450.0);
        const stepDist_mm = speed_mm * dt;

        this.rover.progressOnSegment += stepDist_mm;

        if (this.rover.progressOnSegment >= segDist_mm) {
            // Reached waypoint
            this.rover.x = p2.x;
            this.rover.y = p2.y;
            this.advanceWaypoint();
        } else {
            const fraction = this.rover.progressOnSegment / segDist_mm;
            const prevX = this.rover.x;
            const prevY = this.rover.y;
            this.rover.x = p1.x + dx_mm * fraction;
            this.rover.y = p1.y + dy_mm * fraction;

            // Update spray status
            this.rover.sprayActive = isSpray;

            // Deposit paint if spraying
            if (isSpray) {
                const moved_m = Math.hypot(this.rover.x - prevX, this.rover.y - prevY) / 1000.0;
                const spray_w_m = (this.wall.spray_width_mm || 250.0) / 1000.0;
                const newArea = moved_m * spray_w_m;
                this.rover.areaCovered_m2 += newArea;

                // Paint volume = Area * WFT (50um DFT / 0.45 solids / 0.9 efficiency)
                const wft_mm = (50.0 / 1000.0) / 0.45 / 0.90;
                this.rover.paintDeposited_liters += newArea * wft_mm;

                // Mark grid cells
                this.markCellsUnderRover(this.rover.x, this.rover.y, this.wall.spray_width_mm || 250);

                // Deposit continuous paint trail on canvas
                this.depositPaintOnCanvas(prevX, prevY, this.rover.x, this.rover.y);

                // Emit spray particles
                this.emitParticles(this.rover.x, this.rover.y, this.rover.heading);
            }
        }

        // Update particle physics
        this.updateParticles(dt);

        // Update DOM telemetry
        this.updateTelemetryUI();
    }

    advanceWaypoint() {
        this.rover.currentWpIndex++;
        this.rover.progressOnSegment = 0;

        if (this.rover.currentWpIndex < this.waypoints.length) {
            const currWp = this.waypoints[this.rover.currentWpIndex];
            if (currWp.type === 'TRANSIT_START') {
                this.logMessage(`SOLENOID OFF: Keep-out obstacle bypass transit initiated`, 'warn');
            } else if (currWp.type === 'PASS_START') {
                this.logMessage(`SOLENOID ON: Paint spray engaged at Row #${currWp.row} (220 bar)`, 'info');
            } else if (currWp.type === 'ROW_TRANSITION') {
                this.logMessage(`ROW STEP: Stepping to Pass #${currWp.row + 1}`, 'info');
            }
        }
    }

    markCellsUnderRover(roverX_mm, roverY_mm, radius_mm) {
        const radSq = (radius_mm * 0.75) ** 2;
        this.cells.forEach(cell => {
            if (cell.isKeepOut) return;
            const cx = cell.x_mm + cell.w_mm / 2;
            const cy = cell.y_mm + cell.h_mm / 2;
            const distSq = (cx - roverX_mm) ** 2 + (cy - roverY_mm) ** 2;
            if (distSq <= radSq) {
                cell.isPainted = true;
                cell.paintCoverage = Math.min(1.0, cell.paintCoverage + 0.35);
            }
        });
    }

    depositPaintOnCanvas(x1_mm, y1_mm, x2_mm, y2_mm) {
        const p1 = this.wallToCanvas(x1_mm, y1_mm);
        const p2 = this.wallToCanvas(x2_mm, y2_mm);
        const sprayW_px = (this.wall.spray_width_mm / this.wall.width_mm) * this.drawW;

        this.paintCtx.save();
        this.paintCtx.strokeStyle = 'rgba(92, 225, 210, 0.55)'; // Emerald / Cyan wet paint
        this.paintCtx.lineWidth = sprayW_px;
        this.paintCtx.lineCap = 'round';
        this.paintCtx.lineJoin = 'round';
        this.paintCtx.beginPath();
        this.paintCtx.moveTo(p1.x, p1.y);
        this.paintCtx.lineTo(p2.x, p2.y);
        this.paintCtx.stroke();
        this.paintCtx.restore();
    }

    emitParticles(roverX_mm, roverY_mm, heading) {
        const p = this.wallToCanvas(roverX_mm, roverY_mm);
        const sprayW_px = (this.wall.spray_width_mm / this.wall.width_mm) * this.drawW;

        for (let i = 0; i < 4; i++) {
            const spread = (Math.random() - 0.5) * (sprayW_px * 0.8);
            const normal = heading + Math.PI / 2;
            this.particles.push({
                x: p.x + Math.cos(normal) * spread,
                y: p.y + Math.sin(normal) * spread,
                vx: (Math.random() - 0.5) * 15,
                vy: (Math.random() - 0.5) * 15,
                alpha: 0.85,
                life: 0.35 + Math.random() * 0.25,
                radius: 1.5 + Math.random() * 2.0
            });
        }
    }

    updateParticles(dt) {
        for (let i = this.particles.length - 1; i >= 0; i--) {
            const part = this.particles[i];
            part.x += part.vx * dt;
            part.y += part.vy * dt;
            part.life -= dt;
            part.alpha = Math.max(0, part.life / 0.5);

            if (part.life <= 0) {
                this.particles.splice(i, 1);
            }
        }
    }

    wallToCanvas(x_mm, y_mm) {
        const padX = 70;
        const padY = 55;
        this.drawW = this.viewW - (padX * 2);
        this.drawH = this.viewH - (padY * 2);

        const x = padX + (x_mm / this.wall.width_mm) * this.drawW;
        const y = padY + this.drawH - (y_mm / this.wall.height_mm) * this.drawH; // Origin at bottom-left
        return { x, y };
    }

    render() {
        if (!this.ctx) return;
        const W = this.viewW;
        const H = this.viewH;

        this.ctx.clearRect(0, 0, W, H);

        // 1. Industrial Slate Background
        this.ctx.fillStyle = '#061015';
        this.ctx.fillRect(0, 0, W, H);

        const padX = 70;
        const padY = 55;
        const drawW = W - (padX * 2);
        const drawH = H - (padY * 2);

        // 2. Wall Face Backing
        this.ctx.fillStyle = '#0c1d24';
        this.ctx.fillRect(padX, padY, drawW, drawH);

        // Draw background source photo if in hybrid/realistic mode (dimmed but visible overlay)
        const bgImg = this.sourceImage || window.loadedImage2D;
        if (this.viewMode !== 'cells' && bgImg) {
            this.ctx.save();
            this.ctx.globalAlpha = (this.viewMode === 'realistic') ? 0.45 : 0.32;
            try {
                if (bgImg.complete && bgImg.naturalWidth > 0) {
                    this.ctx.drawImage(bgImg, padX, padY, drawW, drawH);
                }
            } catch (e) {}
            this.ctx.restore();
        }

        // 3. Draw Continuous Painted Trail
        if (this.viewMode !== 'cells') {
            this.ctx.drawImage(this.paintCanvas, 0, 0, W, H);
        }

        // 4. Render Grid Cells (G01...Gnn) if in cells or hybrid mode
        if (this.viewMode === 'cells' || this.viewMode === 'hybrid') {
            this.renderGridCells(padX, padY, drawW, drawH);
        }

        // 5. Render Keep-Out Obstacles (Windows, Doors, Electrical Panels)
        this.renderObstacles(padX, padY, drawW, drawH);

        // 6. Render Serpentine Toolpath Line
        if (this.viewMode !== 'cells') {
            this.renderToolpath(padX, padY, drawW, drawH);
        }

        // 7. Render Particle Effects
        this.renderParticles();

        // 8. Render Realistic APR Rover Vehicle
        this.renderRover(padX, padY, drawW, drawH);

        // 9. Coordinate Axis Calibration Rulers
        this.renderRulers(padX, padY, drawW, drawH);
    }

    renderGridCells(padX, padY, drawW, drawH) {
        const scaleX = drawW / this.wall.width_mm;
        const scaleY = drawH / this.wall.height_mm;
        const cellW_px = (this.wall.width_mm / this.cols) * scaleX;
        const cellH_px = (this.wall.height_mm / this.rows) * scaleY;

        this.ctx.lineWidth = 1;

        this.cells.forEach(cell => {
            const cx = padX + (cell.x_mm * scaleX);
            const cy = padY + drawH - ((cell.y_mm + cell.h_mm) * scaleY);

            // Determine Fill Color
            if (cell.isKeepOut) {
                // RED Keep-Out Cell
                this.ctx.fillStyle = this.viewMode === 'cells' ? '#5a1e1e' : 'rgba(250, 116, 110, 0.35)';
                this.ctx.strokeStyle = '#fa746e';
            } else if (cell.isPainted) {
                // GREEN Painted Cell
                this.ctx.fillStyle = this.viewMode === 'cells' ? '#164d42' : 'rgba(92, 225, 210, 0.40)';
                this.ctx.strokeStyle = '#5ce1d2';
            } else {
                // Dark Pending Cell
                this.ctx.fillStyle = this.viewMode === 'cells' ? '#0a1920' : 'rgba(16, 37, 45, 0.25)';
                this.ctx.strokeStyle = '#1d3c45';
            }

            // Draw rounded cell card (matching user screenshot)
            this.drawRoundedRect(this.ctx, cx + 2, cy + 2, cellW_px - 4, cellH_px - 4, 4, true, true);

            // Draw Cell Label (G01, G02, ...)
            if (cellW_px > 26) {
                this.ctx.font = '9px "DM Mono", monospace';
                this.ctx.fillStyle = cell.isKeepOut ? '#fa746e' : (cell.isPainted ? '#5ce1d2' : '#57757b');
                this.ctx.textAlign = 'center';
                this.ctx.textBaseline = 'middle';
                this.ctx.fillText(cell.label, cx + (cellW_px / 2), cy + (cellH_px / 2));
            }
        });
    }

    renderObstacles(padX, padY, drawW, drawH) {
        const scaleX = drawW / this.wall.width_mm;
        const scaleY = drawH / this.wall.height_mm;

        this.obstacles.forEach(obs => {
            const ox = padX + (obs.x * scaleX);
            const oy = padY + drawH - ((obs.y + obs.h) * scaleY);
            const ow = obs.w * scaleX;
            const oh = obs.h * scaleY;

            // Obstacle Solid Red Frame
            this.ctx.fillStyle = 'rgba(250, 116, 110, 0.28)';
            this.ctx.fillRect(ox, oy, ow, oh);
            this.ctx.strokeStyle = '#fa746e';
            this.ctx.lineWidth = 2.5;
            this.ctx.strokeRect(ox, oy, ow, oh);

            // Standoff buffer outline (Yellow dashed)
            const buf_px = (this.wall.safety_buffer_mm || 60) * scaleX;
            this.ctx.save();
            this.ctx.setLineDash([4, 4]);
            this.ctx.strokeStyle = 'rgba(244, 184, 96, 0.7)';
            this.ctx.lineWidth = 1.5;
            this.ctx.strokeRect(ox - buf_px, oy - buf_px, ow + (buf_px * 2), oh + (buf_px * 2));
            this.ctx.restore();

            // Label tag badge
            this.ctx.fillStyle = '#fa746e';
            this.ctx.font = 'bold 10px "DM Mono", monospace';
            this.ctx.textAlign = 'left';
            this.ctx.fillText(`⛔ KEEP-OUT: ${obs.label}`, ox + 6, oy + 16);
            this.ctx.fillStyle = '#a0babc';
            this.ctx.font = '9px "DM Mono", monospace';
            this.ctx.fillText(`${obs.w} × ${obs.h} mm (Buffer +60mm)`, ox + 6, oy + 28);
        });
    }

    renderToolpath(padX, padY, drawW, drawH) {
        if (!this.waypoints || this.waypoints.length < 2) return;

        const scaleX = drawW / this.wall.width_mm;
        const scaleY = drawH / this.wall.height_mm;

        this.ctx.lineWidth = 1.5;

        for (let i = 1; i < this.waypoints.length; i++) {
            const wp1 = this.waypoints[i - 1];
            const wp2 = this.waypoints[i];

            const x1 = padX + (wp1.x * scaleX);
            const y1 = padY + drawH - (wp1.y * scaleY);
            const x2 = padX + (wp2.x * scaleX);
            const y2 = padY + drawH - (wp2.y * scaleY);

            const isSpray = wp1.spray_active && wp2.spray_active && (wp1.row === wp2.row);

            this.ctx.beginPath();
            this.ctx.moveTo(x1, y1);
            this.ctx.lineTo(x2, y2);

            if (isSpray) {
                this.ctx.strokeStyle = 'rgba(244, 184, 96, 0.45)'; // Amber
                this.ctx.setLineDash([]);
            } else {
                this.ctx.strokeStyle = 'rgba(157, 78, 221, 0.4)'; // Magenta
                this.ctx.setLineDash([3, 3]);
            }
            this.ctx.stroke();
        }
        this.ctx.setLineDash([]);
    }

    renderParticles() {
        this.particles.forEach(p => {
            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
            this.ctx.fillStyle = `rgba(92, 225, 210, ${p.alpha})`;
            this.ctx.fill();
        });
    }

    renderRover(padX, padY, drawW, drawH) {
        const pos = this.wallToCanvas(this.rover.x, this.rover.y);
        const roverLen_px = 38;
        const roverWid_px = 28;

        this.ctx.save();
        this.ctx.translate(pos.x, pos.y);
        this.ctx.rotate(this.rover.heading);

        // 1. Vacuum Adhesion Suction Halo (Blue/Cyan glow beneath rover)
        this.ctx.beginPath();
        this.ctx.ellipse(0, 0, roverLen_px * 0.7, roverWid_px * 0.7, 0, 0, Math.PI * 2);
        this.ctx.fillStyle = 'rgba(92, 225, 210, 0.18)';
        this.ctx.fill();
        this.ctx.strokeStyle = 'rgba(92, 225, 210, 0.55)';
        this.ctx.lineWidth = 1.5;
        this.ctx.stroke();

        // 2. Active Spray Nozzle Fan (when sprayActive is true)
        if (this.rover.sprayActive) {
            this.ctx.save();
            this.ctx.beginPath();
            this.ctx.moveTo(roverLen_px * 0.5, 0);
            this.ctx.lineTo(roverLen_px * 0.5 + 28, -14);
            this.ctx.lineTo(roverLen_px * 0.5 + 28, 14);
            this.ctx.closePath();

            const sprayGrad = this.ctx.createLinearGradient(roverLen_px * 0.5, 0, roverLen_px * 0.5 + 28, 0);
            sprayGrad.addColorStop(0, 'rgba(244, 184, 96, 0.9)');
            sprayGrad.addColorStop(1, 'rgba(92, 225, 210, 0.15)');
            this.ctx.fillStyle = sprayGrad;
            this.ctx.fill();
            this.ctx.restore();
        }

        // 3. 4 Drive Tracks / Wheels
        this.ctx.fillStyle = '#060d10';
        this.ctx.strokeStyle = '#274b52';
        this.ctx.lineWidth = 1;
        // Front-left
        this.ctx.fillRect(roverLen_px * 0.2, -roverWid_px * 0.6, 12, 6);
        this.ctx.strokeRect(roverLen_px * 0.2, -roverWid_px * 0.6, 12, 6);
        // Rear-left
        this.ctx.fillRect(-roverLen_px * 0.45, -roverWid_px * 0.6, 12, 6);
        this.ctx.strokeRect(-roverLen_px * 0.45, -roverWid_px * 0.6, 12, 6);
        // Front-right
        this.ctx.fillRect(roverLen_px * 0.2, roverWid_px * 0.6 - 6, 12, 6);
        this.ctx.strokeRect(roverLen_px * 0.2, roverWid_px * 0.6 - 6, 12, 6);
        // Rear-right
        this.ctx.fillRect(-roverLen_px * 0.45, roverWid_px * 0.6 - 6, 12, 6);
        this.ctx.strokeRect(-roverLen_px * 0.45, roverWid_px * 0.6 - 6, 12, 6);

        // 4. Main Rover Chassis Body (Industrial robotic look)
        this.ctx.fillStyle = '#0f262e';
        this.ctx.strokeStyle = this.rover.sprayActive ? '#f4b860' : '#5ce1d2';
        this.ctx.lineWidth = 2;
        this.drawRoundedRect(this.ctx, -roverLen_px * 0.45, -roverWid_px * 0.45, roverLen_px * 0.9, roverWid_px * 0.9, 5, true, true);

        // Central electronics module & APR badge
        this.ctx.fillStyle = '#173641';
        this.ctx.fillRect(-roverLen_px * 0.25, -roverWid_px * 0.25, roverLen_px * 0.5, roverWid_px * 0.5);

        // Front spray nozzle emitter block
        this.ctx.fillStyle = this.rover.sprayActive ? '#f4b860' : '#fa746e';
        this.ctx.fillRect(roverLen_px * 0.45 - 2, -4, 6, 8);

        // Rover Label
        this.ctx.fillStyle = '#ffffff';
        this.ctx.font = 'bold 8px "DM Mono", monospace';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText('APR-01', 0, 0);

        this.ctx.restore();

        // 5. Floating Tooltip Tag above Rover
        this.ctx.save();
        this.ctx.font = '10px "DM Mono", monospace';
        this.ctx.textAlign = 'center';

        const tagText = this.rover.sprayActive ? 'ROVER · SPRAYING' : 'ROVER · BYPASS';
        const tagColor = this.rover.sprayActive ? '#f4b860' : '#fa746e';

        this.ctx.fillStyle = '#07171cdf';
        this.ctx.strokeStyle = tagColor;
        this.ctx.lineWidth = 1;
        this.ctx.strokeRect(pos.x - 55, pos.y - 38, 110, 18);
        this.ctx.fillRect(pos.x - 55, pos.y - 38, 110, 18);

        this.ctx.fillStyle = tagColor;
        this.ctx.fillText(tagText, pos.x, pos.y - 25);
        this.ctx.restore();
    }

    renderRulers(padX, padY, drawW, drawH) {
        this.ctx.font = '10px "DM Mono", monospace';
        this.ctx.fillStyle = '#65868c';
        this.ctx.textAlign = 'left';

        // Outer wall boundary box
        this.ctx.strokeStyle = '#27525c';
        this.ctx.lineWidth = 1.5;
        this.ctx.strokeRect(padX, padY, drawW, drawH);

        // Coordinate labels
        this.ctx.fillText('Origin (X:0, Y:0)', padX, padY + drawH + 18);
        this.ctx.textAlign = 'right';
        this.ctx.fillText(`X:${this.wall.width_mm} mm, Y:0`, padX + drawW, padY + drawH + 18);
        this.ctx.textAlign = 'left';
        this.ctx.fillText(`Y:${this.wall.height_mm} mm`, padX, padY - 10);
        this.ctx.textAlign = 'right';
        this.ctx.fillText(`Wall: ${this.wall.width_mm} × ${this.wall.height_mm} mm`, padX + drawW, padY - 10);
    }

    drawRoundedRect(ctx, x, y, width, height, radius, fill, stroke) {
        ctx.beginPath();
        ctx.moveTo(x + radius, y);
        ctx.lineTo(x + width - radius, y);
        ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
        ctx.lineTo(x + width, y + height - radius);
        ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
        ctx.lineTo(x + radius, y + height);
        ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
        ctx.lineTo(x, y + radius);
        ctx.quadraticCurveTo(x, y, x + radius, y);
        ctx.closePath();
        if (fill) ctx.fill();
        if (stroke) ctx.stroke();
    }

    updateTelemetryUI() {
        const setVal = (id, val) => {
            const el = document.getElementById(id);
            if (el) el.textContent = val;
        };

        const totalWp = this.waypoints.length;
        const currWp = Math.min(totalWp, this.rover.currentWpIndex + 1);
        const progressPct = totalWp > 0 ? Math.min(100, Math.round((currWp / totalWp) * 100)) : 0;

        setVal('simRoverStatus', this.rover.sprayActive ? 'SPRAYING (ACTIVE COAT)' : (this.isRunning ? 'BYPASS (SPRAY OFF)' : 'IDLE'));
        setVal('simPosX', `${Math.round(this.rover.x)} mm`);
        setVal('simPosY', `${Math.round(this.rover.y)} mm`);
        setVal('simPosZ', `250.0 mm`);
        setVal('simWpProgress', `${currWp} / ${totalWp}`);
        setVal('simProgressPct', `${progressPct}%`);

        const pBar = document.getElementById('simProgressBar');
        if (pBar) pBar.style.width = `${progressPct}%`;

        setVal('simSolenoidState', this.rover.sprayActive ? 'OPEN · 220 BAR' : 'CLOSED · BYPASS');
        const solEl = document.getElementById('simSolenoidState');
        if (solEl) {
            solEl.className = this.rover.sprayActive ? 'telemetry-badge active-spray' : 'telemetry-badge bypass-transit';
        }

        setVal('simVacuumForce', `${this.rover.adhesionForce_n} N (2.1x Safety)`);
        setVal('simAreaCovered', `${this.rover.areaCovered_m2.toFixed(2)} m²`);
        setVal('simPaintVolume', `${this.rover.paintDeposited_liters.toFixed(2)} L`);

        const mins = Math.floor(this.rover.elapsedSeconds / 60);
        const secs = Math.floor(this.rover.elapsedSeconds % 60);
        setVal('simElapsedTime', `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`);
    }

    logMessage(text, level = 'info') {
        const term = document.getElementById('simTerminal');
        if (!term) return;

        const time = new Date().toTimeString().split(' ')[0];
        const div = document.createElement('div');
        div.className = `sim-log-line ${level}`;
        div.innerHTML = `<b>${time}</b> &nbsp;${text}`;

        term.appendChild(div);
        term.scrollTop = term.scrollHeight;
    }
}


// Attach to window
window.RoverSimulation = RoverSimulation;