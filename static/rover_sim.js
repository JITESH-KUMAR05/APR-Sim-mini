/**
 * Sweep preview: animates the planned path over the wall in the browser, so the operator can see
 * the order of the passes. It is not a physics model; the C++ rover simulator (Physics replay) is.
 * Painted area and paint volume are the plan's own figures, shared out over the spray distance done.
 */

class RoverSimulation {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        this.ctx = this.canvas.getContext('2d');

        this.data = null;
        this.waypoints = [];
        this.obstacles = [];
        this.wall = { width_mm: 4000, height_mm: 2800, spray_width_mm: 250 };
        this.sourceImage = null;

        this.isRunning = false;
        this.isPaused = false;
        this.animId = null;
        this.speedMultiplier = 1;
        this.lastFrameTime = performance.now();
        this.loop = this.loop.bind(this);

        this.viewW = 0;
        this.viewH = 0;
        this.state = { index: 0, along: 0, x: 0, y: 0, spraying: false, sprayDone_mm: 0, seconds: 0, finished: false };
        this.sprayTotal_mm = 0;
        this.speed_mm_s = 250;
    }

    init(missionData) {
        if (!this.canvas || !missionData || !missionData.wall) return;
        this.data = missionData;
        this.wall = missionData.wall;
        this.obstacles = missionData.obstacles || [];
        this.waypoints = missionData.waypoints || [];
        const metrics = missionData.metrics || {};
        this.speed_mm_s = (metrics.nominal_speed_mps || 0.25) * 1000;
        this.sprayTotal_mm = 0;
        for (let i = 1; i < this.waypoints.length; i++) {
            if (this.isSpray(i)) this.sprayTotal_mm += this.segmentLength(i);
        }
        this.resize();
        this.reset();
    }

    isSpray(i) {
        const a = this.waypoints[i - 1];
        const b = this.waypoints[i];
        return a.spray_active && b.spray_active && a.row === b.row;
    }

    segmentLength(i) {
        const a = this.waypoints[i - 1];
        const b = this.waypoints[i];
        return Math.hypot(b.x - a.x, b.y - a.y);
    }

    resize() {
        if (!this.canvas) return;
        const rect = this.canvas.parentElement.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        this.viewW = rect.width;
        this.viewH = rect.height;
        this.canvas.width = Math.max(1, Math.round(rect.width * dpr));
        this.canvas.height = Math.max(1, Math.round(rect.height * dpr));
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    reset() {
        this.stop();
        const first = this.waypoints[0] || { x: 0, y: 0 };
        this.state = { index: 1, along: 0, x: first.x, y: first.y, spraying: false, sprayDone_mm: 0, seconds: 0, finished: false };
        this.updateReadouts();
        this.render();
    }

    start() {
        if (!this.canvas || this.waypoints.length < 2) return;
        if (this.state.finished) this.reset();
        if (this.isRunning && !this.isPaused) return;
        this.isRunning = true;
        this.isPaused = false;
        this.lastFrameTime = performance.now();
        this.animId = requestAnimationFrame(this.loop);
        this.updateReadouts();
    }

    pause() {
        if (!this.isRunning) return;
        this.isPaused = !this.isPaused;
        if (!this.isPaused) {
            this.lastFrameTime = performance.now();
            this.animId = requestAnimationFrame(this.loop);
        }
        this.updateReadouts();
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
    }

    setViewMode() {
        this.render();
    }

    loop(timestamp) {
        if (!this.isRunning || this.isPaused) return;
        // The preview runs at 20x real time so a wall finishes in a minute or two; the speed buttons multiply that
        const dt = Math.min(0.08, (timestamp - this.lastFrameTime) / 1000) * this.speedMultiplier * 20;
        this.lastFrameTime = timestamp;
        this.update(dt);
        this.render();
        if (this.isRunning) this.animId = requestAnimationFrame(this.loop);
    }

    update(dt) {
        const s = this.state;
        let travel = this.speed_mm_s * dt;
        s.seconds += dt;
        while (travel > 0 && s.index < this.waypoints.length) {
            const length = this.segmentLength(s.index);
            const left = length - s.along;
            const step = Math.min(travel, left);
            s.spraying = this.isSpray(s.index);
            if (s.spraying) s.sprayDone_mm += step;
            s.along += step;
            travel -= step;
            const a = this.waypoints[s.index - 1];
            const b = this.waypoints[s.index];
            const f = length > 0 ? s.along / length : 1;
            s.x = a.x + (b.x - a.x) * f;
            s.y = a.y + (b.y - a.y) * f;
            if (s.along >= length - 1e-6) {
                s.index += 1;
                s.along = 0;
            }
        }
        if (s.index >= this.waypoints.length) {
            s.finished = true;
            s.spraying = false;
            this.stop();
        }
        this.updateReadouts();
    }

    geometry() {
        const pad = 24;
        const W = this.wall.width_mm;
        const H = this.wall.height_mm;
        const scale = Math.min((this.viewW - pad * 2) / W, (this.viewH - pad * 2) / H);
        const x0 = (this.viewW - W * scale) / 2;
        const y0 = (this.viewH - H * scale) / 2;
        return { scale, x0, y0, X: (mm) => x0 + mm * scale, Y: (mm) => y0 + (H - mm) * scale };
    }

    render() {
        if (!this.canvas || !this.viewW || !this.viewH) return;
        const theme = window.APRTheme;
        const c = (name) => theme.color(name);
        const ctx = this.ctx;
        ctx.fillStyle = c('canvas');
        ctx.fillRect(0, 0, this.viewW, this.viewH);
        if (!this.data) return;

        const g = this.geometry();
        const W = this.wall.width_mm * g.scale;
        const H = this.wall.height_mm * g.scale;
        ctx.fillStyle = c('wall');
        ctx.fillRect(g.x0, g.y0, W, H);
        if (this.sourceImage && this.sourceImage.complete) {
            ctx.save();
            ctx.globalAlpha = theme.get() === 'dark' ? 0.25 : 0.32;
            ctx.drawImage(this.sourceImage, g.x0, g.y0, W, H);
            ctx.restore();
        }

        // Keep-outs
        this.obstacles.forEach((obs) => {
            const unsure = obs.type === 'unverified';
            ctx.save();
            ctx.strokeStyle = unsure ? c('review') : c('keep');
            ctx.lineWidth = 1;
            if (unsure) ctx.setLineDash([4, 3]);
            ctx.strokeRect(g.X(obs.x), g.Y(obs.y + obs.h), obs.w * g.scale, obs.h * g.scale);
            ctx.restore();
        });

        // The whole planned path, faint, then the part done so far as painted stripes
        const s = this.state;
        ctx.save();
        ctx.strokeStyle = c('muted');
        ctx.globalAlpha = 0.45;
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 4]);
        ctx.beginPath();
        this.waypoints.forEach((wp, i) => {
            if (i === 0) ctx.moveTo(g.X(wp.x), g.Y(wp.y)); else ctx.lineTo(g.X(wp.x), g.Y(wp.y));
        });
        ctx.stroke();
        ctx.restore();

        ctx.save();
        ctx.beginPath();
        ctx.rect(g.x0, g.y0, W, H);
        ctx.clip();
        ctx.strokeStyle = c('accent');
        ctx.globalAlpha = 0.32;
        ctx.lineCap = 'butt';
        ctx.lineWidth = Math.max(1, (this.wall.spray_width_mm || 250) * g.scale);
        const doneUpTo = Math.min(s.index, this.waypoints.length - 1);
        for (let i = 1; i <= doneUpTo; i++) {
            if (!this.isSpray(i)) continue;
            const a = this.waypoints[i - 1];
            const end = i < s.index ? this.waypoints[i] : { x: s.x, y: s.y };
            ctx.beginPath();
            ctx.moveTo(g.X(a.x), g.Y(a.y));
            ctx.lineTo(g.X(end.x), g.Y(end.y));
            ctx.stroke();
        }
        ctx.restore();

        ctx.strokeStyle = c('strong');
        ctx.lineWidth = 1;
        ctx.strokeRect(g.x0 + 0.5, g.y0 + 0.5, W - 1, H - 1);

        // The rover: a ring, filled while it sprays
        const rx = g.X(s.x);
        const ry = g.Y(s.y);
        ctx.beginPath();
        ctx.arc(rx, ry, 6, 0, Math.PI * 2);
        ctx.fillStyle = s.spraying ? c('accent') : c('canvas');
        ctx.fill();
        ctx.strokeStyle = c('text');
        ctx.lineWidth = 1.5;
        ctx.stroke();
    }

    updateReadouts() {
        const set = (id, text) => {
            const el = document.getElementById(id);
            if (el) el.textContent = text;
        };
        const s = this.state;
        const metrics = (this.data && this.data.metrics) || {};
        const share = this.sprayTotal_mm > 0 ? Math.min(1, s.sprayDone_mm / this.sprayTotal_mm) : 0;
        const total = Math.max(0, this.waypoints.length);
        const reached = s.finished ? total : Math.min(total, s.index);
        let status = 'Ready';
        if (s.finished) status = 'Finished';
        else if (this.isRunning && this.isPaused) status = 'Paused';
        else if (this.isRunning) status = s.spraying ? 'Painting' : 'Moving, spray off';

        set('simRoverStatus', status);
        set('simPosX', `x ${Math.round(s.x)} mm`);
        set('simPosY', `y ${Math.round(s.y)} mm`);
        set('simWpProgress', `${reached} / ${total}`);
        set('simProgressPct', `${Math.round(share * 100)}%`);
        set('simAreaCovered', `${((metrics.net_paintable_area_m2 || 0) * share).toFixed(2)} m²`);
        set('simPaintVolume', `${((metrics.paint_volume_liters || 0) * share).toFixed(2)} L`);
        const minutes = Math.floor(s.seconds / 60);
        const seconds = Math.floor(s.seconds % 60);
        set('simElapsedTime', `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`);
        const bar = document.getElementById('simProgressBar');
        if (bar) bar.style.width = `${share * 100}%`;
    }
}

window.RoverSimulation = RoverSimulation;
