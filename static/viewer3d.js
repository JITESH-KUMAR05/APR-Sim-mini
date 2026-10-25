/**
 * 3D view of the planned wall (Three.js): the wall with its photo, the keep-outs as recesses or
 * fixtures with their names, and the painting path just off the wall face, each stripe as wide as
 * the spray. Colours come from the theme tokens; applyTheme() redraws.
 */

class WallViewer3D {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.grid = null;
        this.wallGroup = null;
        this.pathGroup = null;
        this.obstacleGroup = null;
        this.labelGroup = null;

        this.currentData = null;
        this.photo = null;
        this.photoTexture = null;
        this.wireframeMode = false;
        this.showPath = true;
        this.showObstacles = true;
        this.showPhoto = true;
        this.showLabels = true;

        this.init();
    }

    colour(name) {
        return new THREE.Color(window.APRTheme.color(name));
    }

    init() {
        if (!this.container) return;
        const width = this.container.clientWidth || 800;
        const height = this.container.clientHeight || 500;

        this.scene = new THREE.Scene();
        this.scene.background = this.colour('canvas');

        this.camera = new THREE.PerspectiveCamera(40, width / height, 0.05, 100);
        this.camera.position.set(0, 0, 7.5);

        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.container.appendChild(this.renderer.domElement);

        if (THREE.OrbitControls) {
            this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
            this.controls.enableDamping = true;
            this.controls.dampingFactor = 0.08;
            this.controls.screenSpacePanning = true;
            this.controls.minDistance = 0.8;
            this.controls.maxDistance = 30.0;
        }

        this.scene.add(new THREE.HemisphereLight(0xffffff, 0x808080, 0.7));
        const keyLight = new THREE.DirectionalLight(0xffffff, 0.55);
        keyLight.position.set(4, 6, 8);
        this.scene.add(keyLight);

        this.wallGroup = new THREE.Group();
        this.pathGroup = new THREE.Group();
        this.obstacleGroup = new THREE.Group();
        this.labelGroup = new THREE.Group();
        this.scene.add(this.wallGroup, this.pathGroup, this.obstacleGroup, this.labelGroup);
        this.makeGrid(-1.4, 16);

        window.addEventListener('resize', () => this.onResize());
        this.animate = this.animate.bind(this);
        requestAnimationFrame(this.animate);
    }

    makeGrid(y, size) {
        if (this.grid) {
            this.scene.remove(this.grid);
            this.grid.geometry.dispose();
            this.grid.material.dispose();
        }
        const line = this.colour('line');
        this.grid = new THREE.GridHelper(size, Math.round(size * 2), line, line);
        this.grid.material.transparent = true;
        this.grid.material.opacity = 0.6;
        this.grid.position.y = y;
        this.scene.add(this.grid);
    }

    onResize() {
        if (!this.container || !this.renderer || !this.camera) return;
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        if (!width || !height) return;
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    // Theme switched: new background and grid, and the wall rebuilt in the new colours, camera kept
    applyTheme() {
        if (!this.scene) return;
        this.scene.background = this.colour('canvas');
        if (this.currentData) this.loadWallData(this.currentData, { keepCamera: true });
        else this.makeGrid(-1.4, 16);
    }

    // The wall photo, painted on the wall face (null for none)
    setPhoto(image) {
        if (this.photoTexture) this.photoTexture.dispose();
        this.photo = image;
        this.photoTexture = null;
        if (image && window.THREE) {
            this.photoTexture = new THREE.Texture(image);
            this.photoTexture.needsUpdate = true;
        }
        if (this.currentData) this.loadWallData(this.currentData, { keepCamera: true });
    }

    clearGroup(group) {
        while (group.children.length > 0) {
            const obj = group.children.pop();
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) {
                if (obj.material.map && obj.material.map !== this.photoTexture) obj.material.map.dispose();
                obj.material.dispose();
            }
        }
    }

    edges(width, height, colour, x, y, z, dashed) {
        const geometry = new THREE.EdgesGeometry(new THREE.PlaneGeometry(width, height));
        const material = dashed
            ? new THREE.LineDashedMaterial({ color: colour, dashSize: 0.05, gapSize: 0.03 })
            : new THREE.LineBasicMaterial({ color: colour });
        const lines = new THREE.LineSegments(geometry, material);
        if (dashed) lines.computeLineDistances();
        lines.position.set(x, y, z);
        return lines;
    }

    // A flat name tag that always faces the camera: the keep-out's name and its size in mm
    label(name, size, tone, heightM) {
        const css = (n) => window.APRTheme.color(n);
        const scale = 2;
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const nameFont = `500 ${13 * scale}px "IBM Plex Sans", system-ui, sans-serif`;
        const sizeFont = `400 ${12 * scale}px "IBM Plex Mono", ui-monospace, monospace`;
        ctx.font = nameFont;
        const nameW = ctx.measureText(name).width;
        ctx.font = sizeFont;
        const sizeW = ctx.measureText(size).width;
        const pad = 8 * scale;
        canvas.width = Math.ceil(nameW + sizeW + pad * 3);
        canvas.height = 26 * scale;
        ctx.fillStyle = css('panel');
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = tone;
        ctx.lineWidth = scale;
        ctx.strokeRect(scale / 2, scale / 2, canvas.width - scale, canvas.height - scale);
        ctx.textBaseline = 'middle';
        ctx.font = nameFont;
        ctx.fillStyle = tone;
        ctx.fillText(name, pad, canvas.height / 2 + scale);
        ctx.font = sizeFont;
        ctx.fillStyle = css('muted');
        ctx.fillText(size, pad * 2 + nameW, canvas.height / 2 + scale);

        const texture = new THREE.CanvasTexture(canvas);
        const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, depthTest: false }));
        sprite.scale.set(heightM * canvas.width / canvas.height, heightM, 1);
        sprite.renderOrder = 10;
        return sprite;
    }

    loadWallData(data, options = {}) {
        if (!this.scene) return;
        this.currentData = data;
        [this.wallGroup, this.obstacleGroup, this.pathGroup, this.labelGroup].forEach((group) => this.clearGroup(group));

        const tone = {
            wall: this.colour('wall'), panel: this.colour('panel'), canvas: this.colour('canvas'),
            strong: this.colour('strong'), muted: this.colour('muted'), accent: this.colour('accent'),
            keep: this.colour('keep'), review: this.colour('review')
        };
        const wallW = data.wall.width_mm * 0.001;
        const wallH = data.wall.height_mm * 0.001;
        const wallD = 0.2;
        const offsetX = -wallW / 2;
        const offsetY = -wallH / 2;
        const size = Math.max(wallW, wallH);
        this.makeGrid(offsetY, Math.ceil(size * 3));
        const X = (mm) => mm * 0.001 + offsetX;
        const Y = (mm) => mm * 0.001 + offsetY;

        // The wall: a body, the front face with the photo on it, and its outline
        const body = new THREE.Mesh(
            new THREE.BoxGeometry(wallW, wallH, wallD),
            new THREE.MeshLambertMaterial({ color: tone.panel, wireframe: this.wireframeMode })
        );
        body.position.set(0, 0, -wallD / 2);
        this.wallGroup.add(body);

        const withPhoto = this.showPhoto && this.photoTexture;
        const face = new THREE.Mesh(
            new THREE.PlaneGeometry(wallW, wallH),
            new THREE.MeshBasicMaterial(withPhoto
                ? { map: this.photoTexture, color: 0xffffff, wireframe: this.wireframeMode }
                : { color: tone.wall, wireframe: this.wireframeMode })
        );
        face.position.set(0, 0, 0.002);
        this.wallGroup.add(face);
        this.wallGroup.add(this.edges(wallW, wallH, tone.strong, 0, 0, 0.004));

        // Keep-outs: windows and doors are recesses, fixtures stand out from the wall
        const labelH = Math.max(0.07, size * 0.022);
        data.obstacles.forEach((obs) => {
            const ow = obs.w * 0.001;
            const oh = obs.h * 0.001;
            const ox = X(obs.x) + ow / 2;
            const oy = Y(obs.y) + oh / 2;
            const unsure = obs.type === 'unverified';
            const edge = unsure ? tone.review : tone.keep;
            const recess = obs.type === 'window' || obs.type === 'door';
            const depth = recess ? Math.min(0.12, (obs.depth_mm || 80) * 0.001) : 0.05;

            const block = new THREE.Mesh(
                new THREE.BoxGeometry(ow, oh, depth),
                new THREE.MeshLambertMaterial({ color: edge, transparent: true, opacity: recess ? 0.18 : (unsure ? 0.22 : 0.4) })
            );
            block.position.set(ox, oy, recess ? -depth / 2 + 0.003 : depth / 2);
            this.obstacleGroup.add(block);
            this.obstacleGroup.add(this.edges(ow, oh, edge, ox, oy, recess ? 0.008 : depth + 0.002, unsure));

            const tag = this.label(obs.label, `${Math.round(obs.w)} × ${Math.round(obs.h)}`,
                window.APRTheme.color(unsure ? 'review' : 'keep'), labelH);
            tag.position.set(ox, Y(obs.y + obs.h) + labelH * 0.75, 0.1);
            this.labelGroup.add(tag);
        });

        // The path: each spray pass as a band the width of the nozzle, moves with the spray off dashed
        const waypoints = data.waypoints || [];
        const sprayW = (data.wall.spray_width_mm || 250) * 0.001;
        const bandMaterial = new THREE.MeshBasicMaterial({ color: tone.accent, transparent: true, opacity: 0.22, side: THREE.DoubleSide, depthWrite: false });
        const centre = [];
        const moves = [];
        for (let i = 1; i < waypoints.length; i++) {
            const a = waypoints[i - 1];
            const b = waypoints[i];
            const pair = [new THREE.Vector3(X(a.x), Y(a.y), 0.03), new THREE.Vector3(X(b.x), Y(b.y), 0.03)];
            const spray = a.spray_active && b.spray_active && a.row === b.row;
            (spray ? centre : moves).push(...pair);
            if (spray) {
                const horizontal = Math.abs(b.x - a.x) >= Math.abs(b.y - a.y);
                const length = Math.max(Math.abs(b.x - a.x), Math.abs(b.y - a.y)) * 0.001;
                const band = new THREE.Mesh(
                    new THREE.PlaneGeometry(horizontal ? length : sprayW, horizontal ? sprayW : length),
                    bandMaterial
                );
                band.position.set((X(a.x) + X(b.x)) / 2, (Y(a.y) + Y(b.y)) / 2, 0.012);
                this.pathGroup.add(band);
            }
        }
        if (centre.length) {
            this.pathGroup.add(new THREE.LineSegments(
                new THREE.BufferGeometry().setFromPoints(centre),
                new THREE.LineBasicMaterial({ color: tone.accent })
            ));
        }
        if (moves.length) {
            const transit = new THREE.LineSegments(
                new THREE.BufferGeometry().setFromPoints(moves),
                new THREE.LineDashedMaterial({ color: tone.muted, dashSize: 0.04, gapSize: 0.04 })
            );
            transit.computeLineDistances();
            this.pathGroup.add(transit);
        }
        if (waypoints.length) {
            const start = new THREE.Mesh(new THREE.SphereGeometry(Math.max(0.025, size * 0.008), 16, 12), new THREE.MeshBasicMaterial({ color: tone.accent }));
            start.position.set(X(waypoints[0].x), Y(waypoints[0].y), 0.04);
            this.pathGroup.add(start);
        }

        this.pathGroup.visible = this.showPath;
        this.obstacleGroup.visible = this.showObstacles;
        this.labelGroup.visible = this.showObstacles && this.showLabels;
        if (!options.keepCamera) this.setIsometricView();
    }

    animate() {
        requestAnimationFrame(this.animate);
        if (this.controls) this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    lookFrom(x, y, z) {
        this.camera.position.set(x, y, z);
        if (this.controls) {
            this.controls.target.set(0, 0, 0);
            this.controls.update();
        }
    }

    wallSize() {
        return Math.max(this.currentData.wall.width_mm, this.currentData.wall.height_mm) * 0.001;
    }

    resetView() {
        if (!this.currentData) return;
        this.lookFrom(0, 0, this.wallSize() * 1.6);
    }

    setIsometricView() {
        if (!this.currentData) return;
        const size = this.wallSize();
        this.lookFrom(size * 0.55, size * 0.22, size * 1.45);
    }

    setTopDownView() {
        if (!this.currentData) return;
        const size = Math.max(this.wallSize(), 3.0);
        this.lookFrom(0, size * 1.6, 0.08);
    }

    toggleWireframe() {
        this.wireframeMode = !this.wireframeMode;
        this.wallGroup.traverse((child) => {
            if (child.material && 'wireframe' in child.material) child.material.wireframe = this.wireframeMode;
        });
        return this.wireframeMode;
    }

    togglePath() {
        this.showPath = !this.showPath;
        this.pathGroup.visible = this.showPath;
        return this.showPath;
    }

    toggleObstacles() {
        this.showObstacles = !this.showObstacles;
        this.obstacleGroup.visible = this.showObstacles;
        this.labelGroup.visible = this.showObstacles && this.showLabels;
        return this.showObstacles;
    }

    togglePhoto() {
        this.showPhoto = !this.showPhoto;
        if (this.currentData) this.loadWallData(this.currentData, { keepCamera: true });
        return this.showPhoto;
    }

    toggleLabels() {
        this.showLabels = !this.showLabels;
        this.labelGroup.visible = this.showObstacles && this.showLabels;
        return this.showLabels;
    }
}

window.WallViewer3D = WallViewer3D;
