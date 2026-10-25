/**
 * 3D view of the planned wall (Three.js): the wall, its keep-outs as recesses or fixtures, and the
 * painting path just off the wall face. Colours come from the theme tokens; applyTheme() redraws.
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

        this.currentData = null;
        this.wireframeMode = false;
        this.showPath = true;
        this.showObstacles = true;

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

        this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
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
            this.controls.minDistance = 1.5;
            this.controls.maxDistance = 25.0;
        }

        this.scene.add(new THREE.AmbientLight(0xffffff, 0.75));
        const keyLight = new THREE.DirectionalLight(0xffffff, 0.6);
        keyLight.position.set(5, 8, 10);
        this.scene.add(keyLight);

        this.wallGroup = new THREE.Group();
        this.pathGroup = new THREE.Group();
        this.obstacleGroup = new THREE.Group();
        this.scene.add(this.wallGroup, this.pathGroup, this.obstacleGroup);
        this.makeGrid(-1.4);

        window.addEventListener('resize', () => this.onResize());
        this.animate = this.animate.bind(this);
        requestAnimationFrame(this.animate);
    }

    makeGrid(y) {
        if (this.grid) {
            this.scene.remove(this.grid);
            this.grid.geometry.dispose();
            this.grid.material.dispose();
        }
        const line = this.colour('line');
        this.grid = new THREE.GridHelper(16, 32, line, line);
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
        this.makeGrid(this.grid ? this.grid.position.y : -1.4);
        if (this.currentData) this.loadWallData(this.currentData, { keepCamera: true });
    }

    clearGroup(group) {
        while (group.children.length > 0) {
            const obj = group.children.pop();
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) obj.material.dispose();
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

    loadWallData(data, options = {}) {
        if (!this.scene) return;
        this.currentData = data;
        [this.wallGroup, this.obstacleGroup, this.pathGroup].forEach((group) => this.clearGroup(group));

        const tone = {
            wall: this.colour('wall'), panel: this.colour('panel'), canvas: this.colour('canvas'),
            strong: this.colour('strong'), muted: this.colour('muted'), accent: this.colour('accent'),
            keep: this.colour('keep'), review: this.colour('review')
        };
        const wallW = data.wall.width_mm * 0.001;
        const wallH = data.wall.height_mm * 0.001;
        const wallD = 0.25;
        const offsetX = -wallW / 2;
        const offsetY = -wallH / 2;
        if (this.grid) this.grid.position.y = offsetY;

        // The wall: a body, a flat front face and its outline
        const body = new THREE.Mesh(
            new THREE.BoxGeometry(wallW, wallH, wallD),
            new THREE.MeshLambertMaterial({ color: tone.panel, wireframe: this.wireframeMode })
        );
        body.position.set(0, 0, -wallD / 2);
        this.wallGroup.add(body);

        const face = new THREE.Mesh(
            new THREE.PlaneGeometry(wallW, wallH),
            new THREE.MeshBasicMaterial({ color: tone.wall, side: THREE.DoubleSide, wireframe: this.wireframeMode })
        );
        face.position.set(0, 0, 0.002);
        this.wallGroup.add(face);
        this.wallGroup.add(this.edges(wallW, wallH, tone.strong, 0, 0, 0.004));

        // Keep-outs: windows and doors are recesses, fixtures stand out from the wall
        data.obstacles.forEach((obs) => {
            const ow = obs.w * 0.001;
            const oh = obs.h * 0.001;
            const ox = obs.x * 0.001 + offsetX + ow / 2;
            const oy = obs.y * 0.001 + offsetY + oh / 2;
            const depth = (obs.depth_mm || 80) * 0.001;
            const unsure = obs.type === 'unverified';
            const edge = unsure ? tone.review : tone.keep;

            if (obs.type === 'window' || obs.type === 'door') {
                const recess = new THREE.Mesh(
                    new THREE.BoxGeometry(ow, oh, depth),
                    new THREE.MeshLambertMaterial({ color: tone.canvas })
                );
                recess.position.set(ox, oy, -depth / 2 + 0.001);
                this.obstacleGroup.add(recess);
                this.obstacleGroup.add(this.edges(ow, oh, edge, ox, oy, 0.008));
            } else {
                const fixture = new THREE.Mesh(
                    new THREE.BoxGeometry(ow, oh, 0.04),
                    new THREE.MeshLambertMaterial({ color: edge, transparent: true, opacity: unsure ? 0.25 : 0.45 })
                );
                fixture.position.set(ox, oy, 0.02);
                this.obstacleGroup.add(fixture);
                this.obstacleGroup.add(this.edges(ow, oh, edge, ox, oy, 0.042, unsure));
            }
        });

        // The path: stripes in the paint colour, moves with the spray off dashed
        const waypoints = data.waypoints || [];
        const spray = [];
        const moves = [];
        for (let i = 1; i < waypoints.length; i++) {
            const a = waypoints[i - 1];
            const b = waypoints[i];
            const pair = [
                new THREE.Vector3(a.x * 0.001 + offsetX, a.y * 0.001 + offsetY, 0.045),
                new THREE.Vector3(b.x * 0.001 + offsetX, b.y * 0.001 + offsetY, 0.045)
            ];
            (a.spray_active && b.spray_active && a.row === b.row ? spray : moves).push(...pair);
        }
        if (spray.length) {
            this.pathGroup.add(new THREE.LineSegments(
                new THREE.BufferGeometry().setFromPoints(spray),
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
            const start = new THREE.Mesh(new THREE.SphereGeometry(0.03, 16, 12), new THREE.MeshBasicMaterial({ color: tone.accent }));
            start.position.set(waypoints[0].x * 0.001 + offsetX, waypoints[0].y * 0.001 + offsetY, 0.05);
            this.pathGroup.add(start);
        }

        this.pathGroup.visible = this.showPath;
        this.obstacleGroup.visible = this.showObstacles;
        if (!options.keepCamera) this.resetView();
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
        const size = this.wallSize();
        this.lookFrom(0, 0, size * 1.5);
    }

    setIsometricView() {
        if (!this.currentData) return;
        const size = this.wallSize();
        this.lookFrom(size * 0.95, size * 0.55, size * 1.2);
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
        return this.showObstacles;
    }
}

window.WallViewer3D = WallViewer3D;
