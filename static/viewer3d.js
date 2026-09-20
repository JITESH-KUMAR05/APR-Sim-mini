/**
 * PaintPilot 3D Digital Twin Viewer
 * High-performance WebGL renderer powered by Three.js
 * Supports 360° orbital rotation, pan, zoom, layer toggles, and animated rover traversal.
 * Reference: APR Software Subsystem (MSME Grant INC25ETS084848 / WBS 1.2 §3)
 */

class WallViewer3D {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.wallGroup = null;
        this.pathGroup = null;
        this.obstacleGroup = null;
        this.roverMarker = null;

        this.animationId = null;
        this.isSimulating = false;
        this.simIndex = 0;
        this.simWaypoints = [];
        this.simSpeed = 1.0;

        this.currentData = null;
        this.wireframeMode = false;
        this.showPath = true;
        this.showObstacles = true;

        this.init();
    }

    init() {
        if (!this.container) return;

        const width = this.container.clientWidth || 800;
        const height = this.container.clientHeight || 500;

        // 1. Scene setup
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x060f14); // Deep industrial slate

        // Subtle fog for depth perception
        this.scene.fog = new THREE.FogExp2(0x060f14, 0.08);

        // 2. Camera setup
        this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
        this.camera.position.set(0, 0, 7.5);

        // 3. Renderer setup
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.container.appendChild(this.renderer.domElement);

        // 4. OrbitControls
        if (THREE.OrbitControls) {
            this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
            this.controls.enableDamping = true;
            this.controls.dampingFactor = 0.08;
            this.controls.screenSpacePanning = true;
            this.controls.minDistance = 1.5;
            this.controls.maxDistance = 25.0;
        }

        // 5. Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.65);
        this.scene.add(ambientLight);

        const dirLight = new THREE.DirectionalLight(0xffffff, 0.85);
        dirLight.position.set(5, 8, 10);
        dirLight.castShadow = true;
        this.scene.add(dirLight);

        const fillLight = new THREE.DirectionalLight(0x5ce1d2, 0.3);
        fillLight.position.set(-6, -4, 4);
        this.scene.add(fillLight);

        // 6. Ground Reference Grid
        this.grid = new THREE.GridHelper(16, 32, 0x23424a, 0x10252b);
        this.grid.position.y = -1.4;
        this.scene.add(this.grid);

        // Groups for dynamic geometry
        this.wallGroup = new THREE.Group();
        this.pathGroup = new THREE.Group();
        this.obstacleGroup = new THREE.Group();
        this.scene.add(this.wallGroup);
        this.scene.add(this.pathGroup);
        this.scene.add(this.obstacleGroup);

        // Animated Rover Nozzle Indicator
        this.createRoverMarker();

        // Handle window resize
        window.addEventListener('resize', () => this.onResize());

        // Start render loop
        this.animate = this.animate.bind(this);
        requestAnimationFrame(this.animate);
    }

    createRoverMarker() {
        const markerGroup = new THREE.Group();

        // Nozzle cylinder body
        const geom = new THREE.CylinderGeometry(0.04, 0.06, 0.15, 16);
        const mat = new THREE.MeshStandardMaterial({
            color: 0xf4b860,
            metalness: 0.8,
            roughness: 0.2,
            emissive: 0xf4b860,
            emissiveIntensity: 0.4
        });
        const mesh = new THREE.Mesh(geom, mat);
        mesh.rotation.x = Math.PI / 2;
        markerGroup.add(mesh);

        // Spray cone indicator (semi-transparent)
        const coneGeom = new THREE.ConeGeometry(0.18, 0.35, 16, 1, true);
        const coneMat = new THREE.MeshBasicMaterial({
            color: 0x5ce1d2,
            transparent: true,
            opacity: 0.35,
            side: THREE.DoubleSide
        });
        const coneMesh = new THREE.Mesh(coneGeom, coneMat);
        coneMesh.rotation.x = -Math.PI / 2;
        coneMesh.position.z = -0.18;
        markerGroup.add(coneMesh);

        markerGroup.visible = false;
        this.roverMarker = markerGroup;
        this.scene.add(markerGroup);
    }

    onResize() {
        if (!this.container || !this.renderer || !this.camera) return;
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    loadWallData(data) {
        this.currentData = data;
        this.stopSimulation();

        // Clear existing models
        while (this.wallGroup.children.length > 0) {
            const obj = this.wallGroup.children.pop();
            if (obj.geometry) obj.geometry.dispose();
            this.wallGroup.remove(obj);
        }
        while (this.obstacleGroup.children.length > 0) {
            const obj = this.obstacleGroup.children.pop();
            if (obj.geometry) obj.geometry.dispose();
            this.obstacleGroup.remove(obj);
        }
        while (this.pathGroup.children.length > 0) {
            const obj = this.pathGroup.children.pop();
            if (obj.geometry) obj.geometry.dispose();
            this.pathGroup.remove(obj);
        }

        const wallW = data.wall.width_mm * 0.001;
        const wallH = data.wall.height_mm * 0.001;
        const wallD = 0.25; // 250mm visual wall depth

        // Center origin of wall in 3D scene
        const offsetX = -wallW / 2;
        const offsetY = -wallH / 2;

        if (this.grid) {
            this.grid.position.y = offsetY;
        }

        // 1. Base Wall Backing (Concrete/Masonry)
        const wallGeom = new THREE.BoxGeometry(wallW, wallH, wallD);
        const wallMat = new THREE.MeshStandardMaterial({
            color: 0x142830,
            roughness: 0.85,
            metalness: 0.15,
            wireframe: this.wireframeMode
        });
        const wallMesh = new THREE.Mesh(wallGeom, wallMat);
        wallMesh.position.set(0, 0, -wallD / 2);
        wallMesh.receiveShadow = true;
        this.wallGroup.add(wallMesh);

        // 2. Active Paintable Front Face (Highlighted in Cyan/Emerald)
        const frontGeom = new THREE.PlaneGeometry(wallW, wallH);
        const frontMat = new THREE.MeshStandardMaterial({
            color: 0x1a454d,
            emissive: 0x5ce1d2,
            emissiveIntensity: 0.15,
            roughness: 0.5,
            metalness: 0.2,
            side: THREE.DoubleSide
        });
        const frontMesh = new THREE.Mesh(frontGeom, frontMat);
        frontMesh.position.set(0, 0, 0.002);
        this.wallGroup.add(frontMesh);

        // Outer Border Trim
        const borderGeom = new THREE.EdgesGeometry(frontGeom);
        const borderMat = new THREE.LineBasicMaterial({ color: 0x5ce1d2, linewidth: 2 });
        const borderLines = new THREE.LineSegments(borderGeom, borderMat);
        borderLines.position.set(0, 0, 0.004);
        this.wallGroup.add(borderLines);

        // 3. Obstacles (Windows, Doors, Electrical Boxes)
        data.obstacles.forEach((obs) => {
            const ow = obs.w * 0.001;
            const oh = obs.h * 0.001;
            const ox = (obs.x * 0.001) + offsetX + (ow / 2);
            const oy = (obs.y * 0.001) + offsetY + (oh / 2);

            let obsDepth = (obs.depth_mm || 80.0) * 0.001;
            let obsColor = 0xfa746e; // Red/Coral Keep-Out
            let zPos = 0.01;

            if (obs.type === 'window') {
                // Recessed window glazing
                const winGeom = new THREE.BoxGeometry(ow, oh, obsDepth);
                const winMat = new THREE.MeshStandardMaterial({
                    color: 0x0c2538,
                    metalness: 0.9,
                    roughness: 0.1,
                    transparent: true,
                    opacity: 0.85
                });
                const winMesh = new THREE.Mesh(winGeom, winMat);
                winMesh.position.set(ox, oy, -obsDepth / 2);
                this.obstacleGroup.add(winMesh);

                // Frame outline
                const frameGeom = new THREE.EdgesGeometry(new THREE.PlaneGeometry(ow, oh));
                const frameMat = new THREE.LineBasicMaterial({ color: 0xfa746e, linewidth: 3 });
                const frameLine = new THREE.LineSegments(frameGeom, frameMat);
                frameLine.position.set(ox, oy, 0.008);
                this.obstacleGroup.add(frameLine);

            } else if (obs.type === 'switchboard' || obs.type === 'meter_panel') {
                // Extruded fixture
                const fixGeom = new THREE.BoxGeometry(ow, oh, 0.04);
                const fixMat = new THREE.MeshStandardMaterial({
                    color: 0xf4b860,
                    metalness: 0.5,
                    roughness: 0.4
                });
                const fixMesh = new THREE.Mesh(fixGeom, fixMat);
                fixMesh.position.set(ox, oy, 0.02);
                this.obstacleGroup.add(fixMesh);

                const frameGeom = new THREE.EdgesGeometry(new THREE.PlaneGeometry(ow, oh));
                const frameMat = new THREE.LineBasicMaterial({ color: 0xfa746e, linewidth: 2 });
                const frameLine = new THREE.LineSegments(frameGeom, frameMat);
                frameLine.position.set(ox, oy, 0.042);
                this.obstacleGroup.add(frameLine);

            } else {
                // Door / General Opening
                const doorGeom = new THREE.BoxGeometry(ow, oh, obsDepth * 0.5);
                const doorMat = new THREE.MeshStandardMaterial({
                    color: 0x221a22,
                    metalness: 0.3,
                    roughness: 0.7
                });
                const doorMesh = new THREE.Mesh(doorGeom, doorMat);
                doorMesh.position.set(ox, oy, -obsDepth * 0.25);
                this.obstacleGroup.add(doorMesh);

                const frameGeom = new THREE.EdgesGeometry(new THREE.PlaneGeometry(ow, oh));
                const frameMat = new THREE.LineBasicMaterial({ color: 0xfa746e, linewidth: 2 });
                const frameLine = new THREE.LineSegments(frameGeom, frameMat);
                frameLine.position.set(ox, oy, 0.008);
                this.obstacleGroup.add(frameLine);
            }
        });

        // 4. Rover Coverage Toolpath (3D Trajectory Ribbon)
        const waypoints = data.waypoints || [];
        this.simWaypoints = waypoints;

        const pathPointsSpray = [];
        const pathPointsTransit = [];

        for (let i = 1; i < waypoints.length; i++) {
            const p1 = waypoints[i - 1];
            const p2 = waypoints[i];

            const x1 = (p1.x * 0.001) + offsetX;
            const y1 = (p1.y * 0.001) + offsetY;
            const z1 = 0.04; // Slightly offset from wall face

            const x2 = (p2.x * 0.001) + offsetX;
            const y2 = (p2.y * 0.001) + offsetY;
            const z2 = 0.04;

            const isSpray = p1.spray_active && p2.spray_active && (p1.row === p2.row);

            const lineGeom = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(x1, y1, z1),
                new THREE.Vector3(x2, y2, z2)
            ]);

            const lineMat = new THREE.LineBasicMaterial({
                color: isSpray ? 0xf4b860 : 0xfa746e, // Golden for active spray, Coral/Magenta for transit
                linewidth: isSpray ? 3 : 1,
                transparent: true,
                opacity: isSpray ? 0.95 : 0.45
            });

            const line = new THREE.Line(lineGeom, lineMat);
            this.pathGroup.add(line);
        }

        // Add 3D Waypoint Nodes
        const sphereGeom = new THREE.SphereGeometry(0.018, 8, 8);
        const sprayNodeMat = new THREE.MeshBasicMaterial({ color: 0xf4b860 });
        const transitNodeMat = new THREE.MeshBasicMaterial({ color: 0x5ce1d2 });

        waypoints.forEach((wp, idx) => {
            if (idx % 2 === 0 || idx === waypoints.length - 1) { // Sample nodes for visual cleanliness
                const wx = (wp.x * 0.001) + offsetX;
                const wy = (wp.y * 0.001) + offsetY;
                const nodeMesh = new THREE.Mesh(sphereGeom, wp.spray_active ? sprayNodeMat : transitNodeMat);
                nodeMesh.position.set(wx, wy, 0.04);
                this.pathGroup.add(nodeMesh);
            }
        });

        // Set rover marker to first waypoint
        if (waypoints.length > 0 && this.roverMarker) {
            const first = waypoints[0];
            this.roverMarker.position.set(
                (first.x * 0.001) + offsetX,
                (first.y * 0.001) + offsetY,
                0.08
            );
            this.roverMarker.visible = true;
        }

        // Adjust camera distance nicely based on wall dimensions
        const maxDim = Math.max(wallW, wallH);
        this.camera.position.set(0, 0, maxDim * 1.5);
        if (this.controls) {
            this.controls.target.set(0, 0, 0);
            this.controls.update();
        }
    }

    animate() {
        this.animationId = requestAnimationFrame(this.animate);

        if (this.controls) {
            this.controls.update();
        }

        // Animate rover nozzle along waypoints in simulation mode
        if (this.isSimulating && this.simWaypoints.length > 1 && this.currentData) {
            this.stepSimulation();
        }

        this.renderer.render(this.scene, this.camera);
    }

    stepSimulation() {
        if (!this.roverMarker || this.simIndex >= this.simWaypoints.length - 1) {
            this.isSimulating = false;
            const btn = document.getElementById('simPlayBtn');
            if (btn) btn.innerHTML = '▶ Simulate Sweep';
            return;
        }

        const wallW = this.currentData.wall.width_mm * 0.001;
        const wallH = this.currentData.wall.height_mm * 0.001;
        const offsetX = -wallW / 2;
        const offsetY = -wallH / 2;

        const p1 = this.simWaypoints[this.simIndex];
        const p2 = this.simWaypoints[this.simIndex + 1];

        const targetX = (p2.x * 0.001) + offsetX;
        const targetY = (p2.y * 0.001) + offsetY;

        const currX = this.roverMarker.position.x;
        const currY = this.roverMarker.position.y;

        const dx = targetX - currX;
        const dy = targetY - currY;
        const dist = Math.hypot(dx, dy);

        const step = 0.025 * this.simSpeed;

        if (dist < step) {
            this.roverMarker.position.set(targetX, targetY, 0.08);
            this.simIndex++;
        } else {
            this.roverMarker.position.x += (dx / dist) * step;
            this.roverMarker.position.y += (dy / dist) * step;
        }
    }

    startSimulation() {
        if (this.simWaypoints.length < 2) return;
        this.isSimulating = true;
        if (this.simIndex >= this.simWaypoints.length - 1) {
            this.simIndex = 0;
        }
        if (this.roverMarker) this.roverMarker.visible = true;
    }

    stopSimulation() {
        this.isSimulating = false;
        this.simIndex = 0;
    }

    toggleSimulation() {
        this.isSimulating = !this.isSimulating;
        const btn = document.getElementById('simPlayBtn');
        if (btn) {
            btn.innerHTML = this.isSimulating ? '⏸ Pause' : '▶ Simulate Sweep';
        }
    }

    resetView() {
        if (!this.currentData) return;
        const wallW = this.currentData.wall.width_mm * 0.001;
        const wallH = this.currentData.wall.height_mm * 0.001;
        const maxDim = Math.max(wallW, wallH);

        this.camera.position.set(0, 0, maxDim * 1.5);
        if (this.controls) {
            this.controls.target.set(0, 0, 0);
            this.controls.update();
        }
    }

    setIsometricView() {
        if (!this.currentData) return;
        const wallW = this.currentData.wall.width_mm * 0.001;
        const wallH = this.currentData.wall.height_mm * 0.001;
        const maxDim = Math.max(wallW, wallH);

        this.camera.position.set(maxDim * 0.95, maxDim * 0.55, maxDim * 1.2);
        if (this.controls) {
            this.controls.target.set(0, 0, 0);
            this.controls.update();
        }
    }

    setTopDownView() {
        if (!this.currentData) return;
        const wallW = this.currentData.wall.width_mm * 0.001;
        const wallH = this.currentData.wall.height_mm * 0.001;
        const maxDim = Math.max(wallW, wallH, 3.0);

        this.camera.position.set(0, maxDim * 1.6, 0.08);
        if (this.controls) {
            this.controls.target.set(0, 0, 0);
            this.controls.update();
        }
    }

    toggleWireframe() {
        this.wireframeMode = !this.wireframeMode;
        if (this.wallGroup) {
            this.wallGroup.traverse((child) => {
                if (child.material) {
                    child.material.wireframe = this.wireframeMode;
                }
            });
        }
    }

    togglePath() {
        this.showPath = !this.showPath;
        if (this.pathGroup) this.pathGroup.visible = this.showPath;
    }

    toggleObstacles() {
        this.showObstacles = !this.showObstacles;
        if (this.obstacleGroup) this.obstacleGroup.visible = this.showObstacles;
    }
}

// Global instance handle
window.WallViewer3D = WallViewer3D;
