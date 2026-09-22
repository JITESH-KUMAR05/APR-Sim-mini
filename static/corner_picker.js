/**
 * CornerPicker - lets the operator mark the four wall corners on a photo.
 * Points are 0-1 fractions of the image, ordered top-left, top-right, bottom-right, bottom-left.
 */
(function () {
    class CornerPicker {
        constructor(canvas, onChange) {
            this.canvas = canvas;
            this.ctx = canvas.getContext('2d');
            this.onChange = onChange || function () {};
            this.image = null;
            this.points = [];
            this.frame = { x: 0, y: 0, w: 0, h: 0 };
            canvas.addEventListener('click', (event) => this.handleClick(event));
        }

        setImage(image) {
            this.image = image;
            this.points = [];
            const scale = Math.min(this.canvas.width / image.naturalWidth, this.canvas.height / image.naturalHeight);
            const w = image.naturalWidth * scale;
            const h = image.naturalHeight * scale;
            this.frame = { x: (this.canvas.width - w) / 2, y: (this.canvas.height - h) / 2, w, h };
            this.render();
            this.onChange();
        }

        handleClick(event) {
            if (!this.image || this.points.length >= 4) return;
            const box = this.canvas.getBoundingClientRect();
            const px = (event.clientX - box.left) * (this.canvas.width / box.width);
            const py = (event.clientY - box.top) * (this.canvas.height / box.height);
            const nx = (px - this.frame.x) / this.frame.w;
            const ny = (py - this.frame.y) / this.frame.h;
            if (nx < 0 || nx > 1 || ny < 0 || ny > 1) return;
            this.points.push([nx, ny]);
            this.render();
            this.onChange();
        }

        undo() {
            this.points.pop();
            this.render();
            this.onChange();
        }

        isComplete() {
            return this.points.length === 4;
        }

        getCorners() {
            return this.isComplete() ? this.points.map((p) => [p[0], p[1]]) : null;
        }

        render() {
            const { ctx, canvas, frame } = this;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            if (!this.image) return;
            ctx.drawImage(this.image, frame.x, frame.y, frame.w, frame.h);

            const toCanvas = (p) => [frame.x + p[0] * frame.w, frame.y + p[1] * frame.h];
            ctx.lineWidth = 2;
            ctx.strokeStyle = '#5ce1d2';
            ctx.beginPath();
            this.points.forEach((p, i) => {
                const [x, y] = toCanvas(p);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            if (this.isComplete()) ctx.closePath();
            ctx.stroke();

            ctx.font = '14px "DM Mono", monospace';
            this.points.forEach((p, i) => {
                const [x, y] = toCanvas(p);
                ctx.fillStyle = '#f4b860';
                ctx.beginPath();
                ctx.arc(x, y, 6, 0, Math.PI * 2);
                ctx.fill();
                ctx.fillStyle = '#e7f3f1';
                ctx.fillText(String(i + 1), x + 10, y - 8);
            });
        }
    }

    window.CornerPicker = CornerPicker;
})();
