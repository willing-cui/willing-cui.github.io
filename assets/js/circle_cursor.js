class CircleCursor {
    constructor(options = {}) {
        this.size = options.size;
        this.color = options.color || '#333';
        this.trailSize = options.trailSize;
        this.trailDelay = options.trailDelay;
        this.cursors = [];

        this.init();
    }

    init() {
        // 隐藏默认光标
        document.body.style.cursor = 'none';

        // 创建主光标
        this.createMainCursor();

        // 创建尾迹光标
        for (let i = 0; i < this.trailSize; i++) {
            this.createTrailCursor(i);
        }

        this.bindEvents();
    }

    createMainCursor() {
        const cursor = document.createElement('div');
        cursor.className = 'circle-cursor main';
        cursor.style.cssText = `
      position: fixed;
      width: ${this.size}px;
      height: ${this.size}px;
      border: 3px solid ${this.color};
      border-radius: 50%;
      pointer-events: none;
      z-index: 9999;
      transform: translate(-50%, -50%);
      transition: transform 0.15s ease;
    `;
        document.body.appendChild(cursor);
        this.mainCursor = cursor;
    }

    createTrailCursor(index) {
        const cursor = document.createElement('div');
        const size = this.size * (1 - index * 0.2);
        cursor.className = 'circle-cursor trail';
        cursor.style.cssText = `
      position: fixed;
      width: ${size}px;
      height: ${size}px;
      background: ${this.color};
      border-radius: 50%;
      pointer-events: none;
      z-index: 9998;
      transform: translate(-50%, -50%);
      opacity: ${0.8 - index * 0.15};
      transition: all 0.3s ease;
    `;
        document.body.appendChild(cursor);
        this.cursors.push({
            element: cursor,
            x: 0,
            y: 0,
            delay: (index + 1) * this.trailDelay
        });
    }

    bindEvents() {
        let mouseX = 0;
        let mouseY = 0;

        document.addEventListener('mousemove', (e) => {
            mouseX = e.clientX;
            mouseY = e.clientY;

            // 立即更新主光标
            this.mainCursor.style.left = mouseX + 'px';
            this.mainCursor.style.top = mouseY + 'px';

            // 延迟更新尾迹
            this.updateTrails(mouseX, mouseY);
        });

        // 点击效果
        document.addEventListener('mousedown', () => {
            this.mainCursor.style.transform = 'translate(-50%, -50%) scale(0.8)';
            this.mainCursor.style.background = this.color;
        });

        document.addEventListener('mouseup', () => {
            this.mainCursor.style.transform = 'translate(-50%, -50%) scale(1)';
            this.mainCursor.style.background = 'transparent';
        });

        // 悬停效果
        document.addEventListener('mouseover', (e) => {
            if (e.target.matches('a, button, .clickable')) {
                this.mainCursor.style.transform = 'translate(-50%, -50%) scale(1.3)';
                this.mainCursor.style.borderColor = 'rgba(255, 215, 0, 0.7)';
            }
        });

        document.addEventListener('mouseout', (e) => {
            if (e.target.matches('a, button, .clickable')) {
                this.mainCursor.style.transform = 'translate(-50%, -50%) scale(1)';
                this.mainCursor.style.borderColor = this.color;
            }
        });
    }

    updateTrails(x, y) {
        this.cursors.forEach((cursor, index) => {
            setTimeout(() => {
                cursor.element.style.left = x + 'px';
                cursor.element.style.top = y + 'px';
            }, cursor.delay);
        });
    }

    destroy() {
        document.body.style.cursor = '';
        this.mainCursor?.remove();
        this.cursors.forEach(cursor => cursor.element.remove());
    }
}

// 使用示例
const customCursor = new CircleCursor({
    size: 20,
    color: 'rgba(255, 255, 255, 0.5)',
    trailSize: 5,
    trailDelay: 30
});