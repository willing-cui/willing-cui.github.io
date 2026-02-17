class RippleEffect {
    constructor(options = {}) {
        this.color = options.color || 'rgba(255, 255, 255, 0.7)';
        this.duration = options.duration || 600;
        this.init();
    }

    init() {
        // 添加样式
        this.addStyles();

        // 绑定事件
        document.addEventListener('click', this.handleClick.bind(this));
    }

    addStyles() {
        const style = document.createElement('style');
        style.textContent = `
      .ripple-effect {
        position: fixed;
        border-radius: 50%;
        background: ${this.color};
        transform: scale(0);
        animation: ripple ${this.duration}ms linear;
        pointer-events: none;
        z-index: 9999;
      }
      
      @keyframes ripple {
        to {
          transform: scale(0.5);
          opacity: 0;
        }
      }
    `;
        document.head.appendChild(style);
    }

    handleClick(event) {
        const ripple = document.createElement('div');
        ripple.classList.add('ripple-effect');

        // 计算位置和大小
        const size = 100; // 波纹大小
        const x = event.clientX - size / 2;
        const y = event.clientY - size / 2;

        ripple.style.width = ripple.style.height = `${size}px`;
        ripple.style.left = `${x}px`;
        ripple.style.top = `${y}px`;

        document.body.appendChild(ripple);

        // 清理
        setTimeout(() => {
            ripple.remove();
        }, this.duration);
    }

    // 销毁方法
    destroy() {
        document.removeEventListener('click', this.handleClick);
    }
}

// 使用示例
const ripple = new RippleEffect({
    color: 'rgba(255, 255, 255, 0.5)', // 自定义颜色
    duration: 500 // 自定义持续时间
});