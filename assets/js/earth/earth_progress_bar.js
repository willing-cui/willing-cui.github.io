// 创建星星背景
function createStars() {
    const stars = document.getElementById('stars');
    for (let i = 0; i < 100; i++) {
        const star = document.createElement('div');
        star.className = 'star';
        star.style.width = Math.random() * 3 + 1 + 'px';
        star.style.height = star.style.width;
        star.style.left = Math.random() * 100 + '%';
        star.style.top = Math.random() * 100 + '%';
        star.style.animationDelay = Math.random() * 3 + 's';
        star.style.animationDuration = Math.random() * 2 + 2 + 's';
        stars.appendChild(star);
    }
}

// 进度条控制器
class LoadingProgress {
    constructor() {
        this.progressBar = document.getElementById('progress-bar');
        this.progressPercentage = document.getElementById('progress-percentage');
        this.progressText = document.getElementById('progress-text');
        this.earthProgress = document.getElementById('earth-progress');

        this.currentStep = 0;
        this.progress = 0;

        this.init();
    }

    init() {
        // 创建星星背景
        createStars();

        // 添加键盘快捷键 Ctrl+P 显示/隐藏进度条（用于调试）
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'p') {
                e.preventDefault();
                if (this.earthProgress.style.display === 'none') {
                    this.earthProgress.style.display = 'flex';
                    this.earthProgress.style.opacity = '1';
                } else {
                    this.earthProgress.style.opacity = '0';
                    setTimeout(() => {
                        this.earthProgress.style.display = 'none';
                    }, 500);
                }
            }
        });
    }

    updateProgressDisplay() {
        this.progressBar.style.width = this.progress + '%';
        this.progressPercentage.textContent = this.progress + '%';

        // 根据进度改变颜色
        if (this.progress < 30) {
            this.progressBar.style.background = 'linear-gradient(90deg, ' +
                'rgba(0, 100, 200, 0.6) 0%, ' +
                'rgba(0, 150, 255, 0.8) 50%, ' +
                'rgba(0, 200, 255, 1) 100%)';
        } else if (this.progress < 70) {
            this.progressBar.style.background = 'linear-gradient(90deg, ' +
                'rgba(0, 150, 200, 0.6) 0%, ' +
                'rgba(0, 180, 255, 0.8) 50%, ' +
                'rgba(100, 220, 255, 1) 100%)';
        } else {
            this.progressBar.style.background = 'linear-gradient(90deg, ' +
                'rgba(0, 180, 200, 0.6) 0%, ' +
                'rgba(100, 220, 255, 0.8) 50%, ' +
                'rgba(150, 240, 255, 1) 100%)';
        }
    }

    completeLoading() {
        this.progress = 100;
        this.updateProgressDisplay();
        this.progressText.innerHTML = "Completed!";

        // 完成动画
        setTimeout(() => {
            this.earthProgress.style.opacity = '0';
            setTimeout(() => {
                this.earthProgress.style.display = 'none';
            }, 500);
        }, 1000);
    }

    // 手动更新进度的方法
    setProgress(value, text = '') {
        this.progress = Math.max(0, Math.min(100, value));
        this.updateProgressDisplay();

        if (text) {
            this.progressText.innerHTML = text +
                '<span class="loading-dots">' +
                '<span>.</span><span>.</span><span>.</span>' +
                '</span>';
        }
    }

    // 直接完成加载
    finish() {
        this.completeLoading();
    }
}

// 导出进度条实例
let loadingProgress = null;

// 主初始化函数
function initLoadingProgress() {
    loadingProgress = new LoadingProgress();

    // 监听页面实际加载完成
    window.addEventListener('load', () => {
        loadingProgress.finish();
    });

    return loadingProgress;
}

// 导出的函数，用于在其他代码中更新加载进度
export function updateLoadingProgress(progress, text = null, step = null) {
    if (!loadingProgress) {
        console.warn('Loading progress not initialized yet');
        return;
    }

    if (step) {
        progress = loadingProgress.progress;
        progress += step;
    }

    if (progress >= 100) {
        loadingProgress.finish();
    } else if (text) {
        loadingProgress.setProgress(progress, text);
    } else {
        loadingProgress.setProgress(progress);
    }
}

// 获取进度条实例
export function getLoadingProgress() {
    if (loadingProgress) {
        return loadingProgress;
    } else {
        return initLoadingProgress()
    }
}

// DOMContentLoaded 时初始化
// document.addEventListener('DOMContentLoaded', initLoadingProgress);