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

// 进度条控制脚本
document.addEventListener('DOMContentLoaded', function () {
    const progressBar = document.getElementById('progress-bar');
    const progressPercentage = document.getElementById('progress-percentage');
    const progressText = document.getElementById('progress-text');
    const earthProgress = document.getElementById('earth-progress');

    // 创建星星背景
    createStars();

    // 加载步骤
    const loadingSteps = [
        { text: "正在初始化系统...", min: 0, max: 20 },
        { text: "加载3D地球模型...", min: 20, max: 40 },
        { text: "准备照片数据...", min: 40, max: 60 },
        { text: "优化渲染性能...", min: 60, max: 80 },
        { text: "完成最后设置...", min: 80, max: 100 }
    ];

    let currentStep = 0;
    let progress = 0;
    let isSimulating = true;

    function updateProgress() {
        if (currentStep >= loadingSteps.length) {
            completeLoading();
            return;
        }

        const step = loadingSteps[currentStep];

        // 更新进度文本
        progressText.innerHTML = step.text +
            '<span class="loading-dots">' +
            '<span>.</span><span>.</span><span>.</span>' +
            '</span>';

        // 模拟每一步的加载
        function simulateStep() {
            if (progress < step.max && isSimulating) {
                progress++;
                updateProgressDisplay();

                // 随机速度，更接近真实加载
                const speed = 20 + Math.random() * 40;
                setTimeout(simulateStep, speed);
            } else if (progress >= step.max) {
                currentStep++;
                if (currentStep < loadingSteps.length) {
                    // 步骤间短暂暂停
                    setTimeout(() => {
                        isSimulating = true;
                        updateProgress();
                    }, 300);
                } else {
                    completeLoading();
                }
            }
        }

        simulateStep();
    }

    function updateProgressDisplay() {
        progressBar.style.width = progress + '%';
        progressPercentage.textContent = progress + '%';

        // 根据进度改变颜色
        if (progress < 30) {
            progressBar.style.background = 'linear-gradient(90deg, ' +
                'rgba(0, 100, 200, 0.6) 0%, ' +
                'rgba(0, 150, 255, 0.8) 50%, ' +
                'rgba(0, 200, 255, 1) 100%)';
        } else if (progress < 70) {
            progressBar.style.background = 'linear-gradient(90deg, ' +
                'rgba(0, 150, 200, 0.6) 0%, ' +
                'rgba(0, 180, 255, 0.8) 50%, ' +
                'rgba(100, 220, 255, 1) 100%)';
        } else {
            progressBar.style.background = 'linear-gradient(90deg, ' +
                'rgba(0, 180, 200, 0.6) 0%, ' +
                'rgba(100, 220, 255, 0.8) 50%, ' +
                'rgba(150, 240, 255, 1) 100%)';
        }
    }

    function completeLoading() {
        progress = 100;
        updateProgressDisplay();
        progressText.innerHTML = "准备就绪！";

        // 完成动画
        setTimeout(() => {
            earthProgress.style.opacity = '0';
            setTimeout(() => {
                earthProgress.style.display = 'none';
                isSimulating = false;
            }, 500);
        }, 1000);
    }

    // 开始加载
    setTimeout(updateProgress, 500);

    // 监听页面实际加载完成
    window.addEventListener('load', function () {
        if (isSimulating) {
            progress = 100;
            currentStep = loadingSteps.length;
            completeLoading();
        }
    });

    // 设置超时保护（10秒后强制完成）
    setTimeout(() => {
        if (isSimulating) {
            progress = 100;
            currentStep = loadingSteps.length;
            completeLoading();
        }
    }, 10000);

    // 添加键盘快捷键 Ctrl+P 显示/隐藏进度条（用于调试）
    document.addEventListener('keydown', function (e) {
        if (e.ctrlKey && e.key === 'p') {
            e.preventDefault();
            if (earthProgress.style.display === 'none') {
                earthProgress.style.display = 'flex';
                earthProgress.style.opacity = '1';
            } else {
                earthProgress.style.opacity = '0';
                setTimeout(() => {
                    earthProgress.style.display = 'none';
                }, 500);
            }
        }
    });
});