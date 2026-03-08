// const skillButton = document.getElementById("skillButton");
skillButton.addEventListener("click", () => { delayedInitCarousel(); });
window.addEventListener('load', () => { delayedInitCarousel(); });
var prevBtn = document.querySelector('.carousel-prev');
var nextBtn = document.querySelector('.carousel-next');
var autoPlayToggle = document.querySelector('.auto-play-toggle');
var carouselTrack = document.querySelector('.carousel-track');
var thumbnailsContainer = document.querySelector('.carousel-thumbnails');
var indicatorsContainer = document.querySelector('.carousel-indicators');

var carouselTimer = undefined;
var carouselInitialized = false;

// base64占位图
const placeholderImg = 'data:image/png;base64,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'

// 图片数据
const projects = [
    {
        src: "images/mrc_board.webp",
        alt: "A PCB for signal sampling and weighted signal combination.",
        description_en: "A PCB for signal sampling and weighted signal combination.",
        description_zh: "用于8路模拟信号采样和加权相加的PCB模块。",
        color: "#fff"
    },
    {
        src: "images/transceiver_box.webp",
        alt: "An optical transceiver with housing and mounting bracket.",
        description_en: "An optical transceiver with housing and mounting bracket.",
        description_zh: "带外壳和安装支架的无线光通信收发器。",
        color: "#fff"
    },
    {
        src: "images/omnidirectional_transceiver.webp",
        alt: "An omnidirectional optical transceiver.",
        description_en: "An omnidirectional optical transceiver.",
        description_zh: "一个全向8发8收无线光通信收发器。",
        color: "#fff"
    },
    {
        src: "images/omnidirectional_transceiver_boards.webp",
        alt: "PCBs of the omnidirectional optical transceiver.",
        description_en: "PCBs of the omnidirectional optical transceiver.",
        description_zh: "全向8发8收无线光通信收发器的各个PCB组件。",
        color: "#fff"
    },
    {
        src: "images/transceiver_kit.webp",
        alt: "An directional optical transceiver (front view).",
        description_en: "An directional optical transceiver (front view).",
        description_zh: "一个定向光通信收发器 (前视图)",
        color: "#fff"
    },
    {
        src: "images/transceiver_kit_b.webp",
        alt: "An directional optical transceiver (back view).",
        description_en: "An directional optical transceiver (back view).",
        description_zh: "一个定向光通信收发器 (后视图)",
        color: "#fff"
    }
];

// 状态变量
let currentIndex = 0;
let autoPlayInterval = null;
let isAutoPlaying = true;
let currentLanguage = null;
const intervalDuration = 5000; // 5秒

// 获取当前语言
function getCurrentLanguage() {
    const savedLang = localStorage.getItem('preferred-language');
    const browserLang = navigator.language.startsWith('zh') ? 'zh' : 'en';
    return savedLang || browserLang || 'en';
}

// 初始化轮播
function initCarousel() {

    currentLanguage = getCurrentLanguage()

    console.log('正在初始化轮播图，项目数：', projects.length);

    // 创建卡片、缩略图和指示器
    projects.forEach((project, index) => {
        // 创建主卡片
        const card = document.createElement('div');
        card.className = 'carousel-card';
        card.style.cssText = `--clr: ${project.color};`;

        // 创建图片元素 - 先设置占位图
        const img = document.createElement('img');
        img.loading = 'lazy';
        img.src = placeholderImg;  // 先使用占位图
        
        // 预加载真实图片
        const realImage = new Image();
        realImage.src = project.src;
        realImage.onload = function() {
            // 图片加载完成后替换
            img.src = project.src;
        };
        realImage.onerror = function() {
            console.warn('图片加载失败：', project.src);
        };

        // 创建图片容器
        const imageSpan = document.createElement('span');
        imageSpan.className = 'image main card-image';
        imageSpan.appendChild(img);

        // 创建文本块
        const textBlock = document.createElement('div');
        textBlock.className = 'card-text-block';
        if (currentLanguage === 'zh') {
            textBlock.innerHTML = `<div style="font-size: small">${project.description_zh}</div>`;
        } else if (currentLanguage === 'en') {
            textBlock.innerHTML = `<div style="font-size: small">${project.description_en}</div>`;
        }

        // 组装卡片
        card.appendChild(imageSpan);
        card.appendChild(textBlock);
        carouselTrack.appendChild(card);

        // 添加鼠标悬停效果
        card.addEventListener('mousemove', function (e) {
            const rect = this.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            this.style.setProperty('--x', x + 'px');
            this.style.setProperty('--y', y + 'px');
        });

        // 创建缩略图
        const thumbnail = document.createElement('div');
        thumbnail.className = `thumbnail ${index === 0 ? 'active' : ''}`;
        thumbnail.dataset.index = index;
        thumbnail.style.cssText = `--clr: ${project.color};`;

        const thumbImg = document.createElement('img');
        thumbImg.loading = 'lazy';
        thumbImg.src = placeholderImg;  // 先使用占位图
        
        // 预加载缩略图的真实图片
        const thumbRealImage = new Image();
        thumbRealImage.src = project.src;
        thumbRealImage.onload = function() {
            thumbImg.src = project.src;
        };

        thumbnail.appendChild(thumbImg);
        thumbnailsContainer.appendChild(thumbnail);

        // 创建指示器
        const indicator = document.createElement('div');
        indicator.className = `indicator ${index === 0 ? 'active' : ''}`;
        indicator.dataset.index = index;
        indicatorsContainer.appendChild(indicator);

        // 添加点击事件
        thumbnail.addEventListener('click', () => goToSlide(index));
        indicator.addEventListener('click', () => goToSlide(index));
    });

    // 事件监听
    if (prevBtn) {
        prevBtn.addEventListener('click', () => {
            prevSlide();
            resetAutoPlay();
        });
    }

    if (nextBtn) {
        nextBtn.addEventListener('click', () => {
            nextSlide();
            resetAutoPlay();
        });
    }

    if (autoPlayToggle) {
        autoPlayToggle.addEventListener('click', toggleAutoPlay);
    }

    // 添加键盘支持
    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft') {
            prevSlide();
            resetAutoPlay();
        } else if (e.key === 'ArrowRight') {
            nextSlide();
            resetAutoPlay();
        } else if (e.key === ' ' || e.key === 'Spacebar') {
            e.preventDefault();
            toggleAutoPlay();
        }
    });

    // 鼠标悬停时暂停自动播放
    if (carouselTrack) {
        carouselTrack.addEventListener('mouseenter', () => {
            if (isAutoPlaying) {
                stopAutoPlay();
            }
        });

        carouselTrack.addEventListener('mouseleave', () => {
            if (isAutoPlaying) {
                startAutoPlay();
            }
        });
    }

    // 添加触摸滑动支持
    if (carouselTrack) {
        let touchStartX = 0;
        let touchEndX = 0;

        carouselTrack.addEventListener('touchstart', (e) => {
            touchStartX = e.changedTouches[0].screenX;
        }, { passive: true });

        carouselTrack.addEventListener('touchend', (e) => {
            touchEndX = e.changedTouches[0].screenX;
            handleSwipe();
        }, { passive: true });

        function handleSwipe() {
            const swipeThreshold = 50;
            const diff = touchStartX - touchEndX;

            if (Math.abs(diff) > swipeThreshold) {
                if (diff > 0) {
                    nextSlide();
                } else {
                    prevSlide();
                }
                resetAutoPlay();
            }
        }
    }

    // 更新轮播显示
    updateCarousel();

    // 开始自动播放
    startAutoPlay();

    carouselInitialized = true;
    console.log('轮播图初始化成功');
}

// 跳转到指定幻灯片
function goToSlide(index) {
    if (index < 0 || index >= projects.length) return;

    currentIndex = index;
    updateCarousel();
    resetAutoPlay();
}

// 更新轮播显示
function updateCarousel() {
    if (!carouselTrack || carouselTrack.children.length === 0) return;

    // 获取第一个卡片的宽度作为偏移基准
    const cardWidth = carouselTrack.children[0].offsetWidth;
    // 计算总偏移量
    const offset = currentIndex * cardWidth;

    // 更新轨道位置
    carouselTrack.style.transform = `translateX(-${offset}px)`;
    carouselTrack.style.transition = 'transform 0.5s ease-in-out';

    // 更新缩略图状态
    document.querySelectorAll('.thumbnail').forEach((thumb, index) => {
        thumb.classList.toggle('active', index === currentIndex);
    });

    // 更新指示器状态
    document.querySelectorAll('.indicator').forEach((indicator, index) => {
        indicator.classList.toggle('active', index === currentIndex);
    });

    // 确保当前缩略图在可视区域内，强制滑动窗口
    // const activeThumbnail = document.querySelector('.thumbnail.active');
    // if (activeThumbnail) {
    //     activeThumbnail.scrollIntoView({
    //         behavior: 'smooth',
    //         inline: 'center',
    //         block: 'nearest'
    //     });
    // }
}

// 添加窗口大小调整时的重新计算
let resizeTimeout;
window.addEventListener('resize', function () {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(function () {
        updateCarousel();
    }, 250);
});

// 下一张
function nextSlide() {
    currentIndex = (currentIndex + 1) % projects.length;
    updateCarousel();
}

// 上一张
function prevSlide() {
    currentIndex = (currentIndex - 1 + projects.length) % projects.length;
    updateCarousel();
}

// 开始自动播放
function startAutoPlay() {
    if (autoPlayInterval) {
        clearInterval(autoPlayInterval);
    }

    if (isAutoPlaying) {
        autoPlayInterval = setInterval(nextSlide, intervalDuration);
        autoPlayToggle.classList.add('active');
        autoPlayToggle.querySelector('i').className = 'icon solid fa-pause';
    }
}

// 停止自动播放
function stopAutoPlay() {
    if (autoPlayInterval) {
        clearInterval(autoPlayInterval);
        autoPlayInterval = null;
    }
    autoPlayToggle.classList.remove('active');
    autoPlayToggle.querySelector('i').className = 'icon solid fa-play';
}

// 重置自动播放计时器
function resetAutoPlay() {
    if (isAutoPlaying) {
        clearInterval(autoPlayInterval);
        startAutoPlay();
    }
}

// 切换自动播放状态
function toggleAutoPlay() {
    isAutoPlaying = !isAutoPlaying;
    if (isAutoPlaying) {
        startAutoPlay();
    } else {
        stopAutoPlay();
    }
}

function delayedInitCarousel() {
    // 如果已经初始化完成，避免重复初始化
    if (carouselInitialized && currentLanguage == getCurrentLanguage()) {
        console.log("轮播图已初始化")
        return
    }
    // 获取DOM元素
    prevBtn = document.querySelector('.carousel-prev');
    nextBtn = document.querySelector('.carousel-next');
    autoPlayToggle = document.querySelector('.auto-play-toggle');
    carouselTrack = document.querySelector('.carousel-track');
    thumbnailsContainer = document.querySelector('.carousel-thumbnails');
    indicatorsContainer = document.querySelector('.carousel-indicators');

    // 检查元素是否存在
    if (!carouselTrack || !thumbnailsContainer || !indicatorsContainer) {
        console.log('未找到轮播图组件');
        if (carouselTimer == undefined) {
            carouselTimer = setTimeout('delayedInitCarousel()', 500);
        }
    } else {
        // 初始化轮播
        initCarousel();
    }

}

