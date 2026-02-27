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

    console.log('Initializing carousel with', projects.length, 'projects');

    // 创建卡片、缩略图和指示器
    projects.forEach((project, index) => {
        // 创建主卡片
        const card = document.createElement('div');
        card.className = 'carousel-card';
        card.style.cssText = `--clr: ${project.color};`;

        // 创建图片元素
        const img = document.createElement('img');
        img.loading = 'lazy';
        img.src = project.src;
        img.alt = project.alt;

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
        thumbImg.src = project.src;
        thumbImg.alt = project.alt;
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
    console.log('Carousel initialized successfully');
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
        console.log("Already initialized carousel")
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
        console.log('Carousel elements not found!');
        if (carouselTimer == undefined) {
            carouselTimer = setTimeout('delayedInitCarousel()', 500);
        }
    } else {
        // 初始化轮播
        initCarousel();
    }

}

