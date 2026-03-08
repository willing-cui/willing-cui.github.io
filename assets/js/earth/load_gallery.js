// 全局变量
const modal = document.getElementById('image-modal');
const modalImage = document.getElementById('modal-image');
const basePath = "../images/gallery/";
const grid = document.getElementById('waterfall-grid');
const indicator = document.getElementById('state-indicator')
const photoListContainer = document.getElementById('photo-list'); // 获取滚动容器

// 瀑布流布局相关变量
let columnCount = 0; // 默认列数 (初始化时强制刷新)
let columnHeights = []; // 每列的当前高度
let columnTops = []; // 每列的顶部位置
let gap = 20; // 卡片间距
let isWaterfallApplied = false; // 标记是否已应用瀑布流布局
let resizeTimeout = null; // 防抖计时器

// 状态管理
let photoListDict = null; // 改为null，表示未加载
let currentLandmarkName = null;
let loadedCount = 0;
let isLoading = false;
let allPhotos = [];
let observer = null;
let isInitialized = false; // 标记是否已初始化
const BATCH_SIZE = 20; // 每次加载的数量
const PRELOAD_THRESHOLD = 5; // 预加载阈值（距离底部多少张图片时开始加载下一批）
const placeholderImg = 'data:image/png;base64,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'

// 卡片尺寸缓存
let cardWidth = 0;
let lastContainerWidth = 0;

// 初始化函数
async function initializeGallery() {
    if (isInitialized) {
        return; // 如果已初始化，直接返回
    }

    try {
        const response = await fetch(basePath + 'gallery.json');
        photoListDict = await response.json();

        // 初始化Intersection Observer用于懒加载
        initLazyLoadObserver();

        // 监听滚动事件
        if (photoListContainer) {
            photoListContainer.addEventListener('scroll', handleScroll);
        }

        // 监听窗口大小变化，重新计算瀑布流布局
        window.addEventListener('resize', handleResize);

        isInitialized = true;
        console.log('Gallery initialized successfully');
    } catch (error) {
        console.error('加载文件列表失败:', error);
    }
}

// 清理函数（用于重置状态）
function cleanupGallery() {
    resetLoadState();
    photoListDict = null;
    isInitialized = false;
    currentLandmarkName = null;
    isWaterfallApplied = false;

    // 清理观察器
    if (observer) {
        observer.disconnect();
        observer = null;
    }

    // 移除事件监听
    if (photoListContainer) {
        photoListContainer.removeEventListener('scroll', handleScroll);
    }

    window.removeEventListener('resize', handleResize);
    if (resizeTimeout) {
        clearTimeout(resizeTimeout);
    }
}

// 初始化懒加载观察器
function initLazyLoadObserver() {
    // 确定IntersectionObserver构造函数是否可作为全局对象的属性使用window
    if ('IntersectionObserver' in window) {
        observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    const src = img.getAttribute('data-src');
                    if (src) {
                        img.src = src;
                        img.classList.remove('lazy');
                        img.classList.add('loaded');
                        img.removeAttribute('data-src');

                        // 图片加载完成后重新计算瀑布流布局
                        img.onload = () => {
                            setTimeout(applyWaterfallLayout, 100);
                        };
                    }
                    observer.unobserve(img);
                }
            });
        }, {
            root: photoListContainer,
            rootMargin: '50px 0px',
            threshold: 0.1
        });
    }
}

// 滚动事件处理（带防抖）
let scrollTimeout = null;
function handleScroll() {
    if (scrollTimeout) {
        clearTimeout(scrollTimeout);
    }

    scrollTimeout = setTimeout(() => {
        if (!photoListContainer) return;

        const { scrollTop, scrollHeight, clientHeight } = photoListContainer;
        const scrollBottom = scrollHeight - scrollTop - clientHeight;

        // 当接近底部时加载更多
        if (scrollBottom < 300 && !isLoading && loadedCount < allPhotos.length) {
            loadMorePhotos();
        }
    }, 100);
}

// 窗口大小变化处理（带防抖）
function handleResize() {
    if (resizeTimeout) {
        clearTimeout(resizeTimeout);
    }

    resizeTimeout = setTimeout(() => {
        // 根据窗口宽度动态调整列数
        updateColumnCount();
        // 重新计算瀑布流布局
        if (isWaterfallApplied) {
            console.log("因为窗口大小改变造成布局刷新")
            applyWaterfallLayout();
        }
    }, 500);

    // 重复刷新一次，避免css样式过渡中卡片高度读取不准的情况
    resizeTimeout = setTimeout(() => {
        if (isWaterfallApplied) {
            applyWaterfallLayout();
        }
    }, 1000);
}

// 根据窗口宽度更新列数
function updateColumnCount() {
    if (!grid) return;

    // console.log('设备像素比:', window.devicePixelRatio);
    const containerWidth = grid.clientWidth * window.devicePixelRatio;
    console.log(`显示窗口宽度为 ${containerWidth} px`)

    let newColumnCount = 3; // 默认

    if (containerWidth <= 1080 * 0.4) { // 小桌面
        newColumnCount = 1;
    } else if (containerWidth <= 2560 * 0.4) { // 中等大小桌面
        newColumnCount = 2;
    } else { // 大桌面
        newColumnCount = 3;
    }

    if (newColumnCount !== columnCount) {
        columnCount = newColumnCount;
        return true; // 列数有变化
    }

    return false; // 列数无变化
}

// 计算卡片宽度
function calculateCardWidth() {
    if (!grid) return 0;
    const containerWidth = grid.clientWidth;
    if (containerWidth !== lastContainerWidth || cardWidth === 0) {
        cardWidth = (containerWidth - (gap * (columnCount - 1))) / columnCount;
        lastContainerWidth = containerWidth;
    }
    return cardWidth;
}

// 应用瀑布流布局
function applyWaterfallLayout() {
    if (!grid) return;

    const cards = grid.querySelectorAll('.image-card');
    console.log(`当前图像卡片数量: ${cards.length}`)
    if (cards.length === 0) return;

    // 更新列数
    updateColumnCount();

    // 重置列高度
    columnHeights = new Array(columnCount).fill(0);
    columnTops = new Array(columnCount).fill(0);

    // 获取容器和卡片信息
    const containerWidth = grid.clientWidth;
    cardWidth = (containerWidth - (gap * (columnCount - 1))) / columnCount;

    // 批量计算并应用样式
    cards.forEach((card, index) => {
        // 找到当前最低的列
        const minHeight = Math.min(...columnHeights);
        const colIndex = columnHeights.indexOf(minHeight);

        // 计算位置
        const left = colIndex * (cardWidth + gap);
        const top = columnTops[colIndex];

        // 应用绝对定位
        card.style.position = 'absolute';
        card.style.left = left + 'px';
        card.style.top = top + 'px';
        card.style.width = cardWidth + 'px';
        card.style.margin = '0'; // 清除可能的margin

        // 获取卡片实际高度
        const cardHeight = card.offsetHeight;

        // 更新列高度
        columnHeights[colIndex] += cardHeight + gap;
        columnTops[colIndex] += cardHeight + gap;
    });

    // 设置容器高度
    const maxHeight = Math.max(...columnHeights);
    grid.style.position = 'relative';
    grid.style.height = maxHeight + 'px';

    isWaterfallApplied = true;
    console.log(`瀑布流布局应用完成，${columnCount}列，总高度: ${maxHeight}px`);
}

// 显示模态框
window.showModal = (src) => {
    modalImage.src = src;
    modal.classList.add('is-visible');
    document.body.style.overflow = 'hidden';
};

// 隐藏模态框
window.hideModal = (event) => {
    if (event === undefined || event.target.id === 'image-modal' || event.target.id === 'modal-image') {
        modal.classList.remove('is-visible');
        modalImage.src = '';
        document.body.style.overflow = '';
    }
};

// 创建单个图片卡片（支持懒加载）
function createImageCard(dataUrl, name, time, isLazy = true) {
    const card = document.createElement('div');
    card.className = 'image-card';

    // 在应用布局前保持卡片隐藏
    card.style.visibility = 'hidden';
    card.style.opacity = '0';

    const largeImgUrl = dataUrl.replaceAll("glance", "check");
    const imgSrc = isLazy ? placeholderImg : dataUrl;
    const dataSrc = isLazy ? dataUrl : null;

    card.innerHTML = `
        <div class="image-card-content">
            <img 
                src="${imgSrc}" 
                ${dataSrc ? `data-src="${dataSrc}"` : ''}
                alt="Local image: ${name}" 
                class="image-display ${isLazy ? 'lazy' : ''}"
                onclick="showModal('${largeImgUrl}')"
                loading="${isLazy ? 'lazy' : 'eager'}"
            >
        </div>
        <div class="card-meta-container">
            <span>${name.split(".")[0]}</span>
            <span>${time}</span>
        </div>
    `;

    // 如果是懒加载图片，添加到观察器
    if (isLazy && observer) {
        const img = card.querySelector('img');
        observer.observe(img);
    } else {
        // 非懒加载图片加载完成后显示卡片
        const img = card.querySelector('img');
        img.onload = function () {
            requestAnimationFrame(() => {
                card.style.visibility = 'visible';
                card.style.opacity = '1';
            });
        };
    }

    return card;
}

// 重置加载状态
function resetLoadState() {
    loadedCount = 0;
    isLoading = false;
    allPhotos = [];
    isWaterfallApplied = false;

    if (grid) {
        grid.innerHTML = '';
        grid.style.position = '';
        grid.style.height = '';
    }

    // 清除指示栏状态
    if (indicator) {
        const oldLoadMoreIndicator = indicator.querySelector('.loading-more');
        if (oldLoadMoreIndicator) {
            oldLoadMoreIndicator.remove();
        }
    }

    // 重置滚动事件
    if (scrollTimeout) {
        clearTimeout(scrollTimeout);
        scrollTimeout = null;
    }
}

// 更新指示栏状态
function updateIndicator() {
    if (!indicator) return;

    const oldLoadMoreIndicator = indicator.querySelector('.loading-more');
    if (oldLoadMoreIndicator) {
        oldLoadMoreIndicator.remove();
    }

    if (isLoading) {
        const loadMoreIndicator = document.createElement('div');
        loadMoreIndicator.className = 'loading-more';
        loadMoreIndicator.textContent = 'Loading ...';
        indicator.appendChild(loadMoreIndicator);
    } else if (loadedCount < allPhotos.length) {
        const loadMoreIndicator = document.createElement('div');
        loadMoreIndicator.className = 'loading-more';
        loadMoreIndicator.textContent = 'Scroll to Load More ...';
        indicator.appendChild(loadMoreIndicator);
    } else if (allPhotos.length > 0) {
        const loadMoreIndicator = document.createElement('div');
        loadMoreIndicator.className = 'loading-more';
        loadMoreIndicator.textContent = 'Loading Complete';
        indicator.appendChild(loadMoreIndicator);
    }
}

// 主加载照片函数
async function loadPhotos(landmarkName) {
    // 确保已初始化
    if (!isInitialized) {
        await initializeGallery();
    }

    // 如果正在加载相同的内容，直接返回
    if (currentLandmarkName === landmarkName && loadedCount > 0) {
        return;
    }

    currentLandmarkName = landmarkName;
    resetLoadState();

    if (!photoListDict || !photoListDict["photo"] || !photoListDict["photo"][landmarkName]) {
        grid.innerHTML = '<div class="no-photos">暂无照片</div>';
        updateIndicator();
        return;
    }

    var directories = photoListDict["photo"];
    var photo_file_names = directories[landmarkName];
    var photo_time_info = photoListDict["time"] ? (photoListDict["time"][landmarkName] || []) : [];

    if (photo_file_names.length === 0) {
        grid.innerHTML = '<div class="no-photos">暂无照片</div>';
        updateIndicator();
        return;
    }

    // 准备所有照片数据
    photo_file_names.forEach((file_name, index) => {
        var filePath = basePath + 'glance/' + landmarkName + '/' + file_name;
        var time = photo_time_info[index] || '';
        allPhotos.push({
            path: filePath,
            name: file_name,
            time: time
        });
    });

    // 更新指示栏
    updateIndicator();

    // 初始加载第一批
    loadBatch(0, Math.min(BATCH_SIZE, allPhotos.length));
}

// 加载一批照片
function loadBatch(startIndex, endIndex) {
    if (!grid) return;

    isLoading = true;
    updateIndicator();

    // 计算卡片宽度
    calculateCardWidth();

    // 使用文档片段提高性能
    const fragment = document.createDocumentFragment();
    const cards = [];

    for (let i = startIndex; i < endIndex && i < allPhotos.length; i++) {
        const photo = allPhotos[i];
        const isLazy = i >= 5; // 前5张立即加载，其余懒加载
        const card = createImageCard(photo.path, photo.name, photo.time, isLazy);

        // 设置初始宽度以减少布局抖动
        if (cardWidth > 0) {
            card.style.width = cardWidth + 'px';
        }

        fragment.appendChild(card);
        cards.push(card);
    }

    grid.appendChild(fragment);
    loadedCount = endIndex;

    // 应用瀑布流布局
    if (loadedCount > 0) {
        // 等待下一帧以确保DOM已更新
        requestAnimationFrame(() => {
            applyWaterfallLayout();

            // 延迟显示卡片，避免闪烁
            setTimeout(() => {
                cards.forEach(card => {
                    card.style.visibility = 'visible';
                    card.style.opacity = '1';
                });
            }, 50);

            // 图片加载完成后重新计算布局
            if (startIndex < 5) { // 前5张是立即加载的
                setTimeout(() => {
                    applyWaterfallLayout();
                    isLoading = false;
                    updateIndicator();
                }, 200);
                // 强制刷新，防止因图片加载慢导致卡片重叠
                setTimeout(applyWaterfallLayout, 1000);
            } else {
                isLoading = false;
                updateIndicator();
            }
        });
    } else {
        isLoading = false;
        updateIndicator();
    }
}

// 加载更多照片
function loadMorePhotos() {
    if (isLoading || loadedCount >= allPhotos.length) return;

    const nextBatchStart = loadedCount;
    const nextBatchEnd = Math.min(loadedCount + BATCH_SIZE, allPhotos.length);

    loadBatch(nextBatchStart, nextBatchEnd);
}

// 手动触发瀑布流布局重新计算（用于外部调用）
function refreshWaterfallLayout() {
    if (grid && loadedCount > 0) {
        applyWaterfallLayout();
    }
}

// 设置瀑布流列数（用于外部调用）
function setWaterfallColumns(columns) {
    if (columns >= 1 && columns <= 6) {
        columnCount = columns;
        if (isWaterfallApplied) {
            applyWaterfallLayout();
        }
    }
}

// 导出函数
window.loadPhotos = loadPhotos; // 暴露到window对象
window.refreshWaterfallLayout = refreshWaterfallLayout;
window.setWaterfallColumns = setWaterfallColumns;
export { loadPhotos, initializeGallery, cleanupGallery, refreshWaterfallLayout, setWaterfallColumns };

// 如果需要在页面加载时自动初始化
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeGallery);
} else {
    initializeGallery();
}