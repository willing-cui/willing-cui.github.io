// 全局变量
const modal = document.getElementById('image-modal');
const modalImage = document.getElementById('modal-image');
const basePath = "../images/gallery/";
const grid = document.getElementById('waterfall-grid');
const photoListContainer = document.getElementById('photo-list'); // 获取滚动容器

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
    
    // 清理观察器
    if (observer) {
        observer.disconnect();
        observer = null;
    }
    
    // 移除事件监听
    if (photoListContainer) {
        photoListContainer.removeEventListener('scroll', handleScroll);
    }
}

// 初始化懒加载观察器
function initLazyLoadObserver() {
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
    const largeImgUrl = dataUrl.replaceAll("glance", "check");
    
    const imgSrc = isLazy ? 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZjJmMmYyIi8+PC9zdmc+' : dataUrl;
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
    }
    
    return card;
}

// 重置加载状态
function resetLoadState() {
    loadedCount = 0;
    isLoading = false;
    allPhotos = [];
    if (grid) {
        grid.innerHTML = '';
    }
    
    // 重置滚动事件
    if (scrollTimeout) {
        clearTimeout(scrollTimeout);
        scrollTimeout = null;
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
        return;
    }
    
    var directories = photoListDict["photo"];
    var photo_file_names = directories[landmarkName];
    var photo_time_info = photoListDict["time"] ? (photoListDict["time"][landmarkName] || []) : [];
    
    if (photo_file_names.length === 0) {
        grid.innerHTML = '<div class="no-photos">暂无照片</div>';
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
    
    // 初始加载第一批
    loadBatch(0, Math.min(BATCH_SIZE, allPhotos.length));
}

// 加载一批照片
function loadBatch(startIndex, endIndex) {
    if (!grid) return;
    
    isLoading = true;
    
    // 显示加载指示器
    if (startIndex === 0) {
        grid.innerHTML = '<div class="loading-indicator">加载中...</div>';
    }
    
    // 移除之前的加载指示器
    const loadingIndicator = grid.querySelector('.loading-indicator');
    if (loadingIndicator) {
        loadingIndicator.remove();
    }
    
    // 使用文档片段提高性能
    const fragment = document.createDocumentFragment();
    
    for (let i = startIndex; i < endIndex && i < allPhotos.length; i++) {
        const photo = allPhotos[i];
        const isLazy = i >= 5; // 前5张立即加载，其余懒加载
        const card = createImageCard(photo.path, photo.name, photo.time, isLazy);
        fragment.appendChild(card);
    }
    
    grid.appendChild(fragment);
    loadedCount = endIndex;
    isLoading = false;
    
    // 如果还有更多图片，显示加载更多提示
    if (loadedCount < allPhotos.length) {
        const loadMoreIndicator = document.createElement('div');
        loadMoreIndicator.className = 'loading-more';
        loadMoreIndicator.textContent = '滚动加载更多...';
        grid.appendChild(loadMoreIndicator);
    }
    
    // 如果已经加载了所有图片，移除滚动监听
    if (loadedCount >= allPhotos.length) {
        const loadMoreIndicator = grid.querySelector('.loading-more');
        if (loadMoreIndicator) {
            loadMoreIndicator.remove();
        }
    }
}

// 加载更多照片
function loadMorePhotos() {
    if (isLoading || loadedCount >= allPhotos.length) return;
    
    const nextBatchStart = loadedCount;
    const nextBatchEnd = Math.min(loadedCount + BATCH_SIZE, allPhotos.length);
    
    // 移除"加载更多"提示
    const loadMoreIndicator = grid.querySelector('.loading-more');
    if (loadMoreIndicator) {
        loadMoreIndicator.remove();
    }
    
    loadBatch(nextBatchStart, nextBatchEnd);
}

// 导出函数
window.loadPhotos = loadPhotos; // 暴露到window对象
export { loadPhotos, initializeGallery, cleanupGallery };

// 添加一些CSS样式来改善体验
if (!document.querySelector('#gallery-styles')) {
    const style = document.createElement('style');
    style.id = 'gallery-styles';
    style.textContent = `
        .image-display.lazy {
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        
        .image-display.loaded {
            opacity: 1;
        }
        
        .loading-more, .loading-indicator {
            text-align: center;
            padding: 20px;
            color: #999;
            font-size: 14px;
            grid-column: 1 / -1;
        }
    `;
    document.head.appendChild(style);
}

// 如果需要在页面加载时自动初始化
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeGallery);
} else {
    initializeGallery();
}