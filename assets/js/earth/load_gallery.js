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
            applyWaterfallLayout();
        }
    }, 250);
}

// 根据窗口宽度更新列数
function updateColumnCount() {
    if (!grid) return;

    const containerWidth = grid.clientWidth;
    console.log(`显示窗口宽度为 ${containerWidth} px`)

    let newColumnCount = 3; // 默认

    if (containerWidth <= 576 * 0.5) { // 手机
        newColumnCount = 1;
    } else if (containerWidth <= 768 * 0.5) { // 平板
        newColumnCount = 2;
    } else if (containerWidth <= 1024 * 0.5) { // 小桌面
        newColumnCount = 3;
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
    const imgSrc = isLazy ? 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAYAAABw4pVUAAAACXBIWXMAAAsTAAALEwEAmpwYAAAD+klEQVR4nO2cT4gcRRyFyz/BBDWCSEKIoqKiIAgSAiEiBIIBk5MHFYIHjx7Uk3+uSjwlJ8Gck0NIJAQNKHoxkIABSRAERTwERRQ3y850v1e9izGalBTOwLCMM9093TP1a94H7zIM1V31TTVdNVXlnBBCCCGEEEIIIYQQQgghhBBClKLf7z9A8gxJTzIonBQP4FOSj7cpo59ARYOxZLHtGhcy6BkBwOdra2vbG79Ax+j1evcD+GIg5XTjFxg+puKFGi+8o/T/e6pEIWy88GEXbLzgjsO22k1C6iEhiSEhiSEhiSEhiSEhiSEhiSEhiWFSSJyKAfAGyaODvNmVGQFTQkIIt5I8BODa+kk5AH+R/CCEcJszjCkhAI5Nmy0FcNwZxowQAC+WncIG8JIziiUh31T4T+GSM4oJIcvLy3cBuFmhh9zs9XqbnUFMCMnz/KGq/7wBeMQZxISQlZWVu6v2kCzL7nEGMSFkUN7lCj3kW2cUM0IAHKzQQw46o5gREkK4heTJEkJOxe86o5gREgkh3A7gMIDrY3rFdQBH4necYUwJGZLn+cMk34qj8hiSb8fPXAcwKaTLUELSQkISQ0ISQ0ISQ0ISQ0ISQ0IqEkLYQPIxADu890+EEDa6BpGQkpDcPW43GIA/SX5J8kATUzYSMoWlpaU7SZ4oObH5VVEUW90MSMgEiqLYQvK7CtP+scf8XhTFk64mEvI/rK6ubgPwQxUZI8m898+4GkjIGLIse5DklZoyhlkj+byriISsI25LJvnbjDKGj6+/Sb7quiKE5K64GhHAZwAuxl2qAD4kub+NFYpZlj0F4GoTMkak3ADwumkheZ4/HQVMqewV7/0LTd0vgJ0t76t/36QQAK+NW7s7IR/N2lu898/GrcgtyhjmaFyf3Ea7TaVqwYP/zN+r+Vg4G0LYVOc+8zzfQ7KYg4zhvX4yaXSfhJB4gyQ/nrGi56uuxYqj68FIO8w55+Jas1nbrRJlC/be3wfg64Z+fd+XPcYDwMvjFkvMMZdi3eu2W2XKFBxHtCR/abiiP8fJvyn39srglXRRMoY/oB/XHzSzMCF5nu8lmbdU0avxTW3CS8ONRcsYya+jRzItREgcLM3hcVEAeG70ut77dxIQMC7xdXvX3IXM8iZVs6dcixt94rUBvJtAw0/KKoB9cxMSQrij5FLQpqX8M3IOVdIZHX+1KgTAvQAuLLrCNJTWhMSNMwB+WnQFaSxtCuktunI0mNaEKJQQdiASwrQiIUwrEsK0IiHsvhCd9856AfBH40IGB8svvHK0mRNtLafJEqhcMJZ+K4fxj5xlfnpOCwiC8UTOAHi0FRlCCCGEEEIIIYQQQgghhBBCCNc9/gW1WARKxMla2QAAAABJRU5ErkJggg==' : dataUrl;
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
        img.onload = function() {
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
                }, 100);
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