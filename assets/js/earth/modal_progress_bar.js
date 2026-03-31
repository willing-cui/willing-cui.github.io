// DOM元素引用
const modal = document.getElementById('image-modal');
const modalImage = document.getElementById('modal-image');
const imagePlaceholder = document.getElementById('image-placeholder');
const placeholderIcon = imagePlaceholder.querySelector('.placeholder-icon');
const placeholderText = imagePlaceholder.querySelector('.placeholder-text');
const modalProgressBar = document.getElementById('modal-progress-bar');
const modalProgressText = document.getElementById('modal-progress-text');
const modalLoadingProgress = document.getElementById('modal-loading-progress');

// 全局变量
let modalImageLoadProgress = 0;
let modalProgressInterval = null;
let isImageLoaded = false;
let xhr = null; // XMLHttpRequest对象

// 显示模态框
window.showModal = (src) => {
    // 重置状态
    resetModalState();

    // 显示模态框
    modal.classList.add('is-visible');
    document.body.style.overflow = 'hidden';

    // 开始加载图片
    loadImageWithProgress(src);
};

// 隐藏模态框
window.hideModal = (event) => {
    if (event === undefined || event.target.id === 'image-modal' || event.target.id === 'modal-image') {
        modal.classList.remove('is-visible');
        modalImage.src = '';
        document.body.style.overflow = '';
        resetModalState();
        resetImage(); // 重置变换
        
        // 取消正在进行的请求
        if (xhr) {
            xhr.abort();
            xhr = null;
        }
    }
};

// 使用XMLHttpRequest加载图片并显示真实进度
function loadImageWithProgress(imageSrc) {
    // 显示占位符和进度条
    imagePlaceholder.style.display = 'flex';
    imagePlaceholder.style.opacity = '1';
    modalLoadingProgress.style.display = 'block';
    modalLoadingProgress.style.opacity = '1';

    // 隐藏图片
    modalImage.classList.remove('is-loaded');
    modalImage.style.opacity = '0';

    // 重置占位符
    placeholderIcon.style.opacity = '0.3';
    placeholderIcon.style.filter = 'grayscale(1)';

    // 重置进度
    modalImageLoadProgress = 0;
    modalProgressBar.style.width = '0%';
    modalProgressText.textContent = '0%';
    isImageLoaded = false;

    // 清除之前的定时器
    if (modalProgressInterval) clearInterval(modalProgressInterval);

    // 创建XMLHttpRequest对象
    xhr = new XMLHttpRequest();
    
    // 配置请求
    xhr.open('GET', imageSrc, true);
    xhr.responseType = 'blob';
    
    // 监听进度事件
    xhr.addEventListener('progress', function(event) {
        if (event.lengthComputable && event.total > 0) {
            // 计算真实进度百分比
            modalImageLoadProgress = Math.round((event.loaded / event.total) * 100);
            
            // 限制进度最大值，为最终onload事件留出空间
            if (modalImageLoadProgress > 95) {
                modalImageLoadProgress = 95;
            }
            
            // 更新进度条
            modalProgressBar.style.width = modalImageLoadProgress + '%';
            modalProgressText.textContent = modalImageLoadProgress + '%';

            // 更新占位符清晰度
            const clarity = modalImageLoadProgress / 100;
            placeholderIcon.style.opacity = 0.3 + (clarity * 0.7); // 0.3 -> 1.0
            placeholderIcon.style.filter = `grayscale(${1 - clarity})`;

            // 更新占位符文本
            placeholderText.textContent = `加载中... ${modalImageLoadProgress}%`;
        } else {
            // 如果无法计算进度，使用保守的模拟进度
            if (modalImageLoadProgress < 50) {
                modalImageLoadProgress += 5;
                modalProgressBar.style.width = modalImageLoadProgress + '%';
                modalProgressText.textContent = modalImageLoadProgress + '%';
            }
        }
    });
    
    // 请求完成事件
    xhr.addEventListener('load', function() {
        if (xhr.status === 200) {
            // 获取图片数据
            const blob = xhr.response;
            const objectURL = URL.createObjectURL(blob);
            
            // 完成进度到100%
            modalImageLoadProgress = 100;
            modalProgressBar.style.width = '100%';
            modalProgressText.textContent = '100%';
            
            // 最终更新占位符
            placeholderIcon.style.opacity = '1';
            placeholderIcon.style.filter = 'grayscale(0)';
            placeholderText.textContent = '加载完成';

            // 设置实际图片
            modalImage.onload = function() {
                // 释放blob URL内存
                URL.revokeObjectURL(objectURL);
                isImageLoaded = true;
                
                // 延迟显示图片
                setTimeout(() => {
                    // 淡出占位符和进度条
                    imagePlaceholder.style.opacity = '0';
                    modalLoadingProgress.style.opacity = '0';

                    // 淡入图片
                    setTimeout(() => {
                        modalImage.classList.add('is-loaded');
                        modalImage.style.opacity = '1';

                        // 隐藏占位符和进度条
                        setTimeout(() => {
                            imagePlaceholder.style.display = 'none';
                            modalLoadingProgress.style.display = 'none';
                        }, 300);
                    }, 200);
                }, 500);
            };
            
            modalImage.src = objectURL;
        } else {
            // 处理加载错误
            handleImageLoadError();
        }
    });
    
    // 请求错误事件
    xhr.addEventListener('error', function() {
        handleImageLoadError();
    });
    
    // 请求中止事件
    xhr.addEventListener('abort', function() {
        // 用户取消了加载，不做错误处理
        xhr = null;
    });
    
    // 发送请求
    xhr.send();
}

// 图片加载错误处理函数
function handleImageLoadError() {
    isImageLoaded = true;
    xhr = null;

    // 错误状态
    modalProgressBar.style.background = '#ff6b6b';
    modalProgressText.textContent = '加载失败';
    modalProgressText.style.color = '#ff6b6b';

    // 更新占位符
    placeholderIcon.innerHTML = `
        <svg viewBox="0 0 24 24">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
        </svg>
    `;
    placeholderIcon.style.fill = '#ff6b6b';
    placeholderIcon.style.opacity = '1';
    placeholderIcon.style.filter = 'none';
    placeholderText.textContent = '图片加载失败';
    placeholderText.style.color = '#ff6b6b';

    // 延迟隐藏
    setTimeout(() => {
        imagePlaceholder.style.opacity = '0';
        modalLoadingProgress.style.opacity = '0';

        setTimeout(() => {
            imagePlaceholder.style.display = 'none';
            modalLoadingProgress.style.display = 'none';
        }, 300);
    }, 2000);
}

// 重置模态框状态
function resetModalState() {
    // 重置图片
    modalImage.classList.remove('is-loaded');
    modalImage.style.opacity = '0';

    // 重置占位符
    imagePlaceholder.style.display = 'none';
    imagePlaceholder.style.opacity = '1';
    placeholderIcon.innerHTML = `
        <svg viewBox="0 0 24 24">
            <path d="M21 19V5c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2zM8.5 13.5l2.5 3.01L14.5 12l4.5 6H5l3.5-4.5z"/>
        </svg>
    `;
    placeholderIcon.style.opacity = '0.3';
    placeholderIcon.style.filter = 'grayscale(1)';
    placeholderIcon.style.fill = 'white';
    placeholderText.textContent = '图片加载中...';
    placeholderText.style.color = 'white';

    // 重置进度条
    modalLoadingProgress.style.display = 'none';
    modalLoadingProgress.style.opacity = '1';
    modalProgressBar.style.width = '0%';
    modalProgressBar.style.background = 'linear-gradient(90deg, #667eea 0%, #764ba2 50%, #667eea 100%)';
    modalProgressBar.style.backgroundSize = '200% 100%';
    modalProgressText.textContent = '0%';
    modalProgressText.style.color = 'white';

    // 重置状态
    isImageLoaded = false;

    // 清除定时器
    if (modalProgressInterval) {
        clearInterval(modalProgressInterval);
        modalProgressInterval = null;
    }
    
    // 取消请求
    if (xhr) {
        xhr.abort();
        xhr = null;
    }
}

// 键盘事件
document.addEventListener('keydown', function (event) {
    if (modal.classList.contains('is-visible') && event.key === 'Escape') {
        hideModal(event);
    }
});

// 初始化
document.addEventListener('DOMContentLoaded', function () {
    // 初始化模态框状态
    resetModalState();

    // 为图片添加加载完成检测
    modalImage.addEventListener('load', function () {
        if (!isImageLoaded) {
            isImageLoaded = true;
            
            // 确保进度条和占位符正确更新
            modalImageLoadProgress = 100;
            if (modalProgressBar) {
                modalProgressBar.style.width = '100%';
            }
            if (modalProgressText) {
                modalProgressText.textContent = '100%';
            }

            // 更新占位符
            if (placeholderIcon) {
                placeholderIcon.style.opacity = '1';
                placeholderIcon.style.filter = 'grayscale(0)';
            }
            if (placeholderText) {
                placeholderText.textContent = '加载完成';
            }
        }
    });
});