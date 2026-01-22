class BlogIndexGenerator {
    constructor() {
        this.jsonFile = './blogs/blogs.json';
        this.isInitialized = false;
        this.currentSortMode = 'default'; // 'default', 'date-asc', 'date-desc'
    }

    /**
     * 初始化博客生成器
     */
    async init() {
        if (this.isInitialized) return;
        
        try {
            // 检查是否在博客索引页面
            if (!this.isBlogsIndexPage()) return;
            
            const blogData = await this.loadBlogData();
            this.renderBlogs(blogData);
            this.addSortControls();
            this.isInitialized = true;
        } catch (error) {
            console.error('Failed to initialize blog index generator:', error);
            this.showErrorMessage('Failed to load blog data. Please refresh the page and try again.');
        }
    }

    /**
     * 检查当前是否在博客索引页面
     */
    isBlogsIndexPage() {
        const currentHash = window.location.hash;
        return currentHash === '#blogs_index' || 
               document.getElementById('blogsIndex')?.children.length === 0;
    }

    /**
     * 加载JSON数据
     */
    async loadBlogData() {
        try {
            const response = await fetch(this.jsonFile);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error('Failed to load JSON data:', error);
            throw error;
        }
    }

    /**
     * 渲染博客列表
     */
    renderBlogs(blogData) {
        const container = document.getElementById('blogsIndex');
        if (!container) {
            console.error('Cannot found #blogsIndex element.');
            return;
        }

        // 生成博客列表HTML
        const html = this.generateBlogsHTML(blogData);
        container.innerHTML = html;
    }

    /**
     * 添加排序控件
     */
    addSortControls() {
        const container = document.getElementById('blogsIndex');
        if (!container) return;

        // 获取当前排序模式的显示文本
        const getSortLabelText = () => {
            switch (this.currentSortMode) {
                case 'default':
                    return 'Default';
                case 'date-desc':
                    return 'Newest';
                case 'date-asc':
                    return 'Oldest';
                default:
                    return 'Default';
            }
        };

        // 创建排序按钮容器
        const sortControls = document.createElement('div');
        sortControls.className = 'blog-sort-controls';
        sortControls.innerHTML = `
            <span class="sort-label">Sorted by ${getSortLabelText()} </span>
            <div class="sort-buttons">
                <button class="sort-btn ${this.currentSortMode === 'default' ? 'active' : ''}" data-sort="default" title="Default Order">
                    <i class="fa-solid fa-bars"></i>
                </button>
                <button class="sort-btn ${this.currentSortMode === 'date-desc' ? 'active' : ''}" data-sort="date-desc" title="Newest First">
                    <i class="fa-solid fa-arrow-down-wide-short"></i>
                </button>
                <button class="sort-btn ${this.currentSortMode === 'date-asc' ? 'active' : ''}" data-sort="date-asc" title="Oldest First">
                    <i class="fa-solid fa-arrow-down-short-wide"></i>
                </button>
            </div>
        `;

        // 插入到标题下方
        const titleElement = container.querySelector('h2.major');
        if (titleElement) {
            titleElement.parentNode.insertBefore(sortControls, titleElement.nextSibling);
        }

        // 添加点击事件监听器
        sortControls.querySelectorAll('.sort-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.handleSortChange(btn.dataset.sort);
            });
        });
    }

    /**
     * 处理排序方式变化
     */
    async handleSortChange(sortMode) {
        if (this.currentSortMode === sortMode) return;
        
        this.currentSortMode = sortMode;
        
        try {
            const blogData = await this.loadBlogData();
            this.renderBlogs(blogData);
            this.addSortControls(); // 重新添加控件以更新激活状态
        } catch (error) {
            console.error('Failed to reload blog data after sort change:', error);
        }
    }

    /**
     * 生成博客列表HTML
     */
    generateBlogsHTML(blogData) {
        const categorizedBlogs = this.categorizeBlogs(blogData.blogs);
        
        let html = '<h2 class="major">Blogs</h2>\n';
        html += '<div class="blog-ol-wrapper">\n';

        // 按固定顺序显示分类
        const categoryOrder = [
            'AI & RL',
            'Embodied Intelligence', 
            'Wireless Communication',
            'Control Theory',
            'Other Notes'
        ];

        categoryOrder.forEach(category => {
            if (categorizedBlogs[category]) {
                html += this.generateCategoryHTML(category, categorizedBlogs[category]);
            }
        });

        html += '</div>';
        return html;
    }

    /**
     * 按分类分组博客
     */
    categorizeBlogs(blogs) {
        const categories = {};
        
        blogs.forEach(blog => {
            if (!categories[blog.category]) {
                categories[blog.category] = [];
            }
            categories[blog.category].push(blog);
        });

        return categories;
    }

    /**
     * 生成分类区块HTML
     */
    generateCategoryHTML(categoryName, blogs) {
        const iconClass = this.getCategoryIcon(categoryName);
        const blogCount = blogs.length; // 获取该分类下的文章数量
        
        let html = `
            <div class="category-header">
                <h3>
                    <i class="${iconClass}"></i>
                    ${categoryName}
                </h3>
                <span class="blog-count-badge">${blogCount}</span>
            </div>
            <ol class="blog-ol" start="1">
        `;

        // 根据当前排序模式对博客进行排序
        const sortedBlogs = this.sortBlogs(blogs, this.currentSortMode);

        // 添加博客项
        sortedBlogs.forEach(blog => {
            html += this.generateBlogItemHTML(blog);
        });

        html += '</ol>\n';
        return html;
    }

    /**
     * 根据排序模式对博客排序
     */
    sortBlogs(blogs, sortMode) {
        const blogsCopy = [...blogs]; // 创建副本避免修改原数组
        
        switch (sortMode) {
            case 'date-desc':
                // 时间倒序（最新的在前）
                return blogsCopy.sort((a, b) => new Date(b.date) - new Date(a.date));
            
            case 'date-asc':
                // 时间正序（最旧的在前）
                return blogsCopy.sort((a, b) => new Date(a.date) - new Date(b.date));
            
            case 'default':
            default:
                // 默认排序（保持JSON文件中的原始顺序）
                return blogsCopy;
        }
    }

    /**
     * 获取分类图标类名
     */
    getCategoryIcon(categoryName) {
        const iconMap = {
            'AI & RL': 'fa-solid fa-microchip',
            'Embodied Intelligence': 'fa-solid fa-robot',
            'Wireless Communication': 'fa-solid fa-wifi',
            'Control Theory': 'fa-solid fa-gear',
            'Other Notes': 'fa-solid fa-sheet-plastic'
        };
        
        return iconMap[categoryName] || 'fa-solid fa-file';
    }

    /**
     * 生成博客列表项HTML
     */
    generateBlogItemHTML(blog) {
        const imageHTML = blog.image ? 
            `<img loading="lazy" src="${blog.image}" alt="${blog.title}" />` : '';
        
        const titleClass = blog.image ? '' : 'title';
        
        return `
            <li class="blog-li">
                <a href="index.html?part=blogs&id=${blog.id}">
                    <div class="blog-card">
                        ${imageHTML}
                        <span class="${titleClass}">${blog.title}</span>
                        <span>${blog.date}</span>
                    </div>
                </a>
            </li>
        `;
    }

    /**
     * 显示错误信息
     */
    showErrorMessage(message) {
        const container = document.getElementById('blogsIndex');
        if (container) {
            let html = '<h2 class="major"><i class="fa-solid fa-link-slash"></i> Lost of Signal</h2>\n';
            html += `<div class="error-message">${message}</div>`
            container.innerHTML = html;
        }
    }
}

// 全局博客生成器实例
window.BlogIndexGenerator = new BlogIndexGenerator();

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    // 延迟初始化以确保所有资源加载完成
    setTimeout(() => {
        window.BlogIndexGenerator.init();
    }, 100);
});

// 监听hash变化（页面导航）
window.addEventListener('hashchange', () => {
    setTimeout(() => {
        window.BlogIndexGenerator.init();
    }, 50);
});