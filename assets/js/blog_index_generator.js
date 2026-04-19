class BlogIndexGenerator {
    constructor() {
        // 使用统一的JSON文件
        this.jsonFile = './blogs/blogs.json';
        this.isInitialized = false;
        this.currentSortMode = 'default'; // 'default', 'date-asc', 'date-desc'
        this.currentLanguage = 'en'; // 默认语言
        this.expandedCategories = new Set(); // 存储已展开的分类
        this.MAX_INITIAL_BLOGS = 10; // 默认显示的最大博文数量
    }

    /**
     * 初始化博客生成器
     */
    async init() {
        if (this.isInitialized) return;

        try {
            // 获取当前语言
            this.currentLanguage = this.getCurrentLanguage();

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
     * 获取当前语言
     */
    getCurrentLanguage() {
        const savedLang = localStorage.getItem('preferred-language');
        const browserLang = navigator.language.startsWith('zh') ? 'zh' : 'en';
        return savedLang || browserLang || 'en';
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
    async renderBlogs(blogData) {
        const container = document.getElementById('blogsIndex');
        if (!container) {
            console.error('Cannot found #blogsIndex element.');
            return;
        }

        // 生成博客列表HTML
        const html = this.generateBlogsHTML(blogData);
        container.innerHTML = html;
        
        // 绑定展开/收起按钮事件
        this.bindExpandButtons();
    }

    /**
     * 添加排序控件
     */
    addSortControls() {
        const container = document.getElementById('blogsIndex');
        if (!container) return;

        // 获取当前排序模式的显示文本（支持多语言）
        const getSortLabelText = () => {
            const translations = {
                'en': {
                    'default': 'Default',
                    'date-desc': 'Newest',
                    'date-asc': 'Oldest',
                    'sorted_by': 'Sorted by'
                },
                'zh': {
                    'default': '默认',
                    'date-desc': '最新',
                    'date-asc': '最旧',
                    'sorted_by': '排序方式：'
                }
            };

            const lang = this.currentLanguage;
            const t = translations[lang] || translations['en'];

            return t.sorted_by + ' ' + t[this.currentSortMode];
        };

        // 创建排序按钮容器
        const sortControls = document.createElement('div');
        sortControls.className = 'blog-sort-controls';
        sortControls.innerHTML = `
            <span class="sort-label">${getSortLabelText()}</span>
            <div class="sort-buttons">
                <button class="sort-btn ${this.currentSortMode === 'default' ? 'active' : ''}" data-sort="default" title="${this.currentLanguage === 'zh' ? '默认排序' : 'Default Order'}">
                    <i class="fa-solid fa-bars"></i>
                </button>
                <button class="sort-btn ${this.currentSortMode === 'date-desc' ? 'active' : ''}" data-sort="date-desc" title="${this.currentLanguage === 'zh' ? '最新优先' : 'Newest First'}">
                    <i class="fa-solid fa-arrow-down-wide-short"></i>
                </button>
                <button class="sort-btn ${this.currentSortMode === 'date-asc' ? 'active' : ''}" data-sort="date-asc" title="${this.currentLanguage === 'zh' ? '最旧优先' : 'Oldest First'}">
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
        this.expandedCategories.clear(); // 清空展开状态

        try {
            const blogData = await this.loadBlogData();
            await this.renderBlogs(blogData);
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

        // 根据语言选择标题
        const title = this.currentLanguage === 'zh' ? '博客文章' : 'Blogs';
        let html = `<h2 class="major">${title}</h2>\n`;
        html += '<div class="blog-ol-wrapper">\n';

        // 根据语言选择分类
        const categoryOrder = this.getCategoryOrder();

        categoryOrder.forEach(category => {
            if (categorizedBlogs[category]) {
                html += this.generateCategoryHTML(category, categorizedBlogs[category]);
            }
        });

        html += '</div>';
        return html;
    }

    /**
     * 获取分类顺序和映射
     */
    getCategoryOrder() {
        const categoryMap = {
            'AI & RL': '人工智能与强化学习',
            'Embodied Intelligence': '具身智能',
            'Control Theory': '控制理论',
            'Classical Algorithm': '传统算法',
            'Wireless Communication': '无线通信',
            'Other Notes': '其他笔记'
        };

        if (this.currentLanguage === 'zh') {
            return Object.values(categoryMap);
        } else {
            return Object.keys(categoryMap);
        }
    }

    /**
     * 按分类分组博客
     */
    categorizeBlogs(blogs) {
        const categories = {};

        blogs.forEach(blog => {
            // 根据当前语言选择分类名
            const categoryName = this.getLocalizedCategory(blog.category);

            if (!categories[categoryName]) {
                categories[categoryName] = [];
            }
            categories[categoryName].push(blog);
        });

        return categories;
    }

    /**
     * 获取本地化的分类名
     */
    getLocalizedCategory(originalCategory) {
        const categoryMap = {
            'AI & RL': {
                en: 'AI & RL',
                zh: '人工智能与强化学习'
            },
            'Embodied Intelligence': {
                en: 'Embodied Intelligence',
                zh: '具身智能'
            },
            'Control Theory': {
                en: 'Control Theory',
                zh: '控制理论'
            },
            'Classical Algorithm': {
                en: 'Classical Algorithm',
                zh: '传统算法'
            },
            'Wireless Communication': {
                en: 'Wireless Communication',
                zh: '无线通信'
            },
            'Other Notes': {
                en: 'Other Notes',
                zh: '其他笔记'
            }
        };

        // 如果原始分类不在映射表中，直接返回
        if (!categoryMap[originalCategory]) {
            return originalCategory;
        }

        return categoryMap[originalCategory][this.currentLanguage] ||
            categoryMap[originalCategory]['en'];
    }

    /**
     * 生成分类区块HTML
     */
    generateCategoryHTML(categoryName, blogs) {
        const iconClass = this.getCategoryIcon(categoryName);
        const blogCount = blogs.length;
        const isExpanded = this.expandedCategories.has(categoryName);
        
        // 获取显示文本
        const expandText = this.currentLanguage === 'zh' ? 
            (isExpanded ? '收起' : `展开全部 (${blogCount}篇)`) : 
            (isExpanded ? 'Collapse' : `Expand All (${blogCount} posts)`);
        const expandIcon = isExpanded ? 'fa-chevron-up' : 'fa-chevron-down';

        // 根据排序模式对博客排序
        const sortedBlogs = this.sortBlogs(blogs, this.currentSortMode);
        
        // 确定要显示的博客列表
        const blogsToShow = isExpanded ? 
            sortedBlogs : 
            sortedBlogs.slice(0, this.MAX_INITIAL_BLOGS);
        
        // 是否需要展开按钮
        const hasMore = sortedBlogs.length > this.MAX_INITIAL_BLOGS;
        const displayCount = isExpanded ? blogCount : Math.min(this.MAX_INITIAL_BLOGS, blogCount);

        let html = `
            <div class="category-header">
                <h3>
                    <i class="${iconClass}"></i>
                    ${categoryName}
                    ${hasMore ? `<span class="blog-count-badge">${displayCount}/${blogCount}</span>` : ''}
                </h3>
                ${hasMore ? `
                    <button class="expand-all-btn" data-category="${categoryName}">
                        <span>${expandText}</span>
                        <i class="fas ${expandIcon}"></i>
                    </button>
                ` : ''}
            </div>
            <ol class="blog-ol" start="1">
        `;

        // 添加博客项
        blogsToShow.forEach(blog => {
            html += this.generateBlogItemHTML(blog);
        });

        html += '</ol>\n';
        return html;
    }

    /**
     * 绑定展开/收起按钮事件
     */
    bindExpandButtons() {
        const buttons = document.querySelectorAll('.expand-all-btn');
        buttons.forEach(button => {
            button.addEventListener('click', async (e) => {
                const categoryName = button.dataset.category;
                
                if (this.expandedCategories.has(categoryName)) {
                    this.expandedCategories.delete(categoryName);
                } else {
                    this.expandedCategories.add(categoryName);
                }
                
                // 重新加载并渲染数据
                try {
                    const blogData = await this.loadBlogData();
                    const container = document.getElementById('blogsIndex');
                    if (container) {
                        const html = this.generateBlogsHTML(blogData);
                        container.innerHTML = html;
                        this.bindExpandButtons(); // 重新绑定事件
                        this.addSortControls(); // 重新添加排序控件
                    }
                } catch (error) {
                    console.error('Failed to reload blog data after expand/collapse:', error);
                }
            });
        });
    }

    /**
     * 根据排序模式对博客排序
     */
    sortBlogs(blogs, sortMode) {
        const blogsCopy = [...blogs];

        switch (sortMode) {
            case 'date-desc':
                return blogsCopy.sort((a, b) => new Date(b.date) - new Date(a.date));

            case 'date-asc':
                return blogsCopy.sort((a, b) => new Date(a.date) - new Date(b.date));

            case 'default':
            default:
                return blogsCopy;
        }
    }

    /**
     * 获取分类图标类名
     */
    getCategoryIcon(categoryName) {
        const iconMap = {
            // 英文分类
            'AI & RL': 'fa-solid fa-microchip',
            'Embodied Intelligence': 'fa-solid fa-robot',
            'Control Theory': 'fa-solid fa-gear',
            'Classical Algorithm': 'fa-solid fa-chart-diagram',
            'Wireless Communication': 'fa-solid fa-wifi',
            'Other Notes': 'fa-solid fa-sheet-plastic',

            // 中文分类
            '人工智能与强化学习': 'fa-solid fa-microchip',
            '具身智能': 'fa-solid fa-robot',
            '控制理论': 'fa-solid fa-gear',
            '传统算法': 'fa-solid fa-chart-diagram',
            '无线通信': 'fa-solid fa-wifi',
            '其他笔记': 'fa-solid fa-sheet-plastic'
        };

        return iconMap[categoryName] || 'fa-solid fa-file';
    }

    /**
     * 生成博客列表项HTML
     */
    generateBlogItemHTML(blog) {
        const imageHTML = blog.image ?
            `<img loading="lazy" src="${blog.image}" alt="${this.getBlogTitle(blog)}" />` : '';

        const titleClass = blog.image ? '' : 'title';
        const title = this.getBlogTitle(blog);

        return `
            <li class="blog-li">
                <a href="index.html?part=blogs&id=${blog.id}">
                    <div class="blog-card">
                        ${imageHTML}
                        <span class="${titleClass}">${title}</span>
                        <span>${blog.date}</span>
                    </div>
                </a>
            </li>
        `;
    }

    /**
     * 获取博客标题（根据语言选择）
     */
    getBlogTitle(blog) {
        if (this.currentLanguage === 'zh' && blog.title_zh) {
            return blog.title_zh;
        }
        return blog.title_en;
    }

    /**
     * 显示错误信息
     */
    showErrorMessage(message) {
        const container = document.getElementById('blogsIndex');
        if (container) {
            const errorTitle = this.currentLanguage === 'zh' ?
                '<i class="fa-solid fa-link-slash"></i> 信号丢失' :
                '<i class="fa-solid fa-link-slash"></i> Lost of Signal';

            let html = `<h2 class="major">${errorTitle}</h2>\n`;
            html += `<div class="error-message">${message}</div>`
            container.innerHTML = html;
        }
    }

    /**
     * 重新加载博客数据（用于全局语言切换时调用）
     */
    async reloadForLanguage(lang) {
        this.currentLanguage = lang;
        this.expandedCategories.clear(); // 清空展开状态
        this.isInitialized = false;
        await this.init();
    }
}

// 全局博客生成器实例
window.BlogIndexGenerator = new BlogIndexGenerator();

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
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

// 监听全局语言变化（从主页面切换语言时调用）
window.addEventListener('languageChange', (event) => {
    if (window.BlogIndexGenerator && event.detail && event.detail.language) {
        window.BlogIndexGenerator.reloadForLanguage(event.detail.language);
    }
});

// 在全局作用域添加一个函数，供主语言切换功能调用
if (typeof window !== 'undefined') {
    window.reloadBlogIndexForLanguage = function (lang) {
        if (window.BlogIndexGenerator) {
            window.BlogIndexGenerator.reloadForLanguage(lang);
        }
    };
}