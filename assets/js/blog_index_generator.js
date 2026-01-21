class BlogIndexGenerator {
    constructor() {
        this.jsonFile = './blogs/blogs.json';
        this.isInitialized = false;
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
        
        let html = `
            <h3>
                <i class="${iconClass}"></i>
                ${categoryName}
            </h3>
            <ol class="blog-ol" start="1">
        `;

        // 按日期排序（最新的在前）
        blogs.sort((a, b) => new Date(b.date) - new Date(a.date));

        // 添加博客项
        blogs.forEach(blog => {
            html += this.generateBlogItemHTML(blog);
        });

        html += '</ol>\n';
        return html;
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