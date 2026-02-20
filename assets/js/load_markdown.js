var blogTitle = "";
var blogContent = "";
var initialLoad = true;

// 从JSON文件动态加载的博客数据
var blogData = null;
var pathList = [];
var nameList = [];

/**
 * 从JSON文件加载博客数据
 */
async function loadBlogData(currentLanguage) {
    try {
        const response = await fetch('./blogs/blogs.json');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        blogData = await response.json();
        blogData.blogs.sort((a, b) => a.id - b.id);

        // 生成pathList和nameList
        pathList = [];
        nameList = [];

        if (currentLanguage === 'zh') {
            blogData.blogs.forEach(blog => {
                // 读取路径
                const path = blog.path;
                pathList.push(path);
                nameList.push(blog.title_zh);
            });
        } else if (currentLanguage === 'en') {
            blogData.blogs.forEach(blog => {
                // 读取路径
                const path = blog.path;
                pathList.push(path);
                nameList.push(blog.title_en);
            });
        }

        console.log('成功加载博客数据:', blogData.blogs.length, '篇文章');
        // console.log(pathList)
        // console.log(nameList)

    } catch (error) {
        console.error('加载博客数据失败:', error);
        // 使用默认数据作为后备
        loadFallbackData();
    }
}


function loadMd(name, id = null) {
    if (id) {
        path = pathList[id - 1];
        name = nameList[id - 1];
    } else {
        return;
    }

    blogTitle = name;
    var file = "./blogs/" + path + "/" + path + ".md";

    var blogContent = document.getElementById("blogContent");
    if (blogContent == null) {
        setTimeout("loadMd(blogTitle)", 50);
    } else {
        fetch(file)
            .then((res) => res.arrayBuffer())
            .then((fileContent) => {
                const decoder = new TextDecoder("UTF-8");
                const text = decoder.decode(fileContent);

                // 使用marked解析Markdown
                blogContent.innerHTML = marked.parse(text);

                // 高亮所有代码块
                document.querySelectorAll('pre code').forEach((block) => {
                    hljs.highlightElement(block);
                });

                // Generate catalogue in the container
                generateTOC();
                setupTOCScrollBehavior();

                document.getElementById("blogTitle").innerHTML = name;

                if (initialLoad) {
                    $("#blogContent").append(
                        // 使用本地文件加载时，插件会出现路径问题(就算下载整个项目)，导致无法解析\boldsymbol命令
                        // '<script id="MathJax-script" async src="./assets/js/tex-mml-chtml.js"></script>'
                        '<script src="https://cdn.jsdelivr.net/npm/mathjax@4/tex-mml-chtml.js" defer></script>'
                    );
                    initialLoad = false;
                } else {
                    MathJax.typeset();
                }
            })
            .catch((error) => {
                console.log(error);
            });
    }
}

// Catalogue generation functions
function generateTOC() {
    const headings = document.querySelectorAll(
        "#blogContent h2, #blogContent h3"
    );
    const tocContainer = document.querySelector("#catalogue .toc-links");

    if (headings.length > 1) {
        let tocHTML = "<ul>";

        headings.forEach((heading, index) => {
            if (!heading.id) {
                heading.id = `blog-${index}`;
            }
            const indent =
                heading.tagName === "H3" ? ' style="margin-left: 15px;"' : "";
            tocHTML += `<li${indent}><a href="#${heading.id}">${heading.textContent}</a></li>`;
        });

        tocHTML += "</ul>";
        tocContainer.innerHTML = tocHTML;
        document.getElementById("catalogue").style.display = "block";
    } else {
        document.getElementById("catalogue").style.display = "none";
    }
}

function setupTOCScrollBehavior() {
    // Smooth scrolling for TOC links
    document.addEventListener("click", (e) => {
        if (e.target.closest(".toc-links a")) {
            e.preventDefault();
            const targetId = e.target.getAttribute("href");
            const targetElement = document.querySelector(targetId);

            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 80,
                    behavior: "smooth",
                });

                // history.pushState(null, null, targetId);
            }
        } else if (e.target.closest(".shortcuts a")) {
            const target = e.target.getAttribute("href");

            if (target == "_") {
                e.preventDefault();
                window.scrollTo({
                    top: 0,
                    behavior: "smooth",
                });
            }
        }
    });

    // Highlight current section
    window.addEventListener("scroll", highlightCurrentSection);
}

function highlightCurrentSection() {
    const tocLinks = document.querySelectorAll("#catalogue a");
    const scrollPosition = window.scrollY;

    // Reset all active states
    tocLinks.forEach((link) => link.classList.remove("active"));

    // Find current section
    document
        .querySelectorAll("#blogContent h2, #blogContent h3")
        .forEach((heading) => {
            const headingTop = heading.offsetTop;
            const headingHeight = heading.offsetHeight;

            if (
                scrollPosition >= headingTop - 100 &&
                scrollPosition < headingTop + headingHeight
            ) {
                const correspondingLink = document.querySelector(
                    `#catalogue a[href="#${heading.id}"]`
                );
                if (correspondingLink) {
                    correspondingLink.classList.add("active");
                }
            }
        });
}

/**
 * 获取当前语言
 */
function getCurrentLanguage() {
    const savedLang = localStorage.getItem('preferred-language');
    const browserLang = navigator.language.startsWith('zh') ? 'zh' : 'en';
    return savedLang || browserLang || 'en';
}


// 页面加载时初始化博客数据
document.addEventListener('DOMContentLoaded', () => {

    const currentLanguage = getCurrentLanguage()

    // 加载博客数据
    loadBlogData(currentLanguage).then(() => {
        console.log('博客数据加载完成');

        // 检查URL参数，如果有博客ID则自动加载
        const urlParams = new URLSearchParams(window.location.search);
        const part = urlParams.get('part');
        const id = urlParams.get('id');

        if (part === 'blogs' && id) {
            setTimeout(() => {
                loadMd(null, parseInt(id));
            }, 100);
        }
    }).catch(error => {
        console.error('初始化博客数据失败:', error);
    });
});