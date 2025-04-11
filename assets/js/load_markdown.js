var blogTitle = "";
var blogContent = "";
var initialLoad = true;

const pathList = ['一阶倒立摆的系统模型', '对一阶倒立摆的LQR控制', 'RL_Multi-Armed_Bandit'];
const nameList = ['一阶倒立摆的系统模型', '对一阶倒立摆的LQR控制', 'Multi-Armed Bandit Problem (RL)'];

function loadMd(name, id = null) {
    if (id) {
        path = pathList[id];
        name = nameList[id];
    }

    blogTitle = name;
    var file = "./blogs/" + path + "/" + path + ".md";

    var blogContent = document.getElementById('blogContent');
    if (blogContent == null) {
        setTimeout('loadMd(blogTitle)', 50);
    } else {
        fetch(file)
            .then((res) => res.arrayBuffer())
            .then((fileContent) => {
                const decoder = new TextDecoder('UTF-8');
                const text = decoder.decode(fileContent);

                blogContent.innerHTML = marked.parse(text);

                // Generate catalogue in the container
                generateTOC();
                setupTOCScrollBehavior();

                document.getElementById('blogTitle').innerHTML = name;

                if (initialLoad) {
                    $("#blogContent").append('<script id="MathJax-script" async src="./assets/js/tex-mml-chtml.js"></script>');
                    initialLoad = false;
                } else {
                    MathJax.typeset();
                }
            })
            .catch((error) => {
                console.log(error);
            })
    }
}

// Catalogue generation functions
function generateTOC() {
    const headings = document.querySelectorAll('#blogContent h2, #blogContent h3');
    const tocContainer = document.querySelector('#catalogue .toc-links');

    if (headings.length > 1) {
        let tocHTML = '<ul>';

        headings.forEach((heading, index) => {
            if (!heading.id) {
                heading.id = `blog-${index}`;
            }
            const indent = heading.tagName === 'H3' ? ' style="margin-left: 15px;"' : '';
            tocHTML += `<li${indent}><a href="#${heading.id}">${heading.textContent}</a></li>`;
        });

        tocHTML += '</ul>';
        tocContainer.innerHTML = tocHTML;
        document.getElementById('catalogue').style.display = 'block';
    } else {
        document.getElementById('catalogue').style.display = 'none';
    }
}

function setupTOCScrollBehavior() {
    // Smooth scrolling for TOC links
    document.addEventListener('click', (e) => {
        if (e.target.closest('.toc-links a')) {
            e.preventDefault();
            const targetId = e.target.getAttribute('href');
            const targetElement = document.querySelector(targetId);

            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 80,
                    behavior: 'smooth'
                });

                // history.pushState(null, null, targetId);
            }
        } else if (e.target.closest('.shortcuts a')) {
            const target = e.target.getAttribute('href');

            if (target == "_") {
                e.preventDefault();
                window.scrollTo({
                    top: 0,
                    behavior: 'smooth'
                });
            }
        }
    });

    // Highlight current section
    window.addEventListener('scroll', highlightCurrentSection);
}

function highlightCurrentSection() {
    const tocLinks = document.querySelectorAll('#catalogue a');
    const scrollPosition = window.scrollY;

    // Reset all active states
    tocLinks.forEach(link => link.classList.remove('active'));

    // Find current section
    document.querySelectorAll('#blogContent h2, #blogContent h3').forEach(heading => {
        const headingTop = heading.offsetTop;
        const headingHeight = heading.offsetHeight;

        if (scrollPosition >= (headingTop - 100) &&
            scrollPosition < (headingTop + headingHeight)) {
            const correspondingLink = document.querySelector(`#catalogue a[href="#${heading.id}"]`);
            if (correspondingLink) {
                correspondingLink.classList.add('active');
            }
        }
    });
}