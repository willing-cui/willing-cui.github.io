var blogTitle = "";
var blogContent = "";
var initialLoad = true;

const pathList = [
    "1_一阶倒立摆的系统模型",
    "2_对一阶倒立摆的LQR控制",
    "3_RL_Multi-Armed_Bandit",
    "4_WiFi_CSI",
    "5_Sinusoidal_Positional_Encoding",
    "6_ResNet",
    "7_Normalization_in_Deep_Learning",
    "8_Attention_Mechanism",
    "9_Transformer_BERT_GPT",
    "10_Multimodal_Fusion",
    "11_WWH_When_Attention_Mechanism_Fail",
    "12_SenseFi_Paper_Reading_Report",
    "13_Transformer_for_Classification_Tasks",
    "14_Activation_Functions",
    "15_Access_Repo_via_GitHub_SSH",
    "16_Feature_Alignment",
    "17_Autocorrelation_of_CSI",
    "18_Autocorrelation_Cross-correlation_Convolution",
    "19_MVX-Net_Paper_Reading_Report",
    "20_DeepFusion_Paper_Reading_Report",
    "21_Cross-Attention_Mechanism",
    "22_Dropout_Function",
    "23_Two-Layer_MLP_Can_Approximate_Any_Function",
    "24_Multi-Head_Attention_Mechanism",
    "25_Training_Stability",
    "26_RoPE_Positional_Encoding",
    "27_Chinchilla_Scaling_Laws",
	"28_Mixture_of_Experts",
	"29_Robotics_Research_Production_Line",
];
const nameList = [
    "一阶倒立摆的系统模型",
    "对一阶倒立摆的LQR控制",
    "Multi-Armed Bandit Problem (RL)",
    "Introduction to WiFi CSI",
    "Sinusoidal Positional Encoding",
    "Introduction to ResNet",
    "Normalization in Deep Learning",
    "Introduction to Attention Mechanism",
    "Introduction to Transformer, BERT and GPT",
    "Multimodal Fusion",
    "What, Why, and How: When Attention Mechanisms Fail",
    "SenseFi Paper Reading Report",
    "Transformer for Classification Tasks",
    "Activation Functions",
    "Access Repo via GitHub SSH",
    "Feature Alignment",
    "Autocorrelation of CSI",
    "Autocorrelation, Cross-correlation, and Convolution",
    "MVX-Net Paper Reading Report",
    "DeepFusion Paper Reading Report",
    "Introduction to Cross-Attention Mechanism",
    "Dropout Function",
    "Two-Layer MLP Can Approximate Any Function?",
    "Multi-Head Attention Mechanism",
    "Stable Model Training: Reducing Validation Variance",
    "RoPE Positional Encoding",
    "Chinchilla Scaling Laws",
	"Mixture of Experts",
	"Robotics Research Production Line",
];

function loadMd(name, id = null) {
    if (id) {
        path = pathList[id - 1];
        name = nameList[id - 1];
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

                blogContent.innerHTML = marked.parse(text);

                // Generate catalogue in the container
                generateTOC();
                setupTOCScrollBehavior();

                document.getElementById("blogTitle").innerHTML = name;

                if (initialLoad) {
                    $("#blogContent").append(
                        '<script id="MathJax-script" async src="./assets/js/tex-mml-chtml.js"></script>'
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
