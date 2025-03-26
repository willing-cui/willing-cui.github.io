var blogTitle = "";
var blogContent = "";
var initialLoad = true;

const nameList = ['一阶倒立摆的系统模型', '对一阶倒立摆的LQR控制'];

function loadMd(name, id = null) {

    if (id) {
        name = nameList[id]
    }

    blogTitle = name;
    var file = "./blogs/" + name + "/" + name + ".md";

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