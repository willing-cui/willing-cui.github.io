var str = 'Do something great!';
var i = 0;
var timer = undefined;

function typing() {
    var typer = document.getElementById('typewriter');
    if (i <= str.length) {
        typer.innerHTML = str.slice(0, i++) + '_';
        timer = setTimeout('typing()', 250);    // 递归调用
    }
    else {
        if (i % 2 == 0) {
            typer.innerHTML = str;              // 结束打字, 移除 _ 光标
        } else {
            typer.innerHTML = str + '_';
        }
        i++;
        timer = setTimeout('typing()', 500);    // 递归调用
    }
}

function startTyping() {
    clearTimeout(timer);
    i = 0; typing();
}

const workButton = document.getElementById("workButton");
workButton.addEventListener("click", () => { startTyping(); });
window.addEventListener("load", () => { startTyping(); });
