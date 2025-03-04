var str = 'Do something great!';
var i = 0;

function typing() {
    var typer = document.getElementById('typewriter');
    if (i <= str.length) {
        typer.innerHTML = str.slice(0, i++) + '_';
        setTimeout('typing()', 200);    // 递归调用
    }
    else {
        if (i % 2 == 0) {
            typer.innerHTML = str;  // 结束打字, 移除 _ 光标
        } else {
            typer.innerHTML = str + '_';
        }
        i++;
        setTimeout('typing()', 500);    // 递归调用
    }
}

typing();