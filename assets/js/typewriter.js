var typerString = 'Do something great!';
var typerStrInd = 0;
var typerTimer = undefined;
const workButton = document.getElementById("workButton");
workButton.addEventListener("click", () => { delayedStartTyping(); });
window.addEventListener('load', () => { delayedStartTyping(); });

function typing() {
    var typer = document.getElementById('typewriter');
    if (typerStrInd <= typerString.length) {
        typer.innerHTML = typerString.slice(0, typerStrInd++) + '_';
        typerTimer = setTimeout('typing()', 250);    // 递归调用
    } else {
        if (typerStrInd % 2 == 0) {
            typer.innerHTML = typerString;              // 结束打字, 移除 _ 光标
        } else {
            typer.innerHTML = typerString + '_';
        }
        typerStrInd++;
        typerTimer = setTimeout('typing()', 500);    // 递归调用
    }
}

function startTyping() {
    if (typerTimer != undefined) {
        clearTimeout(typerTimer);
    }
    typerStrInd = 0; 
    typing();
}

function delayedStartTyping() {
    if (document.getElementById('typewriter')) {
        startTyping();
    } else {
        if (typerTimer == undefined) {
            typerTimer = setTimeout('delayedStartTyping()', 500);
        }
    }
}
