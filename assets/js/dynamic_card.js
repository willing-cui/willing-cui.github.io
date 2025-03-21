var cards = undefined;
var enableCardTimer = undefined;
const skillButton = document.getElementById("skillButton");
skillButton.addEventListener("click", () => { enableCard(); });
window.addEventListener('load', () => { enableCard(); });

function enableDynamicCard() {
    cards.forEach(card => {
        card.onmousemove = (event) => {
            const { pageX, pageY } = event;

            const articleOffsetX = $('#skill').offset().left;
            const articleOffsetY = $('#skill').offset().top;

            const x = pageX - card.offsetLeft - articleOffsetX;
            const y = pageY - card.offsetTop - articleOffsetY;
            
            card.style.setProperty('--x', x + 'px');
            card.style.setProperty('--y', y + 'px');
        }
    })
}

function enableCard() {
    if (enableCardTimer != undefined) {
        clearTimeout(enableCardTimer);
    }
    delayedEnableCard();
}

function delayedEnableCard() {
    cards = document.querySelectorAll('.card');
    if (cards.length != 0) {
        enableDynamicCard();
    } else {
        if (enableCardTimer == undefined) {
            enableCardTimer = setTimeout('delayedEnableCard()', 500);
        }
    }
}