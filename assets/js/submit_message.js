// 监听页面加载和语言切换
window.addEventListener('load', delayedInitVisitorInfoSubmit);
window.addEventListener('languageChange', delayedInitVisitorInfoSubmit);

// TODO: Can not read the JSON returned by the Google Apps Script.

var visitorMsgTimer = undefined;

function initVisitorInfoSubmit() {
    clearTimeout(visitorMsgTimer);
    const lang = getCurrentLanguage();
    
    const form = document.getElementById('visitorMsg');
    if (!form) {
        console.log("未找到用户反馈表单");
        return;
    }
    
    // 每次都需要重新绑定事件，因为表单是全新的DOM元素
    form.addEventListener('submit', function (e) {
        e.preventDefault();
        const data = new FormData(form);
        const action = e.target.action;
        
        fetch(action, {
            method: 'POST',
            body: data,
            mode: 'no-cors',
        })
        .then(() => {
            Swal.fire({
                title: lang === 'en' ? "SUCCESS" : "提交成功",
                text: lang === 'en' ? "Your information has been submitted!" : "表单已提交, 感谢来信!",
                icon: "success",
                theme: "dark"
            });
            form.reset();
        })
        .catch(error => {
            console.error('Error:', error);
            Swal.fire({
                title: lang === 'en' ? "ERROR" : "提交失败",
                text: lang === 'en' ? "An error occurred. Please try again." : "请稍后重试。",
                icon: "error",
                theme: "dark"
            });
        });
    });
    
    console.log("用户反馈提交表单初始化成功，当前语言：" + lang);
}

function getCurrentLanguage() {
    const savedLang = localStorage.getItem('preferred-language');
    const browserLang = navigator.language.startsWith('zh') ? 'zh' : 'en';
    return savedLang || browserLang || 'en';
}

function delayedInitVisitorInfoSubmit() {
    clearTimeout(visitorMsgTimer);
    
    if (document.getElementById('visitorMsg')) {
        initVisitorInfoSubmit();
    } else {
        // 如果新HTML还没完全加载，等待一下
        visitorMsgTimer = setTimeout(delayedInitVisitorInfoSubmit, 200);
    }
}