(function ($) {
    var $window = $(window), $body = $('body');
    // Play initial animations on page load.
    $window.on('load', function () {
        window.setTimeout(function () {
            $body.removeClass('is-preload');
        }, 100);
    });
})(jQuery);

// ===========================================
// --- 语言资源文件（用于图表和动态内容） ---
// ===========================================
const dynamicTranslations = {
    en: {
        "last_updated": "Last updated",
        "hot_search": "Hot Search",
        "chart_title": "999.9 Gold Price (HKD/g)",
        "y_axis_label": "Price (HKD/g)",
        "no_data": "No data files found for this period.",
        "failed_load": "Failed to load trending data.",
        "ensure_file": "This feature is currently under maintenance and will be redeployed at a later date."
    },
    zh: {
        "last_updated": "最后更新",
        "hot_search": "热门搜索",
        "chart_title": "999.9 黄金价格 (港元/克)",
        "y_axis_label": "价格 (港元/克)",
        "no_data": "该时间段内未找到数据文件。",
        "failed_load": "加载话题数据失败。",
        "ensure_file": "该功能目前处于维护升级阶段，后续将择机重新上线，敬请期待。"
    }
};

// 获取当前语言 - 使用全局函数
function getCurrentLanguage() {
    // 如果全局已定义，使用全局函数
    if (typeof window.getCurrentLanguage === 'function') {
        return window.getCurrentLanguage();
    }

    // 否则使用本地逻辑
    const savedLang = localStorage.getItem('preferred-language');
    const browserLang = navigator.language.startsWith('zh') ? 'zh' : 'en';
    return savedLang || browserLang || 'en';
}

// 获取翻译文本 - 优先使用全局translations，否则使用dynamicTranslations
function getTranslation(key) {
    const lang = getCurrentLanguage();

    // 优先查找全局translations
    if (window.translations && window.translations[lang] && window.translations[lang][key]) {
        return window.translations[lang][key];
    }

    // 然后查找dynamicTranslations
    if (dynamicTranslations[lang] && dynamicTranslations[lang][key]) {
        return dynamicTranslations[lang][key];
    }

    // 返回key作为备选
    return key;
}

// 更新动态内容语言
function updateDynamicContentLanguage() {
    const lang = getCurrentLanguage();

    // 重新加载数据以更新语言相关的内容
    reloadDataWithCurrentLanguage();
}

// 根据当前语言重新加载数据
function reloadDataWithCurrentLanguage() {
    // 重新加载趋势数据
    if (typeof loadTrendingData === 'function') {
        loadTrendingData();
    }

    // 重新加载金价图表（如果存在）
    if (typeof updateChartLanguage === 'function' && goldPriceChart) {
        updateChartLanguage();
    }
}

// 更新图表语言
function updateChartLanguage() {
    if (!goldPriceChart) return;
    const lang = getCurrentLanguage();

    // 更新图表标题
    if (goldPriceChart.data.datasets[0]) {
        goldPriceChart.data.datasets[0].label = getTranslation('chart_title');
    }

    // 更新Y轴标签
    if (goldPriceChart.options.scales.y && goldPriceChart.options.scales.y.title) {
        goldPriceChart.options.scales.y.title.text = getTranslation('y_axis_label');
    }

    // 更新工具提示
    if (goldPriceChart.options.plugins.tooltip) {
        goldPriceChart.options.plugins.tooltip.callbacks = {
            label: function (context) {
                let label = context.dataset.label || '';
                if (label) {
                    label += ': ';
                }
                if (context.parsed.y !== null) {
                    if (lang === 'zh') {
                        label += context.parsed.y.toFixed(2) + ' 港元';
                    } else {
                        label += context.parsed.y.toFixed(2) + ' HKD';
                    }
                }
                return label;
            }
        };
    }

    goldPriceChart.update();
}

// ===========================================
// --- 本地缓存管理器 ---
// ===========================================

class GoldPriceCacheManager {
    constructor() {
        this.CACHE_KEY = 'gold_price_cache';
        this.CACHE_EXPIRY_DAYS = 365; // 缓存过期天数
        this.maxCacheSize = 365; // 最大缓存天数
    }

    // 获取完整的缓存数据
    getCache() {
        try {
            const cacheStr = localStorage.getItem(this.CACHE_KEY);
            if (!cacheStr) return { data: {}, lastCleanup: null };
            
            const cache = JSON.parse(cacheStr);
            if (!cache.data || !cache.lastCleanup) {
                return { data: {}, lastCleanup: null };
            }
            return cache;
        } catch (error) {
            console.error('读取金价数据缓存时发生错误:', error);
            return { data: {}, lastCleanup: null };
        }
    }

    // 保存缓存数据
    saveCache(cache) {
        try {
            localStorage.setItem(this.CACHE_KEY, JSON.stringify(cache));
        } catch (error) {
            console.error('保存金价数据缓存时发生错误:', error);
        }
    }

    // 清理过期缓存
    cleanupCache() {
        const cache = this.getCache();
        const now = new Date();
        
        // 如果最近清理过（一天内），跳过
        if (cache.lastCleanup) {
            const lastCleanup = new Date(cache.lastCleanup);
            if ((now - lastCleanup) < 24 * 60 * 60 * 1000) {
                return;
            }
        }

        const expiredDate = new Date();
        expiredDate.setDate(expiredDate.getDate() - this.CACHE_EXPIRY_DAYS);
        const expiredDateStr = formatDate(expiredDate);

        // 清理过期数据和限制缓存大小
        const newData = {};
        const dates = Object.keys(cache.data).sort();
        
        // 保留未过期且最新的数据
        dates.slice(-this.maxCacheSize).forEach(date => {
            if (date >= expiredDateStr) {
                newData[date] = cache.data[date];
            }
        });

        cache.data = newData;
        cache.lastCleanup = now.toISOString();
        this.saveCache(cache);
    }

    // 获取缓存中的金价数据
    getCachedPrice(dateStr) {
        const cache = this.getCache();
        return cache.data[dateStr];
    }

    // 批量获取缓存中的金价数据
    getCachedPrices(dateStrings) {
        const cache = this.getCache();
        const result = [];
        
        dateStrings.forEach(dateStr => {
            if (cache.data[dateStr]) {
                result.push({
                    date: dateStr,
                    price: cache.data[dateStr]
                });
            }
        });
        
        return result;
    }

    // 设置缓存数据
    setCachedPrice(dateStr, price) {
        const cache = this.getCache();
        cache.data[dateStr] = price;
        this.saveCache(cache);
    }

    // 批量设置缓存数据
    setCachedPrices(priceData) {
        const cache = this.getCache();
        priceData.forEach(item => {
            if (item.date && item.price) {
                cache.data[item.date] = item.price;
            }
        });
        this.saveCache(cache);
    }

    // 获取需要更新的日期列表
    getDatesToUpdate(dateStrings) {
        const cache = this.getCache();
        return dateStrings.filter(dateStr => !cache.data[dateStr]);
    }

    // 获取已缓存的日期列表
    getCachedDates(dateStrings) {
        const cache = this.getCache();
        return dateStrings.filter(dateStr => cache.data[dateStr]);
    }
}

// 创建缓存管理器实例
const cacheManager = new GoldPriceCacheManager();

// ===========================================
// --- 原始代码部分（修改后） ---
// ===========================================

const notes = document.querySelectorAll('.frosted-note');
const tabs = document.querySelectorAll('.note-tab');
let maxZIndex = 2;

// ===========================================
// --- 1. Utility Functions ---
// ===========================================

function formatDate(date) {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
}

function getDateRangeFilePaths(days) {
    const filePaths = [];
    let date = new Date();
    date.setDate(date.getDate());

    for (let i = 0; i < days; i++) {
        const dateString = formatDate(date);
        filePaths.push(`../scripts/gold_prices/gold_prices_${dateString}.json`);
        date.setDate(date.getDate() - 1);
    }
    return filePaths.reverse();
}

// 获取日期范围
function getDateRange(days) {
    const dates = [];
    let date = new Date();
    date.setDate(date.getDate());

    for (let i = 0; i < days; i++) {
        const dateString = formatDate(date);
        dates.push(dateString);
        date.setDate(date.getDate() - 1);
    }
    return dates.reverse();
}

// ===========================================
// --- 2. Trending Data Logic (Note 1) ---
// ===========================================

function getCurrentDateFilePath() {
    const today = new Date();
    const dateString = formatDate(today);
    return `../scripts/hot_words/all_${dateString}.json`;
}

async function loadTrendingData() {
    const container = document.getElementById('trending-data-container');
    const latestFilePath = getCurrentDateFilePath();

    try {
        const response = await fetch(latestFilePath);
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}. File may not exist for today's date.`);
        }

        const text = await response.text();
        const records = text.trim().split('\n').filter(line => line.length > 0);
        const latestRecord = JSON.parse(records[records.length - 1]);

        let html = `<h4>${getTranslation('last_updated')}: ${new Date(latestRecord.time).toLocaleTimeString()}</h4>`;

        latestRecord.results.forEach(result => {
            if (result.success && result.data && result.data.length > 0) {
                const platformTime = result.timestamp ? new Date(result.timestamp).toLocaleTimeString() : '';
                html += `<div class="trending-platform">`;
                html += `<h4>${result.platform} ${getTranslation('hot_search')} ${platformTime ? `(${platformTime})` : ''}</h4>`;
                html += `<ol>`;
                result.data.slice(0, 5).forEach(item => {
                    const hotValue = item.hot_value ? ` (${item.hot_value})` : '';
                    const keyword = item.keyword || 'N/A';
                    html += `<li><a href="${item.link}" target="_blank">${keyword}${hotValue}</a></li>`;
                });
                html += `</ol></div>`;
            }
        });

        container.innerHTML = html;

    } catch (error) {
        console.error('加载热搜词条时发生错误:', error);
        container.innerHTML = `<p style="color: red;">${getTranslation('failed_load')}</p><p>${getTranslation('ensure_file')}</p>`;
    }
}

// ===========================================
// --- 3. Gold Price Chart Logic (Note 2) ---
// ===========================================

let goldPriceChart = null;

function renderChart(data) {
    const chartArea = document.getElementById('chart-area');
    const debugData = document.getElementById('chart-debug-data');

    chartArea.innerHTML = '';
    debugData.innerHTML = '';

    if (data.length === 0) {
        chartArea.innerHTML = `<p style="text-align: center; margin-top: 50px;">${getTranslation('no_data')}</p>`;
        debugData.innerHTML = '<p>No data found.</p>';
        return;
    }

    const canvas = document.createElement('canvas');
    canvas.id = 'goldPriceCanvas';
    chartArea.appendChild(canvas);

    const labels = data.map(item => item.date);
    const prices = data.map(item => item.price);

    if (goldPriceChart) {
        goldPriceChart.destroy();
    }

    goldPriceChart = new Chart(canvas, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: getTranslation('chart_title'),
                data: prices,
                borderColor: 'rgb(255, 193, 7)',
                backgroundColor: 'rgba(255, 193, 7, 0.2)',
                tension: 0.2,
                fill: true,
                pointRadius: 3,
                pointHoverRadius: 5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    display: true,
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45,
                        callback: function (value, index, values) {
                            if (data.length > 30) {
                                return index % 5 === 0 ? labels[index] : '';
                            }
                            return labels[index];
                        }
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: getTranslation('y_axis_label')
                    },
                    beginAtZero: false
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            const lang = getCurrentLanguage();
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                if (lang === 'zh') {
                                    label += context.parsed.y.toFixed(2) + ' 港元';
                                } else {
                                    label += context.parsed.y.toFixed(2) + ' HKD';
                                }
                            }
                            return label;
                        }
                    }
                }
            }
        }
    });

    const lang = getCurrentLanguage();
    let html = '<ol>';
    data.forEach(item => {
        if (lang === 'zh') {
            html += `<li>${item.date}: ${item.price.toFixed(2)} 港元</li>`;
        } else {
            html += `<li>${item.date}: ${item.price.toFixed(2)} HKD</li>`;
        }
    });
    html += '</ol>';
    debugData.innerHTML = html;
}

// 解析单个文件获取金价
async function parseGoldPriceFile(filePath, dateStr) {
    try {
        const response = await fetch(filePath);
        if (!response.ok) {
            throw new Error(`Status: ${response.status}`);
        }

        const text = await response.text();
        const records = text.trim().split('\n').filter(line => line.length > 0);
        const latestRecord = JSON.parse(records[records.length - 1]);

        const zhubaohuiResult = latestRecord.results.find(r => r.platform === '周大福');

        if (zhubaohuiResult && zhubaohuiResult.success && zhubaohuiResult.data) {
            const sellingPriceItem = zhubaohuiResult.data.find(item => item.type === '999.9饰金卖出价');

            if (sellingPriceItem) {
                const priceString = sellingPriceItem.price_per_gram;
                const price = parseFloat(priceString.split(' ')[0]);

                if (!isNaN(price)) {
                    return { date: dateStr, price: price };
                }
            }
        }
    } catch (error) {
        // 文件不存在或解析失败
        console.warn(`无法加载金价数据 ${dateStr}:`, error.message);
    }
    return null;
}

// 优化后的聚合数据函数
async function fetchAndAggregateData(days) {
    const dates = getDateRange(days);
    const aggregatedData = [];
    
    // 清理过期缓存
    cacheManager.cleanupCache();
    
    // 1. 从缓存获取已有数据
    const cachedData = cacheManager.getCachedPrices(dates);
    aggregatedData.push(...cachedData);
    
    // 2. 检查哪些日期需要更新
    const datesToUpdate = cacheManager.getDatesToUpdate(dates);
    
    if (datesToUpdate.length > 0) {
        console.log(`获取 ${datesToUpdate.length} 个新的金价记录...`);
        
        // 并发获取新数据
        const updatePromises = datesToUpdate.map(async (dateStr) => {
            const filePath = `../scripts/gold_prices/gold_prices_${dateStr}.json`;
            return await parseGoldPriceFile(filePath, dateStr);
        });

        const newData = await Promise.all(updatePromises);
        
        // 3. 缓存新获取的数据
        const validNewData = newData.filter(item => item !== null);
        if (validNewData.length > 0) {
            cacheManager.setCachedPrices(validNewData);
            aggregatedData.push(...validNewData);
        }
    } else {
        console.log('所有金价数据已缓存');
    }
    
    // 排序并渲染
    aggregatedData.sort((a, b) => new Date(a.date) - new Date(b.date));
    renderChart(aggregatedData);
}

function setupGoldPriceTracker() {
    const controls = document.getElementById('chart-controls');
    
    // 初始化时清理缓存
    cacheManager.cleanupCache();
    
    // 首次加载数据
    fetchAndAggregateData(7);

    controls.addEventListener('click', function (event) {
        const button = event.target.closest('button');
        if (button && button.dataset.window) {
            controls.querySelectorAll('button').forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');

            const days = parseInt(button.dataset.window, 10);
            fetchAndAggregateData(days);
        }
    });
}

// ===========================================
// --- 4. Tab Activation Logic ---
// ===========================================

function activateNote(noteId) {
    notes.forEach(n => n.classList.remove('active'));
    tabs.forEach(t => t.classList.remove('active-tab'));

    const targetNote = document.querySelector(`.frosted-note[data-note="${noteId}"]`);
    if (targetNote) {
        maxZIndex++;
        if (maxZIndex > 99) maxZIndex = 100;

        targetNote.style.zIndex = maxZIndex;
        targetNote.classList.add('active');
    }

    const targetTab = document.querySelector(`.note-tab[data-note-id="${noteId}"]`);
    if (targetTab) {
        targetTab.classList.add('active-tab');
    }
}

const initialActiveNote = document.querySelector('.frosted-note.active');
if (initialActiveNote) {
    const initialNoteId = initialActiveNote.dataset.note;
    const initialActiveTab = document.querySelector(`.note-tab[data-note-id="${initialNoteId}"]`);
    if (initialActiveTab) {
        initialActiveTab.classList.add('active-tab');
    }
}

tabs.forEach(tab => {
    tab.addEventListener('click', function (event) {
        event.stopPropagation();
        const noteId = this.dataset.noteId;
        activateNote(noteId);
    });
});

notes.forEach(note => {
    note.addEventListener('click', function () {
        const noteId = this.dataset.note;
        activateNote(noteId);
    });
});

// ===========================================
// --- 5. Initialization ---
// ===========================================

document.addEventListener('DOMContentLoaded', () => {
    // 加载数据
    loadTrendingData();
    setupGoldPriceTracker();

    // 初始化完成后，如果全局有语言切换功能，添加监听器
    if (typeof window.setLanguage === 'function') {
        // 将更新函数暴露给全局，让HTML中的setLanguage可以调用
        window.updateDynamicContentLanguage = updateDynamicContentLanguage;
    }
});

// 添加语言切换监听器
window.addEventListener('storage', function (e) {
    if (e.key === 'preferred-language') {
        // 如果有全局setLanguage函数，调用它
        if (typeof window.setLanguage === 'function') {
            window.setLanguage(e.newValue || 'en');
        } else {
            // 否则直接更新动态内容
            updateDynamicContentLanguage();
        }
    }
});

// 添加缓存清理功能到全局
window.clearGoldPriceCache = function() {
    localStorage.removeItem('gold_price_cache');
    console.log('金价数据缓存已清除');
    return 'Cache cleared successfully.';
};

// 显示缓存状态
window.showCacheStatus = function() {
    const cache = cacheManager.getCache();
    const cacheDates = Object.keys(cache.data);
    return {
        cacheSize: cacheDates.length,
        cachedDates: cacheDates.sort(),
        lastCleanup: cache.lastCleanup
    };
};

// 将函数暴露到全局作用域
window.getTranslation = getTranslation;
window.updateDynamicContentLanguage = updateDynamicContentLanguage;