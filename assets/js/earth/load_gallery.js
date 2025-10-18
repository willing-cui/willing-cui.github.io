// Define global functions for click handlers in HTML, placing them outside the DOMContentLoaded scope
// so they can be called directly from inline 'onclick' attributes.

const modal = document.getElementById('image-modal');
const modalImage = document.getElementById('modal-image');
const basePath = "../images/gallery/";
var photoListDict = {};

// 页面加载时获取文件列表
window.onload = async function () {
    try {
        const response = await fetch(basePath + 'gallery.json');
        photoListDict = await response.json();
    } catch (error) {
        console.error('加载文件列表失败:', error);
    }
};

window.showModal = (src) => {
    modalImage.src = src;
    modal.classList.add('is-visible');
    document.body.style.overflow = 'hidden'; // Prevent scrolling background
};

window.hideModal = (event) => {
    // Check if the click occurred on the modal backdrop or the enlarged image itself
    if (event === undefined || event.target.id === 'image-modal' || event.target.id === 'modal-image') {
        modal.classList.remove('is-visible');
        modalImage.src = ''; // Clear source to free memory
        document.body.style.overflow = ''; // Restore scrolling
    }
};

const grid = document.getElementById('waterfall-grid');

// Function to Create a Single Image Card
function createImageCard(dataUrl, name, time) {
    const card = document.createElement('div');
    card.className = 'image-card';
    card.innerHTML = `
					<div class="image-card-content">
						<img 
							src="${dataUrl}" 
							alt="Local image: ${name}" 
							class="image-display"
							onclick="showModal('${dataUrl}')"
						>
					</div>
					<div class="card-meta-container">
						<span>${name.split(".")[0]}</span>
						<span>${time}</span>
					</div>
				`;
    return card;
};

function loadPhotos(landmarkName) {
    // Clear previous images from the grid
    grid.innerHTML = '';

    var directories = photoListDict["photo"];
    var photo_file_names = directories[landmarkName];
    var photo_time_info = photoListDict["time"][landmarkName];

    if (photo_file_names.length === 0) return;

    // Process each file selected by the user
    var ind = 0;
    Array.from(photo_file_names).forEach(file_name => {

        var filePath = basePath + landmarkName + '/' + file_name;
        var time = photo_time_info[ind];
        ind = ind + 1;

        const card = createImageCard(filePath, file_name, time);
        grid.appendChild(card);
    });
};

export { loadPhotos }