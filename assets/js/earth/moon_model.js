import * as THREE from "three";
import { updateLoadingProgress } from './earth_progress_bar.js';


const moonLoadingManager = new THREE.LoadingManager();
moonLoadingManager.onProgress = function (url, itemsLoaded, itemsTotal) {
    console.log('Loading file: ' + url + '.\nLoaded ' + itemsLoaded + ' of ' + itemsTotal + ' items.');
    // You can update a progress bar here, e.g.:
    // updateProgressBar(progress);
    updateLoadingProgress(null, "Loading Texture", 10)
};

moonLoadingManager.onLoad = function () {
    console.log('Moon model texture loading complete!');
};

moonLoadingManager.onError = function (url) {
    console.log('There was an error loading ' + url);
};

// 月球模型
const moonModel = () => {
    // 月球
    const moonGeometry = new THREE.SphereGeometry(0.27, 64, 64);
    // 纹理加载器
    const moonTextureLoader = new THREE.TextureLoader(moonLoadingManager).setPath("../images/earth/");
    // .load()方法加载图像，返回一个纹理对象Texture
    const moonTexture = moonTextureLoader.load("moon.webp");
    // 注意最新版本，webgl渲染器默认编码方式已经改变，为了避免色差，纹理对象编码方式要修改为THREE.SRGBColorSpace
    moonTexture.colorSpace = THREE.SRGBColorSpace; //设置为SRGB颜色空间
    moonTexture.wrapS = THREE.RepeatWrapping;

    const moonMaterial = new THREE.MeshLambertMaterial({
        map: moonTexture,
    });
    // 网格模型
    const moonMesh = new THREE.Mesh(moonGeometry, moonMaterial);

    return moonMesh;
};

export { moonModel };