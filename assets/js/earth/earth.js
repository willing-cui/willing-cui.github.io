import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { earthGroup } from "./earth_model.js";
import { sunModel } from "./sun_model.js";
import getStarfield from "./star_field.js";
import {
    setupTextClickHandler,
    landMarkClicked,
    currentSelectedLandmarkName,
    updateCameraAnimation,
    resetCurrentSelectedLandmark
} from "./landmark.js"
import { loadPhotos } from "./load_gallery.js"

// Canvas 容器
const container = document.getElementById("gallery_container");

// 3D场景
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000); // 渲染场景背景

// 太阳
const sun = sunModel();
// 太阳光
scene.add(sun.sunLight);

// 地球
const earth = earthGroup(sun);
scene.add(earth.group);

// 星空
const stars = getStarfield({ numStars: 5000 });
scene.add(stars);

// 相机
const camera = new THREE.PerspectiveCamera(25, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(8, 3.5, 5);

const renderer = new THREE.WebGPURenderer();
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setAnimationLoop(animate);
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
// controls.dampingFactor = 0.01;
controls.minDistance = 0.1;
controls.maxDistance = 50;

setupTextClickHandler(renderer, camera, scene, controls)

let clock = new THREE.Clock();

// 动画函数
function animate() {
    const deltaTime = clock.getDelta();

    if (landMarkClicked == false) {
        earth.earthAutoRotation(); // 地球自转
        hidePhotoList();
    } else {
        raisePhotoList();
    }
    earth.moonRevolution(); // 月球公转和自转
    stars.rotation.y -= 0.0001;

    // 更新相机动画
    updateCameraAnimation(camera, deltaTime);

    renderer.render(scene, camera); //执行渲染操作
}

// onresize 事件会在窗口被调整大小时发生
window.onresize = function () {
    // 重置渲染器输出画布canvas尺寸
    renderer.setSize(window.innerWidth, window.innerHeight);
    // 全屏情况下：设置观察范围长宽比aspect为窗口宽高比
    camera.aspect = window.innerWidth / window.innerHeight;
    // 渲染器执行render方法的时候会读取相机对象的投影矩阵属性projectionMatrix
    // 但是不会每渲染一帧，就通过相机的属性计算投影矩阵(节约计算资源)
    // 如果相机的一些属性发生了变化，需要执行updateProjectionMatrix ()方法更新相机的投影矩阵
    camera.updateProjectionMatrix();
};

// 显示或隐藏图片列表

var photoList = document.getElementById('photo-list');
var photoListRaised = false;
var loadedLandmarkName = null;

function raisePhotoList() {
    if (photoListRaised == false) {
        photoListRaised = true;
        photoList.style.display = 'inline';
        
        setTimeout(() => {
            photoList.style.visibility = 'visible';
            photoList.style.opacity = 100;
        }, 1000)
    }

    if (currentSelectedLandmarkName != loadedLandmarkName) {
        loadedLandmarkName = currentSelectedLandmarkName;
        document.getElementById('landmarkName').innerHTML = currentSelectedLandmarkName;
        loadPhotos(currentSelectedLandmarkName);
    }
}

function hidePhotoList() {
    if (photoListRaised == true) {
        photoListRaised = false;
        photoList.style.visibility = 'hidden';
        photoList.style.opacity = 0;

        setTimeout(() => {
            photoList.style.display = 'none';
        }, 1000)
    }
}

export { earth };