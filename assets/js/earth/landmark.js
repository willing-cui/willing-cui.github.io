import * as THREE from "three";

var landMarkClicked = false
var dom = document.getElementById("dom")
var areas = [
    {
        name: "Shenzhen",
        position: [114.085947, 22.547],
        text_position: [115.585947, 21.247]
    }, {
        name: "Dali",
        position: [100.2676, 25.6065],
        text_position: [97.9676, 25.6065]
    }, {
        name: "Lijiang",
        position: [100.2271, 26.8565],
        text_position: [100.2271, 27.3565]
    }, {
        name: "Zhongshan",
        position: [113.393, 22.516],
        text_position: [110.893, 21.416]
    }, {
        name: "HKSAR",
        position: [114.1694, 22.3193],
        text_position: [115.1694, 20.0193]
    }, {
        name: "Dongguan",
        position: [113.7518, 23.0207],
        text_position: [114.7518, 23.6207]
    }, {
        name: "Zhangzhou",
        position: [117.6472, 24.5135],
        text_position: [117.6472, 25.0135]
    }, {
        name: "Huizhou",
        position: [114.4155, 23.1125],
        text_position: [116.4155, 22.3125]
    }, {
        name: "Shanghai",
        position: [121.4737, 31.2304],
        text_position: [121.4737, 31.7304]
    }, {
        name: "Kunming",
        position: [102.8332, 24.8797],
        text_position: [102.8332, 25.3797]
    }, {
        name: "Shouguang",
        position: [118.7910, 36.8554],
        text_position: [118.7910, 37.3554]
    }, {
        name: "Wuhan",
        position: [114.3052, 30.5928],
        text_position: [114.3052, 31.0928]
    }, {
        name: "Qingyuan",
        position: [113.0561, 23.6820],
        text_position: [110.5561, 22.6820]
    }
];

// 存储当前选中的地标位置和相机动画状态
var currentSelectedLandmark = null;
var currentSelectedLandmarkName = null;
var cameraPointer = null;
var isCameraAnimating = false;
var cameraAnimation = {
    startPosition: new THREE.Vector3(),
    targetPosition: new THREE.Vector3(),
    startQuaternion: new THREE.Quaternion(),
    targetQuaternion: new THREE.Quaternion(),
    startFov: 25,
    targetFov: 25,
    progress: 0,
    duration: 2.0, // 稍微延长动画时间让过渡更平滑
    easing: function (t) {
        // 使用更平滑的缓动函数
        // 三次贝塞尔曲线：缓入缓出
        return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    }
};

// 默认相机位置
const DEFAULT_CAMERA_POSITION = new THREE.Vector3(8, 3.5, 5);
const DEFAULT_CAMERA_FOV = 25;

// 创建地标
function createLandmark(group) {
    // 循环创建地标，文字标签
    for (let i = 0, length = areas.length; i < length; i++) {
        const name = areas[i].name
        const position = createPosition(areas[i].position)
        const hexagon = createHexagon(position); // 地标函数
        const text_position = createPosition(areas[i].text_position)
        const fontMesh = createTxt(text_position, name); // 精灵标签函数
        group.add(hexagon)
        group.add(fontMesh)
    }
}

// 经纬度转坐标
function createPosition(lnglat) {
    let spherical = new THREE.Spherical
    spherical.radius = 1.01;
    const lng = lnglat[0]
    const lat = lnglat[1]
    const theta = (lng + 90) * (Math.PI / 180)
    const phi = (90 - lat) * (Math.PI / 180)
    spherical.phi = phi; // phi是方位面（水平面）内的角度，范围0~360度
    spherical.theta = theta; // theta是俯仰面（竖直面）内的角度，范围0~180度
    let position = new THREE.Vector3()
    position.setFromSpherical(spherical)
    return position
}

// 创建地标标记（使用 BufferGeometry）
function createHexagon(position) {
    var hexagon = new THREE.Object3D();

    // 创建六边形线框和平面（BufferGeometry）
    const hexagonLine = new THREE.BufferGeometry();
    const hexagonPlane = new THREE.BufferGeometry();

    // 生成六边形顶点（使用三角函数计算）
    const segments = 64;
    const radiusLine = 0.006;
    const radiusPlane = 0.003;
    const verticesLine = [];
    const verticesPlane = [];
    const indices = []; // 用于平面三角剖分的索引

    for (let i = 0; i <= segments; i++) {
        const theta = (i / segments) * Math.PI * 2;
        verticesLine.push(Math.cos(theta) * radiusLine, Math.sin(theta) * radiusLine, 0);
        verticesPlane.push(Math.cos(theta) * radiusPlane, Math.sin(theta) * radiusPlane, 0);
    }

    // 添加第一个点作为最后一个点以闭合六边形线框
    verticesLine.push(verticesLine[0], verticesLine[1], verticesLine[2]);

    // 为平面创建三角剖分索引（中心到各顶点）
    const centerIndex = verticesPlane.length / 3;
    verticesPlane.push(0, 0, 0); // 添加中心点

    for (let i = 0; i < segments; i++) {
        indices.push(centerIndex, i, (i + 1) % segments);
    }

    // 设置几何体属性
    hexagonLine.setAttribute('position', new THREE.Float32BufferAttribute(verticesLine, 3));
    hexagonPlane.setAttribute('position', new THREE.Float32BufferAttribute(verticesPlane, 3));
    hexagonPlane.setIndex(indices); // 设置平面索引

    // 创建材质
    let lineMaterial = new THREE.LineBasicMaterial({
        color: 0xffffff,
        linewidth: 2
    });

    let planeMaterial = new THREE.MeshBasicMaterial({
        color: 0xffffff,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.5
    });

    // 创建线框和平面
    let circleLine = new THREE.Line(hexagonLine, lineMaterial);
    let circlePlane = new THREE.Mesh(hexagonPlane, planeMaterial);

    // 设置位置和朝向
    circleLine.position.copy(position);
    circlePlane.position.copy(position);
    circlePlane.lookAt(new THREE.Vector3(0, 0, 0));
    circleLine.lookAt(new THREE.Vector3(0, 0, 0));

    hexagon.add(circleLine);
    hexagon.add(circlePlane);
    return hexagon;
}

// Store glowing state for each text sprite
const textSprites = new Map();

function createTxt(position, name) {
    // Create normal (non-glowing) canvas texture
    const normalCanvas = createTextCanvas(name, false);
    const glowCanvas = createTextCanvas(name, true);

    // Create textures
    const normalTexture = new THREE.CanvasTexture(normalCanvas);
    const glowTexture = new THREE.CanvasTexture(glowCanvas);

    // Create sprite with normal texture
    const material = new THREE.SpriteMaterial({
        map: normalTexture,
        transparent: true,
        alphaTest: 0.1,
        depthTest: true,
        depthWrite: false,
        fog: true
    });

    const sprite = new THREE.Sprite(material);
    sprite.scale.set(0.08, 0.04, 1);
    sprite.position.copy(position).add(
        new THREE.Vector3(
            position.x > 0 ? 0.02 : -0.02,
            position.y > 0 ? 0.02 : -0.02,
            position.z > 0 ? 0.02 : -0.02
        )
    );

    // Store references for toggling
    textSprites.set(sprite, {
        normalTexture,
        name,
        glowTexture,
        material,
        isGlowing: false,
        position: position.clone() // 存储地标位置
    });

    return sprite;
}

// Toggle glow effect on click
function setupTextClickHandler(renderer, camera, scene) {
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    function onClick(event) {
        // Calculate mouse position in normalized device coordinates
        mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
        mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

        // Update the raycaster
        raycaster.setFromCamera(mouse, camera);

        // Calculate objects intersecting the ray
        const intersects = raycaster.intersectObjects(Array.from(textSprites.keys()));

        if (intersects.length > 0) {
            const clickedSprite = intersects[0].object;
            toggleGlow(clickedSprite, camera);
        }
    }

    window.addEventListener('click', onClick, false);

    return {
        dispose: () => {
            window.removeEventListener('click', onClick);
        }
    };
}

// 计算合适的相机位置，确保地球在画面偏左侧
function calculateCameraPosition(landmarkPosition, cameraDistance) {
    // 确保相机距离地标足够远，不会穿过地球
    const minDistanceFromEarth = 2.0;
    const distance = Math.max(cameraDistance, minDistanceFromEarth);

    // 计算从地标指向相机的方向（从地标指向世界原点，然后向外延伸）
    const direction = new THREE.Vector3().subVectors(landmarkPosition, new THREE.Vector3(0, 0, 0)).normalize();

    // 计算基础相机位置（直接在地标后方）
    const basePosition = new THREE.Vector3().copy(landmarkPosition).addScaledVector(direction, distance);

    // 计算从地球中心到地标的方向
    const earthToLandmark = new THREE.Vector3().copy(landmarkPosition).normalize();

    // 创建一个向右的向量（相对于相机视角）
    const upVector = new THREE.Vector3(0, 1, 0);
    const rightVector = new THREE.Vector3().crossVectors(earthToLandmark, upVector).normalize();

    // 计算一个偏移向量，让地球向左移动
    const offsetDistance = 1.2;
    const offsetVector = rightVector.multiplyScalar(offsetDistance);

    // 最终相机位置 = 基础位置 + 偏移向量
    const finalPosition = new THREE.Vector3().copy(basePosition).add(offsetVector);

    return finalPosition;
}

// 计算相机朝向目标点时的四元数
function calculateCameraQuaternion(cameraPosition, targetLookAt) {
    const cameraMatrix = new THREE.Matrix4();
    const targetPosition = targetLookAt.clone();

    // 创建朝向矩阵
    cameraMatrix.lookAt(cameraPosition, targetPosition, new THREE.Vector3(0, 1, 0));

    // 从矩阵中提取四元数
    const quaternion = new THREE.Quaternion();
    quaternion.setFromRotationMatrix(cameraMatrix);

    return quaternion;
}

// 修改startCameraAnimation函数，使用四元数插值
function startCameraAnimation(camera, targetPosition, targetLookAt, isZoomIn) {
    cameraAnimation.startPosition.copy(camera.position);
    cameraAnimation.startFov = camera.fov;
    cameraAnimation.startQuaternion.copy(camera.quaternion);
    cameraAnimation.progress = 0;
    isCameraAnimating = true;

    if (isZoomIn) {
        // 计算目标相机的四元数
        cameraAnimation.targetPosition.copy(targetPosition);
        cameraAnimation.targetQuaternion = calculateCameraQuaternion(targetPosition, targetLookAt);
        cameraAnimation.targetFov = 20;
    } else {
        // 返回默认位置
        cameraAnimation.targetPosition.copy(DEFAULT_CAMERA_POSITION);
        cameraAnimation.targetQuaternion = calculateCameraQuaternion(DEFAULT_CAMERA_POSITION, new THREE.Vector3(0, 0, 0));
        cameraAnimation.targetFov = DEFAULT_CAMERA_FOV;
    }
}

// 更新相机动画（需要在animate循环中调用）
function updateCameraAnimation(camera, deltaTime) {
    if (!isCameraAnimating) return;

    cameraAnimation.progress += deltaTime / cameraAnimation.duration;

    if (cameraAnimation.progress >= 1) {
        // 动画完成
        cameraAnimation.progress = 1;
        isCameraAnimating = false;
    }

    // 应用更平滑的缓动函数
    const easedProgress = cameraAnimation.easing(cameraAnimation.progress);

    // 使用球面线性插值（SLERP）来平滑插值相机位置
    const tempPosition = new THREE.Vector3();
    tempPosition.lerpVectors(cameraAnimation.startPosition, cameraAnimation.targetPosition, easedProgress);
    camera.position.copy(tempPosition);

    // 使用四元数球面线性插值（SLERP）来平滑旋转相机
    const tempQuaternion = new THREE.Quaternion();
    tempQuaternion.slerpQuaternions(cameraAnimation.startQuaternion, cameraAnimation.targetQuaternion, easedProgress);
    camera.quaternion.copy(tempQuaternion);

    // 插值视野
    camera.fov = THREE.MathUtils.lerp(
        cameraAnimation.startFov,
        cameraAnimation.targetFov,
        easedProgress
    );

    camera.updateProjectionMatrix();
}

// 修改toggleGlow函数
function toggleGlow(sprite, camera) {
    const spriteData = textSprites.get(sprite);
    if (!spriteData) return;

    if (isCameraAnimating) return;

    // 获取精灵的实时世界位置
    const worldPosition = new THREE.Vector3();
    sprite.getWorldPosition(worldPosition);

    if (!spriteData.isGlowing) {
        // 点击未选中的地标（选中新地标）
        spriteData.isGlowing = true;
        spriteData.material.map = spriteData.glowTexture;
        spriteData.material.needsUpdate = true;
        sprite.scale.multiplyScalar(1.1);
        landMarkClicked = true;

        // 计算目标相机位置
        const cameraDistance = 3.0;
        const targetCameraPosition = calculateCameraPosition(worldPosition, cameraDistance);

        // 开始相机动画
        startCameraAnimation(camera, targetCameraPosition, worldPosition, true);

        // 取消其他地标的高亮
        textSprites.forEach((data, otherSprite) => {
            if (otherSprite !== sprite && data.isGlowing) {
                data.isGlowing = false;
                data.material.map = data.normalTexture;
                data.material.needsUpdate = true;
                otherSprite.scale.multiplyScalar(1 / 1.1);
            }
        });
        currentSelectedLandmark = sprite;
        currentSelectedLandmarkName = spriteData.name;
        cameraPointer = camera;
    } else {
        // 点击已选中的地标（取消选中）
        spriteData.isGlowing = false;
        spriteData.material.map = spriteData.normalTexture;
        spriteData.material.needsUpdate = true;
        sprite.scale.multiplyScalar(1 / 1.1);
        landMarkClicked = false;

        // 返回默认视角
        startCameraAnimation(camera, DEFAULT_CAMERA_POSITION, new THREE.Vector3(0, 0, 0), false);
        currentSelectedLandmark = null;
        currentSelectedLandmarkName = null;
    }
}

// 在导出之前，同时添加到全局作用域
function resetCurrentSelectedLandmark() {
    const spriteData = textSprites.get(currentSelectedLandmark);
    if (!spriteData || !currentSelectedLandmark) return;

    spriteData.isGlowing = false;
    spriteData.material.map = spriteData.normalTexture;
    spriteData.material.needsUpdate = true;
    currentSelectedLandmark.scale.multiplyScalar(1 / 1.1);
    landMarkClicked = false;

    if (cameraPointer) {
        startCameraAnimation(cameraPointer, DEFAULT_CAMERA_POSITION, new THREE.Vector3(0, 0, 0), false);
    }

    currentSelectedLandmark = null;
    currentSelectedLandmarkName = null;
}

// 添加到全局作用域
window.resetCurrentSelectedLandmark = resetCurrentSelectedLandmark;

function createTextCanvas(text, isGlowing) {
    const w = 600, h = 300;
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = w;
    canvas.height = h;
    ctx.clearRect(0, 0, w, h);

    ctx.font = `${h / 3}px "微软雅黑", sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    if (isGlowing) {
        // Draw glow effect
        ctx.fillStyle = 'rgba(255, 255, 255, 0.1)';
        for (let i = 10; i > 0; i--) {
            ctx.filter = `blur(${i}px)`;
            ctx.fillText(text, w / 2, h / 2);
        }

        ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
        for (let i = 5; i > 0; i--) {
            ctx.filter = `blur(${i}px)`;
            ctx.fillText(text, w / 2, h / 2);
        }
    }

    // Draw solid text
    ctx.filter = 'none';
    ctx.fillStyle = '#ffffff';
    ctx.fillText(text, w / 2, h / 2);

    return canvas;
}

export {
    createLandmark,
    setupTextClickHandler,
    landMarkClicked,
    currentSelectedLandmarkName,
    updateCameraAnimation,
    resetCurrentSelectedLandmark
}