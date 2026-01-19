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
        text_position: [116.4155, 22.2125]
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

// 定义金色颜色
const GOLD_COLOR = 0xFFD700; // 金色
const WHITE_COLOR = 0xFFFFFF; // 白色

const OFFSET_DISTANCE = 1.2;

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
        color: WHITE_COLOR,
        linewidth: 2
    });

    let planeMaterial = new THREE.MeshBasicMaterial({
        color: WHITE_COLOR,
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
    
    // 存储材质引用，以便后续修改颜色
    hexagon.userData.lineMaterial = lineMaterial;
    hexagon.userData.planeMaterial = planeMaterial;
    
    return hexagon;
}

// Store glowing state for each text sprite
const textSprites = new Map();
// Store landmark hexagons
const landmarks = new Map();

// 为鼠标悬停创建高亮画布
function createHoverTextCanvas(text, isGlowing, isSelected = false) {
    const w = 600, h = 300;
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = w;
    canvas.height = h;
    ctx.clearRect(0, 0, w, h);

    ctx.font = `${h / 3}px "微软雅黑", sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    if (isSelected) {
        // 选中状态：金色高亮
        // Draw gold glow effect
        ctx.fillStyle = 'rgba(255, 215, 0, 0.2)'; // 金色半透明
        for (let i = 12; i > 0; i--) {
            ctx.filter = `blur(${i}px)`;
            ctx.fillText(text, w / 2, h / 2);
        }

        ctx.fillStyle = 'rgba(255, 215, 0, 0.6)'; // 更亮的金色
        for (let i = 6; i > 0; i--) {
            ctx.filter = `blur(${i}px)`;
            ctx.fillText(text, w / 2, h / 2);
        }
    } else if (isGlowing) {
        // 高亮状态
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
    } else {
        // 普通悬停状态
        ctx.fillStyle = 'rgba(255, 255, 255, 0.1)';
        for (let i = 8; i > 0; i--) {
            ctx.filter = `blur(${i}px)`;
            ctx.fillText(text, w / 2, h / 2);
        }
        
        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)'; // 悬停时为白色
        for (let i = 4; i > 0; i--) {
            ctx.filter = `blur(${i}px)`;
            ctx.fillText(text, w / 2, h / 2);
        }
    }

    // Draw solid text
    ctx.filter = 'none';
    if (isSelected) {
        ctx.fillStyle = '#FFD700'; // 金色文字
    } else if (isGlowing) {
        ctx.fillStyle = '#FFFFFF'; // 白色文字
    } else {
        ctx.fillStyle = '#FFFFFF'; // 白色文字
    }
    ctx.fillText(text, w / 2, h / 2);

    return canvas;
}

// 创建悬停纹理
function createHoverTexture(text, isHover, isSelected = false) {
    const canvas = createHoverTextCanvas(text, isHover, isSelected);
    return new THREE.CanvasTexture(canvas);
}

function createTxt(position, name) {
    // 创建三种状态下的纹理
    const normalTexture = createHoverTexture(name, false, false);
    const hoverTexture = createHoverTexture(name, false, false); // 悬停纹理
    const glowTexture = createHoverTexture(name, true, true); // 选中状态使用金色纹理
    const selectedTexture = createHoverTexture(name, true, true); // 新增：专门为选中状态准备的金色纹理

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

    // 存储原始缩放
    sprite.userData.originalScale = sprite.scale.clone();
    
    // Store references for toggling
    textSprites.set(sprite, {
        normalTexture,
        hoverTexture,
        glowTexture,
        selectedTexture, // 新增选中纹理
        name,
        material,
        isGlowing: false,
        isSelected: false, // 新增选中状态
        isHovering: false,
        position: position.clone(),
        originalScale: sprite.scale.clone()
    });

    return sprite;
}

// 修改setupTextClickHandler函数，同时处理点击和悬停事件
function setupTextClickHandler(renderer, camera, scene, controls) {
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    let hoveredSprite = null;
    let timer = null;
    
    // 用于控制光标样式
    const rendererDom = renderer.domElement;
    
    function updateMouse(event) {
        // Calculate mouse position in normalized device coordinates
        mouse.x = (event.clientX / rendererDom.clientWidth) * 2 - 1;
        mouse.y = -(event.clientY / rendererDom.clientHeight) * 2 + 1;
    }
    
    function onMouseMove(event) {
        if (isCameraAnimating) return;
        
        updateMouse(event);
        
        // Update the raycaster
        raycaster.setFromCamera(mouse, camera);

        // Calculate objects intersecting the ray
        const intersects = raycaster.intersectObjects(Array.from(textSprites.keys()));
        
        // 移除之前的悬停效果
        if (hoveredSprite && textSprites.has(hoveredSprite)) {
            const hoveredData = textSprites.get(hoveredSprite);
            if (hoveredData && !hoveredData.isSelected) { // 仅对非选中状态应用悬停效果
                if (hoveredData.isGlowing) {
                    // 如果是高亮状态（非选中），恢复高亮状态
                    hoveredSprite.scale.copy(hoveredData.originalScale).multiplyScalar(1.2);
                    hoveredData.material.map = hoveredData.hoverTexture;
                } else {
                    // 普通状态
                    hoveredSprite.scale.copy(hoveredData.originalScale);
                    hoveredData.material.map = hoveredData.normalTexture;
                    
                    // 解除视角锁定
                    if (timer) clearTimeout(timer);
                    // 设置新的定时器
                    timer = setTimeout(() => {
                        console.log('解除视角锁定');
                        controls.enabled = true;
                    }, 1000);
                }
                hoveredData.material.needsUpdate = true;
                hoveredData.isHovering = false;
            }
        }
        
        // 设置光标样式
        if (intersects.length > 0) {
            // 禁用 controls, 锁定视角
            controls.enabled = false;

            rendererDom.style.cursor = 'pointer';
            hoveredSprite = intersects[0].object;
            const hoveredData = textSprites.get(hoveredSprite);
            
            if (hoveredData && !hoveredData.isSelected) { // 选中状态不应用悬停效果
                // 应用悬停效果
                if (hoveredData.isGlowing) {
                    hoveredSprite.scale.copy(hoveredData.originalScale).multiplyScalar(1.3);
                } else {
                    hoveredSprite.scale.multiplyScalar(1.2); // 放大效果
                }
                hoveredData.material.map = hoveredData.hoverTexture;
                hoveredData.material.needsUpdate = true;
                hoveredData.isHovering = true;
            }
        } else {
            rendererDom.style.cursor = 'auto';
            hoveredSprite = null;
        }
    }
    
    function onMouseLeave() {
        rendererDom.style.cursor = 'auto';
        
        // 移除悬停效果
        if (hoveredSprite && textSprites.has(hoveredSprite)) {
            const hoveredData = textSprites.get(hoveredSprite);
            if (hoveredData && !hoveredData.isSelected && hoveredData.isHovering) {
                if (hoveredData.isGlowing) {
                    hoveredSprite.scale.copy(hoveredData.originalScale).multiplyScalar(1.2);
                    hoveredData.material.map = hoveredData.hoverTexture;
                } else {
                    hoveredSprite.scale.copy(hoveredData.originalScale);
                    hoveredData.material.map = hoveredData.normalTexture;
                }
                hoveredData.material.needsUpdate = true;
                hoveredData.isHovering = false;
            }
        }
        hoveredSprite = null;
    }
    
    function onClick(event) {
        console.log("检测到鼠标点击")
        if (isCameraAnimating) return;
        
        updateMouse(event);
        
        // Update the raycaster
        raycaster.setFromCamera(mouse, camera);

        // Calculate objects intersecting the ray
        const intersects = raycaster.intersectObjects(Array.from(textSprites.keys()), false);

        if (intersects.length > 0) {
            console.log("检测到有效地标点击事件")
            const clickedSprite = intersects[0].object;
            toggleGlow(clickedSprite, camera, scene);
        } else {
            console.log("点击了文本精灵之外的位置");
            // 明确不执行任何操作
            return;
        }
    }
    
    // 添加事件监听器
    rendererDom.addEventListener('mousemove', onMouseMove, false);
    rendererDom.addEventListener('mouseleave', onMouseLeave, false);
    rendererDom.addEventListener('click', onClick, false);

    return {
        dispose: () => {
            rendererDom.removeEventListener('mousemove', onMouseMove);
            rendererDom.removeEventListener('mouseleave', onMouseLeave);
            rendererDom.removeEventListener('click', onClick);
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
    const offsetVector = rightVector.multiplyScalar(OFFSET_DISTANCE);

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

// 更新地标的发光效果
function updateLandmarkColor(landmark, isGold) {
    if (landmark && landmark.userData.lineMaterial && landmark.userData.planeMaterial) {
        if (isGold) {
            // 设为金色
            landmark.userData.lineMaterial.color.setHex(GOLD_COLOR);
            landmark.userData.planeMaterial.color.setHex(GOLD_COLOR);
        } else {
            // 恢复白色
            landmark.userData.lineMaterial.color.setHex(WHITE_COLOR);
            landmark.userData.planeMaterial.color.setHex(WHITE_COLOR);
        }
        
        // 更新材质
        landmark.userData.lineMaterial.needsUpdate = true;
        landmark.userData.planeMaterial.needsUpdate = true;
    }
}

// 修改toggleGlow函数，确保点击时恢复悬停状态
function toggleGlow(sprite, camera, scene) {
    console.log("进入 toggleGlow 函数")
    const spriteData = textSprites.get(sprite);
    if (!spriteData) return;

    if (isCameraAnimating) return;

    // 获取精灵的实时世界位置
    const worldPosition = new THREE.Vector3();
    sprite.getWorldPosition(worldPosition);

    if (!spriteData.isSelected) {
        console.log("点击新地标")
        // 点击未选中的地标（选中新地标）
        spriteData.isGlowing = true;
        spriteData.isSelected = true;
        spriteData.material.map = spriteData.selectedTexture; // 使用金色选中纹理
        spriteData.material.needsUpdate = true;
        sprite.scale.copy(spriteData.originalScale).multiplyScalar(1.3); // 点击时放大更多
        landMarkClicked = true;

        // 计算目标相机位置
        const cameraDistance = 3.0;
        const targetCameraPosition = calculateCameraPosition(worldPosition, cameraDistance);

        // 开始相机动画
        startCameraAnimation(camera, targetCameraPosition, worldPosition, true);

        // 取消其他地标的高亮
        textSprites.forEach((data, otherSprite) => {
            if (otherSprite !== sprite && (data.isGlowing || data.isSelected)) {
                data.isGlowing = false;
                data.isSelected = false;
                data.material.map = data.normalTexture;
                data.material.needsUpdate = true;
                otherSprite.scale.copy(data.originalScale);
                
                // 恢复其他地标的颜色
                updateLandmarkColor(landmarks.get(data.name), false);
            }
        });
        
        // 查找对应的地标并设为金色
        let foundLandmark = null;
        scene.traverse((child) => {
            if (child.userData.isLandmark && child.position.distanceTo(worldPosition) < 0.1) {
                foundLandmark = child;
            }
        });
        
        if (foundLandmark) {
            updateLandmarkColor(foundLandmark, true);
        }
        
        currentSelectedLandmark = sprite;
        currentSelectedLandmarkName = spriteData.name;
        cameraPointer = camera;
    } else {
        // 点击已选中的地标（取消选中）
        spriteData.isGlowing = false;
        spriteData.isSelected = false;
        spriteData.material.map = spriteData.normalTexture;
        spriteData.material.needsUpdate = true;
        sprite.scale.copy(spriteData.originalScale);
        landMarkClicked = false;
        
        // 恢复地标颜色
        let foundLandmark = null;
        scene.traverse((child) => {
            if (child.userData.isLandmark && child.position.distanceTo(worldPosition) < 0.1) {
                foundLandmark = child;
            }
        });
        
        if (foundLandmark) {
            updateLandmarkColor(foundLandmark, false);
        }

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
    spriteData.isSelected = false;
    spriteData.material.map = spriteData.normalTexture;
    spriteData.material.needsUpdate = true;
    currentSelectedLandmark.scale.copy(spriteData.originalScale);
    landMarkClicked = false;
    
    // 恢复地标颜色
    if (cameraPointer && cameraPointer.parent) { // scene is the parent of camera
        const worldPosition = new THREE.Vector3();
        currentSelectedLandmark.getWorldPosition(worldPosition);
        
        let foundLandmark = null;
        cameraPointer.parent.traverse((child) => {
            if (child.userData.isLandmark && child.position.distanceTo(worldPosition) < 0.1) {
                foundLandmark = child;
            }
        });
        
        if (foundLandmark) {
            updateLandmarkColor(foundLandmark, false);
        }
    }

    if (cameraPointer) {
        startCameraAnimation(cameraPointer, DEFAULT_CAMERA_POSITION, new THREE.Vector3(0, 0, 0), false);
    }

    currentSelectedLandmark = null;
    currentSelectedLandmarkName = null;
}

function createLandmark(group) {
    // 循环创建地标，文字标签
    for (let i = 0, length = areas.length; i < length; i++) {
        const name = areas[i].name
        const position = createPosition(areas[i].position)
        const hexagon = createHexagon(position); // 地标函数
        const text_position = createPosition(areas[i].text_position)
        const fontMesh = createTxt(text_position, name); // 精灵标签函数
        
        // 标记为地标
        hexagon.userData.isLandmark = true;
        hexagon.userData.name = name;
        
        // 存储地标引用
        landmarks.set(name, hexagon);
        
        group.add(hexagon)
        group.add(fontMesh)
    }
}

// 添加到全局作用域
window.resetCurrentSelectedLandmark = resetCurrentSelectedLandmark;

export {
    createLandmark,
    setupTextClickHandler,
    landMarkClicked,
    currentSelectedLandmarkName,
    updateCameraAnimation,
    resetCurrentSelectedLandmark
}