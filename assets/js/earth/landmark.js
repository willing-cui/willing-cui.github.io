import * as THREE from "three";

var landMarkClicked = false
var dom = document.getElementById("dom")
var areas = [
    {
        name: "Shenzhen",
        position: [114.085947, 22.547],
    }, {
        name: "Dali",
        position: [100.2676, 25.6065],
    }, {
        name: "Lijiang",
        position: [100.2271, 26.8565],
    }, {
        name: "Zhongshan",
        position: [113.393, 22.516],
    }, {
        name: "HKSAR",
        position: [114.1694, 22.3193],
    }, {
        name: "Dongguan",
        position: [113.7518, 23.0207],
    }, {
        name: "Zhangzhou",
        position: [117.6472, 24.5135],
    }, {
        name: "Huizhou",
        position: [114.4155, 23.1125],
    }, {
        name: "Shanghai",
        position: [121.4737, 31.2304],
    }, {
        name: "Kunming",
        position: [102.8332, 24.8797],
    }, {
        name: "Shouguang",
        position: [118.7910, 36.8554],
    }, {
        name: "Wuhan",
        position: [114.3052, 30.5928],
    }, {
        name: "Qingyuan",
        position: [113.0561, 23.6820],
    }, {
        name: "Zhucheng",
        position: [119.4098, 35.9958],
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

// 文字的实际长宽比（宽0.08，高0.04，所以高/宽=0.5）
const TEXT_ASPECT_RATIO = 0.5; // 高度是宽度的一半
const TEXT_WIDTH = 0.08;
const TEXT_HEIGHT = TEXT_WIDTH * TEXT_ASPECT_RATIO;

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


// 调整文本位置以避免重叠（优化版 - 保持相对位置关系）
function adjustTextPositions(textPositions) {
    // 创建副本，保持原始位置不变
    const adjustedPositions = textPositions.map(pos => ({
        ...pos,
        position: pos.position.clone(),
        originalPosition: pos.position.clone(),
        isFixed: false,
        overlapCount: 0,
        // 计算原始方位角（相对于地球中心）
        originalAzimuth: Math.atan2(pos.position.y, pos.position.x)
    }));

    const maxIterations = 1000;
    const maxDistanceFromOriginal = 0.15; // 严格控制最大偏移

    // 由于文字是水平排列的，水平方向需要更大间距
    const minDistanceHorizontal = TEXT_WIDTH * 1.5 * 0.7;  // 水平方向最小距离
    const minDistanceVertical = TEXT_HEIGHT * 1.5 * 0.4;   // 垂直方向可以稍小一些

    for (let iteration = 0; iteration < maxIterations; iteration++) {
        let totalOverlap = 0;

        // 第一阶段：检测重叠并计算移动
        for (let i = 0; i < adjustedPositions.length; i++) {
            if (adjustedPositions[i].isFixed) continue;

            const pos1 = adjustedPositions[i];
            let totalForce = new THREE.Vector3(0, 0, 0);
            let overlapCount = 0;

            for (let j = 0; j < adjustedPositions.length; j++) {
                if (i === j || adjustedPositions[j].isFixed) continue;

                const pos2 = adjustedPositions[j];
                const distance = pos1.position.distanceTo(pos2.position);

                // 计算两个位置之间的方向向量
                const direction = new THREE.Vector3()
                    .subVectors(pos1.position, pos2.position)
                    .normalize();

                // 计算在水平方向和垂直方向的分量
                const dotProductHorizontal = Math.abs(direction.dot(new THREE.Vector3(1, 0, 0)));
                const dotProductVertical = Math.abs(direction.dot(new THREE.Vector3(0, 1, 0)));

                // 判断主要重叠方向
                const isHorizontalOverlap = dotProductHorizontal > dotProductVertical;
                const requiredDistance = isHorizontalOverlap ?
                    minDistanceHorizontal : minDistanceVertical;

                if (distance < requiredDistance) {
                    overlapCount++;

                    // 根据原始相对位置调整排斥方向
                    const forceDirection = calculateAdjustedDirection(pos1, pos2, direction);

                    const overlapFactor = (requiredDistance - distance) / requiredDistance;
                    const forceWeight = isHorizontalOverlap ? 1.2 : 0.8; // 水平重叠权重更高
                    const force = forceDirection.multiplyScalar(overlapFactor * forceWeight * 0.005);

                    totalForce.add(force);
                }
            }

            pos1.overlapCount = overlapCount;
            totalOverlap += overlapCount;

            // 应用排斥力
            if (totalForce.length() > 0.001) {
                const newPosition = pos1.position.clone().add(totalForce).normalize().multiplyScalar(1.05);

                // 严格检查偏移范围
                const distanceFromOriginal = newPosition.distanceTo(pos1.originalPosition);
                if (distanceFromOriginal <= maxDistanceFromOriginal) {
                    pos1.position.copy(newPosition);
                } else {
                    // 超出范围，强制向原始位置拉回
                    const pullDirection = new THREE.Vector3()
                        .subVectors(pos1.originalPosition, newPosition)
                        .normalize();
                    const pullDistance = distanceFromOriginal - maxDistanceFromOriginal;
                    pos1.position.add(pullDirection.multiplyScalar(pullDistance)).normalize().multiplyScalar(1.05);
                }
            }
        }

        // 第二阶段：检查是否可以固定位置
        for (let i = 0; i < adjustedPositions.length; i++) {
            if (adjustedPositions[i].isFixed) continue;

            const pos = adjustedPositions[i];
            let canBeFixed = true;

            for (let j = 0; j < adjustedPositions.length; j++) {
                if (i === j) continue;

                const otherPos = adjustedPositions[j];
                const distance = pos.position.distanceTo(otherPos.position);

                // 同样考虑方向性的距离检查
                const direction = new THREE.Vector3()
                    .subVectors(pos.position, otherPos.position)
                    .normalize();

                const dotProductHorizontal = Math.abs(direction.dot(new THREE.Vector3(1, 0, 0)));
                const dotProductVertical = Math.abs(direction.dot(new THREE.Vector3(0, 1, 0)));
                const isHorizontal = dotProductHorizontal > dotProductVertical;
                const requiredDistance = isHorizontal ? minDistanceHorizontal : minDistanceVertical;

                if (distance < requiredDistance * 0.9) {
                    canBeFixed = false;
                    break;
                }
            }

            if (canBeFixed && pos.overlapCount === 0) {
                pos.isFixed = true;
            }
        }

        // 第三阶段：根据原始相对位置进行方位校正
        for (let i = 0; i < adjustedPositions.length; i++) {
            const pos = adjustedPositions[i];

            // 计算当前方位角
            const currentAzimuth = Math.atan2(pos.position.y, pos.position.x);

            // 计算方位角偏差
            let azimuthDiff = currentAzimuth - pos.originalAzimuth;

            // 将偏差标准化到 -π 到 π 范围内
            if (azimuthDiff > Math.PI) azimuthDiff -= 2 * Math.PI;
            if (azimuthDiff < -Math.PI) azimuthDiff += 2 * Math.PI;

            // 如果方位角偏差过大，进行校正
            if (Math.abs(azimuthDiff) > Math.PI / 12) { // 15度阈值
                const correctionStrength = 0.1;
                const correctionAngle = -azimuthDiff * correctionStrength;

                // 创建旋转矩阵来校正方位
                const rotationMatrix = new THREE.Matrix4().makeRotationZ(correctionAngle);
                pos.position.applyMatrix4(rotationMatrix).normalize().multiplyScalar(1.01);
            }
        }

        // 第四阶段：持续向原始位置拉回，避免过度偏移
        for (let i = 0; i < adjustedPositions.length; i++) {
            const pos = adjustedPositions[i];
            const distanceFromOriginal = pos.position.distanceTo(pos.originalPosition);

            // 根据当前偏移程度调整拉回力度
            if (distanceFromOriginal > maxDistanceFromOriginal * 0.5) {
                const pullStrength = Math.min(0.1, (distanceFromOriginal - maxDistanceFromOriginal * 0.5) * 0.3);
                const pullDirection = new THREE.Vector3()
                    .subVectors(pos.originalPosition, pos.position)
                    .normalize();
                pos.position.add(pullDirection.multiplyScalar(pullStrength)).normalize().multiplyScalar(1.01);
            }
        }

        // 检查终止条件
        const fixedCount = adjustedPositions.filter(p => p.isFixed).length;
        if (totalOverlap === 0 || fixedCount === adjustedPositions.length || iteration > 800) {
            break;
        }
    }

    return adjustedPositions;
}

// 辅助函数：根据原始相对位置计算调整后的方向
function calculateAdjustedDirection(pos1, pos2, baseDirection) {
    // 计算原始相对位置
    const originalDiffX = pos1.originalPosition.x - pos2.originalPosition.x;
    const originalDiffY = pos1.originalPosition.y - pos2.originalPosition.y;

    // 计算当前相对位置
    const currentDiffX = pos1.position.x - pos2.position.x;
    const currentDiffY = pos1.position.y - pos2.position.y;

    // 如果原始相对位置与当前相对位置方向基本一致，使用基础方向
    const originalAngle = Math.atan2(originalDiffY, originalDiffX);
    const currentAngle = Math.atan2(currentDiffY, currentDiffX);
    const angleDiff = Math.abs(originalAngle - currentAngle);

    // 如果角度差异在可接受范围内，使用基础方向
    if (angleDiff < Math.PI / 6) { // 30度阈值
        return baseDirection.clone();
    }

    // 否则，根据原始相对位置调整方向
    // 优先保持垂直方向的关系
    if (Math.abs(originalDiffY) > Math.abs(originalDiffX) * 2) {
        // 原始位置主要是垂直关系
        if (originalDiffY > 0) {
            // pos1 在 pos2 上方
            return new THREE.Vector3(0, 1, 0);
        } else {
            // pos1 在 pos2 下方
            return new THREE.Vector3(0, -1, 0);
        }
    } else if (Math.abs(originalDiffX) > Math.abs(originalDiffY) * 2) {
        // 原始位置主要是水平关系
        if (originalDiffX > 0) {
            // pos1 在 pos2 右方
            return new THREE.Vector3(1, 0, 0);
        } else {
            // pos1 在 pos2 左方
            return new THREE.Vector3(-1, 0, 0);
        }
    } else {
        // 对角线关系，保持原始相对方向
        return new THREE.Vector3(
            Math.sign(originalDiffX),
            Math.sign(originalDiffY),
            0
        ).normalize();
    }
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
    sprite.scale.set(TEXT_WIDTH, TEXT_HEIGHT, 1);
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
            // 解除视角锁定
            controls.enabled = true;
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
    // 收集所有文本位置
    const textPositions = [];

    for (let i = 0, length = areas.length; i < length; i++) {
        const name = areas[i].name;
        const position = createPosition(areas[i].position);
        const text_position = createPosition(areas[i].position);

        textPositions.push({
            name: name,
            position: text_position,
            originalPosition: text_position.clone()
        });
    }

    // 调整文本位置以避免重叠
    const adjustedPositions = adjustTextPositions(textPositions);

    // 创建地标和文本
    for (let i = 0, length = areas.length; i < length; i++) {
        const name = areas[i].name;
        const position = createPosition(areas[i].position);
        const hexagon = createHexagon(position);

        // 查找调整后的文本位置
        const adjustedPosition = adjustedPositions.find(p => p.name === name)?.position || createPosition(areas[i].text_position);
        const fontMesh = createTxt(adjustedPosition, name);

        // 标记为地标
        hexagon.userData.isLandmark = true;
        hexagon.userData.name = name;

        // 存储地标引用
        landmarks.set(name, hexagon);

        group.add(hexagon);
        group.add(fontMesh);
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