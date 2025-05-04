import * as THREE from "three";

var dom = document.getElementById("dom")
var areas = [
    {
        name: "Shenzhen",
        position: [114.085947, 22.547],
        text_position: [114.085947, 22.547]
    }, {
        name: "Dali",
        position: [100.2676, 25.6065],
        text_position: [100.2676, 25.6065]
    }, {
        name: "Lijiang",
        position: [100.2271, 26.8565],
        text_position: [100.2271, 26.8565]
    }
];


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
        glowTexture,
        material,
        isGlowing: false
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
            toggleGlow(clickedSprite);
        }
    }

    window.addEventListener('click', onClick, false);

    return {
        dispose: () => {
            window.removeEventListener('click', onClick);
        }
    };
}

function toggleGlow(sprite) {
    const spriteData = textSprites.get(sprite);
    if (!spriteData) return;

    spriteData.isGlowing = !spriteData.isGlowing;
    spriteData.material.map = spriteData.isGlowing ? spriteData.glowTexture : spriteData.normalTexture;
    spriteData.material.needsUpdate = true;

    if (spriteData.isGlowing) {
        sprite.scale.multiplyScalar(1.1); // Slightly enlarge when glowing
    } else {
        sprite.scale.multiplyScalar(1 / 1.1);
    }
}

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

// Usage example:
// const text1 = createTxt(new THREE.Vector3(0, 0, 0), "Hello");
// scene.add(text1);
// const clickHandler = setupTextClickHandler(renderer, camera, scene);

export { createLandmark, setupTextClickHandler }