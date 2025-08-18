import * as THREE from "three";
import {
    mix,
    max,
    step,
    vec3,
    vec4,
    texture,
    uv,
    output,
    uniform,
    color,
    normalWorld,
    positionWorld,
    cameraPosition,
    bumpMap,
} from "three/tsl";
import { moonModel } from "./moon_model.js";
import {createLandmark} from "./landmark.js"

const earth_imgs = [
    "earth_january_8k.webp",
    "earth_february_8k.webp",
    "earth_march_8k.webp",
    "earth_april_8k.webp",
    "earth_may_8k.webp",
    "earth_june_8k.webp",
    "earth_july_8k.webp",
    "earth_august_8k.webp",
    "earth_september_8k.webp",
    "earth_october_8k.webp",
    "earth_november_8k.webp",
    "earth_december_8k.webp"
];

const date = new Date();
const month = date.getMonth();

// 大气层白天颜色
const atmosphereDayColor = uniform(color("#4db2ff"));
// 大气层夜晚颜色（暮色）
const atmosphereTwilightColor = uniform(color("#000000"));

const roughnessLow = uniform(0.25);
const roughnessHigh = uniform(0.5);

// 贴图，加载器
const textureLoader = new THREE.TextureLoader().setPath("../images/earth/");

const earthGroup = (sunModel) => {
    const group = new THREE.Group();
    const earthGroup = new THREE.Group();
    const landmarkGroup = new THREE.Group();
    
    // 地球
    const earth = earthModel(sunModel);
    earthGroup.add(earth);

    // 地球大气层
    const atmosphere = atmosphereModel(sunModel, earth);
    earthGroup.add(atmosphere);

    // 云层
    const clouds = cloudsModel(earth);
    earthGroup.add(clouds);

    // 地标
    createLandmark(landmarkGroup)

    earthGroup.add(landmarkGroup);
    group.add(earthGroup);

    // 月亮
    const moon = moonModel();
    // 设置月球的初始位置
    earth.position.copy(moon.position);
    moon.position.x += 2; // 月球距离地球2个单位
    group.add(moon);

    // 地球自转
    const earthAutoroatation = () => {
        earthGroup.rotation.y += 0.0005;
    };

    // 月球公转
    const moonRevolution = () => {
        // 月球绕地球公转
        const time = Date.now() * 0.001; // 获取当前时间（秒）
        const orbitRadius = 6; // 公转半径
        const speed = 0.3; // 公转速度

        // 计算月球的新位置
        moon.position.x = Math.sin(time * speed) * orbitRadius;
        moon.position.z = Math.cos(time * speed) * orbitRadius;

        // 因为潮汐锁定，月球公转周期和自转周期一样。有一面永远对着地球
        moon.lookAt(earth.position);
    };

    return { group, earthGroup, landmarkGroup, earthAutoroatation, moonRevolution };
};

// 地球模型
const earthModel = (sunModel) => {
    // 地球
    const earthGeometry = new THREE.SphereGeometry(1, 64, 64);

    // 地球材质
    // 在Three.js中，PMREM主要用于环境映射照明。使用标准的HDRI纹理时，可能会遇到周围的照明问题，导致阴影完全黑色。而使用PMREMGenerator可以解决这个问题
    // MeshStandardNodeMaterial 属于 PMREM
    const earthMaterial = new THREE.MeshStandardNodeMaterial();

    const bumpRoughnessLands = getBumpRoughnessLands();

    earthMaterial.colorNode = getTextureDay(bumpRoughnessLands.landsStrength); // 白天的贴图
    earthMaterial.outputNode = getTextureNight(sunModel); // 夜晚的贴图
    earthMaterial.roughnessNode = getTextureRoughness(bumpRoughnessLands); // 陆地粗糙度的贴图
    earthMaterial.normalNode = getTextureNormal(bumpRoughnessLands); // 法线的贴图

    // 地球网格模型
    const earthMesh = new THREE.Mesh(earthGeometry, earthMaterial);

    // 设置地球的倾斜角度
    // earthMesh.rotateX(-Math.PI / 7.6);

    return earthMesh;
};

// 云模型
const cloudsModel = (earthModel) => {
    const cloudsMat = new THREE.MeshStandardMaterial({
        map: textureLoader.load("earth_cloud_8k.webp"),
        transparent: true,
        opacity: 0.3,
        blending: THREE.AdditiveBlending,
    });

    const cloudsMesh = new THREE.Mesh(earthModel.geometry, cloudsMat);
    cloudsMesh.scale.setScalar(1.01);

    return cloudsMesh;
}

// 大气层模型
const atmosphereModel = (sunModel, earthModel) => {
    const atmosphereMaterial = new THREE.MeshBasicNodeMaterial({ side: THREE.BackSide, transparent: true });

    const fresnel = getFresnel();
    let alpha = fresnel.remap(0.73, 1, 1, 0).pow(3);
    alpha = alpha.mul(sunModel.sunOrientation.smoothstep(-0.5, 1));

    // 大气层颜色
    const atmosphereColor = getAtmosphereColor(sunModel);
    atmosphereMaterial.outputNode = vec4(atmosphereColor, alpha);

    const atmosphereMesh = new THREE.Mesh(earthModel.geometry, atmosphereMaterial);
    atmosphereMesh.scale.setScalar(1.04);

    return atmosphereMesh;
};

const getFresnel = () => {
    const viewDirection = positionWorld.sub(cameraPosition).normalize();
    const fresnel = viewDirection.dot(normalWorld).abs().oneMinus().toVar();

    return fresnel;
};

// 大气层颜色
const getAtmosphereColor = (sunModel) => {
    // 太阳光的方向
    const sunOrientation = sunModel.sunOrientation;

    // 大气层颜色
    const atmosphereColor = mix(atmosphereTwilightColor, atmosphereDayColor, sunOrientation.smoothstep(-0.25, 0.75));

    return atmosphereColor;
};

const getTextureDayJpg = () => {
    // 贴图, 白天
    const dayTexture = textureLoader.load(earth_imgs[month]);
    dayTexture.colorSpace = THREE.SRGBColorSpace; // 设置为SRGB颜色空间
    dayTexture.anisotropy = 8; // 数值越大Map越清晰，默认值1

    return dayTexture;
};

const getTextureNightJpg = () => {
    const nightTexture = textureLoader.load("earth_night_8k.webp");
    nightTexture.colorSpace = THREE.SRGBColorSpace;
    nightTexture.anisotropy = 8;

    return nightTexture;
};

const getTextureBumpRoughnessLandsJpg = () => {
    const bumpRoughnessLandsTexture = textureLoader.load("earth_topography_8k.png");

    // Improves texture clarity when viewed at oblique angles 
    // (e.g., when the camera looks at a surface edge-on).
    // Anisotropy levels range from 1 (low quality) to the GPU's maximum (often 16).
    // Higher values reduce blurring but may impact performance. 8 is a balanced default.
    bumpRoughnessLandsTexture.anisotropy = 8;

    return bumpRoughnessLandsTexture;
};

const getTextureLandMask = () => {
    const landMask = textureLoader.load("land_mask_v2_8k.png");
    return landMask;
};

// Loads the base texture and calculates a ​​land influence mask​​ (landsStrength)
const getBumpRoughnessLands = () => {

    // Load Texture, get the RGB texture.
    const bumpRoughnessLandsTexture = getTextureBumpRoughnessLandsJpg();

    // Uses uv() to get texture coordinates.
    // , then samples the ​​Blue channel​​ (texture(...).b).
    // Smoothstep Interpolation​​: Applies smoothstep(0.2, 1) to the Blue channel:
    //      Maps values below 0.2 to 0, above 1 to 1, and smoothly interpolates between.
    //      Result (landsStrength) defines where "land" effects are active (e.g., roughness/bump).
    const landsStrength = texture(bumpRoughnessLandsTexture, uv()).r.smoothstep(0.2, 1);

    const landMaskTexture = getTextureLandMask();
    const landMask = texture(landMaskTexture, uv()).r;

    return { landMask, landsStrength, bumpRoughnessLandsTexture };
};

// 白天的贴图
const getTextureDay = (landsStrength) => {
    // 贴图, 白天
    const dayTexture = getTextureDayJpg();

    // colorNode: 设置轮廓颜色，Node<vec3>类型
    // mix: 在两个值之间线性插值，返回Node
    // vec3: ???
    // mul: 返回两个或多个值的乘法
    const colorNode = mix(texture(dayTexture), vec3(1), landsStrength.mul(2));

    return colorNode;
};

// 夜晚的贴图
const getTextureNight = (sunModel) => {
    // 贴图，夜晚
    const nightTexture = getTextureNightJpg();

    // 大气层颜色
    const atmosphereColor = getAtmosphereColor(sunModel);

    const fresnel = getFresnel();
    // 夜晚贴图
    const night = texture(nightTexture);
    // 太阳光白天的影响力
    const sunDayStrength = sunModel.sunOrientation.smoothstep(-0.25, 0.5);

    // 大气层白天的影响力
    const atmosphereDayStrength = sunModel.sunOrientation.smoothstep(-0.5, 1);
    const atmosphereMix = atmosphereDayStrength.mul(fresnel.pow(2)).clamp(0, 1);

    let finalOutput = mix(night.rgb, output.rgb, sunDayStrength);
    finalOutput = mix(finalOutput, atmosphereColor, atmosphereMix);

    // outputNode: 设置最终输出的材质
    // vec4: ???
    const outputNode = vec4(finalOutput, output.a);

    return outputNode;
};

// Generates a ​​roughness map​
// The reflection rate of the land the water is different
const getTextureRoughness = (bumpRoughnessLands) => {
    const landMask = bumpRoughnessLands.landMask;
    // ​Adjusts roughness range (e.g., from [0, 1] to [roughnessLow, roughnessHigh]).
    const roughnessNode = landMask.remap(0, 1, roughnessLow, roughnessHigh);
    return roughnessNode;
};

// Generates a ​​normal map​​ from elevation (Red channel) and land texture.
const getTextureNormal = (bumpRoughnessLands) => {
    const normalNode = bumpMap(texture(bumpRoughnessLands.bumpRoughnessLandsTexture).r.mul(64));
    return normalNode;
};

export { earthGroup };