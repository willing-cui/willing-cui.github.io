import * as THREE from "three";
import { normalWorld, normalize } from "three/tsl";

const sunModel = () => {
    // 自动获取当前日期
    const date = new Date();

    // 1. 根据日期计算太阳角度
    // 假设：夏至（6月21日）角度最大，冬至（12月21日）角度最小
    // 简化模型：角度在 -23.5° 到 23.5° 之间变化（黄赤交角）
    
    const getSunAngle = (date) => {
        // 获取一年中的第几天（0-365）
        const start = new Date(date.getFullYear(), 0, 0);
        const diff = date - start;
        const oneDay = 1000 * 60 * 60 * 24;
        const dayOfYear = Math.floor(diff / oneDay);
        
        // 计算角度（弧度）
        // 使用正弦函数模拟季节变化
        const angleInDegrees = 23.5 * Math.sin(2 * Math.PI * (dayOfYear - 80) / 365.25);
        
        return THREE.MathUtils.degToRad(angleInDegrees);
    };
    
    // 2. 计算y值
    // 已知：夹角θ = arctan(y/10)
    // 所以：y = 10 * tan(θ)
    const angle = getSunAngle(date);
    console.log('当前阳光与赤道平面夹角：%f', angle / 3.1415926 * 180);
    const y = 10 * Math.tan(angle);

    // 太阳，先用光源简单模拟
    const sunLight = new THREE.DirectionalLight("#ffffff", 2);
    sunLight.position.set(0, y, 10);

    // 太阳光的方向
    const sunOrientation = normalWorld.dot(normalize(sunLight.position)).toVar();

    return { 
        sunLight, 
        sunOrientation,
        // 添加一些额外信息，方便调试
        date: date,
        sunAngle: angle,
        sunY: y
    };
};

export { sunModel };