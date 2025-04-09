import * as THREE from "three";

function createRadialGradientTexture(size = 25) {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');

  const gradient = ctx.createRadialGradient(
    size / 2, size / 2, 0,      // Center (brightest)
    size / 2, size / 2, size / 2 // Edge (transparent)
  );
  gradient.addColorStop(0, 'white');      // Bright center
  gradient.addColorStop(1, 'transparent'); // Faded edge

  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);

  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
}


export default function getStarfield({ numStars = 500 } = {}) {
  // --- Create gradient texture ---
  const gradientTexture = createRadialGradientTexture();

  // --- Sprite Material ---
  const material = new THREE.SpriteMaterial({
    map: gradientTexture,
    color: 0x4488ff,           // Base color (adjustable)
    transparent: true,
    blending: THREE.AdditiveBlending, // Glow effect
    opacity: 0.8,
  });

  // --- Create Instanced Sprites ---
  const starfield = new THREE.Group();

  for (let i = 0; i < numStars; i++) {
    const sprite = new THREE.Sprite(material);
    
    // Random position on a sphere
    const radius = 25 + Math.random() * 25; // 25-50 units from center
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    
    sprite.position.set(
      radius * Math.sin(phi) * Math.cos(theta),
      radius * Math.sin(phi) * Math.sin(theta),
      radius * Math.cos(phi)
    );

    // Random size and color variation
    sprite.scale.setScalar(0.01 + Math.random() * 0.1);
    sprite.material.color.setHSL(0.6, 0.5, 0.7 + Math.random() * 0.3);

    starfield.add(sprite);
  }

  return starfield;
}

