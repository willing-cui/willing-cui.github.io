const images = [
    './images/bg1.webp',
    './images/bg2.webp',
    './images/bg3.webp',
    './images/bg4.webp',
    './images/bg5.webp',
    './images/bg6.webp',
    './images/bg7.webp',
    './images/bg8.webp',
    './images/bg9.webp',
    './images/bg10.webp',
    './images/bg11.webp',
    './images/bg12.webp',
    './images/bg13.webp',
    './images/bg14.webp',
    './images/bg15.webp',
    './images/bg16.webp',
    './images/bg17.webp',
    './images/bg18.webp',
    './images/bg19.webp',
    './images/bg20.webp',
    './images/bg21.webp',
    './images/bg22.webp',
    './images/bg23.webp',
    './images/bg24.webp',
    './images/bg25.webp',
    './images/bg26.webp',
    './images/bg27.webp',
    './images/bg28.webp',
    './images/bg29.webp',
    './images/bg30.webp'
];

// Function to set a random background image
function setRandomBackground() {
    const randomImage = images[Math.floor(Math.random() * images.length)]; // Randomly select an image

    // Create a <style> element to update the :after pseudo-element
    const style = document.createElement('style');
    style.innerHTML = `
    #bg:after {
      background-image: url('${randomImage}');
    }
  `;

    // Add the new style to the document
    document.head.appendChild(style);
}

// Set the random background when the page loads
window.onload = setRandomBackground;