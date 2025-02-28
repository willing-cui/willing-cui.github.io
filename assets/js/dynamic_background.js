const images = [
  './images/bg1.jpg',
  './images/bg2.jpg',
  './images/bg3.jpg',
  './images/bg4.jpg',
  './images/bg5.jpg',
  './images/bg6.jpg',
  './images/bg7.jpg',
  './images/bg8.jpg',
  './images/bg9.jpg',
  './images/bg10.jpg',
  './images/bg11.jpg',
  './images/bg12.jpg',
  './images/bg13.jpg',
  './images/bg14.jpg',
  './images/bg15.jpg',
  './images/bg16.jpg',
  './images/bg17.jpg',
  './images/bg18.jpg',
  './images/bg19.jpg',
  './images/bg20.jpg'
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