const images = [
    'bg1.jpg',
    'bg2.jpg',
    'bg3.jpg'
  ];
  
  const bgElement = document.getElementById('bg'); // Grab the <div> element
  
  // Function to set a random background image
  function setRandomBackground() {
    const randomImage = images[Math.floor(Math.random() * images.length)]; // Randomly select an image
    bgElement.style.backgroundImage = `url('${randomImage}')`; // Change the background image
  }
  
  // Set the random background when the page loads
  window.onload = setRandomBackground;
  