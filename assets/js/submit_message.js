window.addEventListener("load", function() {
    // Wait for the page to fully load
    const form = document.getElementById('visitorMsg'); // Get the form element
    form.addEventListener("submit", function(e) { // Listen for form submission
        e.preventDefault(); // Prevent the default form submission behavior
        const data = new FormData(form); // Create a FormData object from the form
        const action = e.target.action; // Get the form's action URL

        fetch(action, { // Send a POST request to the server
            method: 'POST',
            body: data,
            mode: 'no-cors',
        })
        .then(response => {
            console.log(response);
            alert("Success!");
            form.reset();
        })
        .catch(error => {
            console.error('Error:', error);
            alert("An error occurred. Please try again.");
        });
    });
});