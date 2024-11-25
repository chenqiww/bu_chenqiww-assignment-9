document.getElementById("experiment-form").addEventListener("submit", function(event) {
    event.preventDefault();  // Prevent form submission

    const activation = document.getElementById("activation").value;
    const lr = parseFloat(document.getElementById("lr").value);
    const stepNum = parseInt(document.getElementById("step_num").value);

    // Validation checks
    const acts = ["relu", "tanh", "sigmoid"];
    if (!acts.includes(activation)) {
        alert("Please choose from relu, tanh, sigmoid.");
        return;
    }

    if (isNaN(lr)) {
        alert("Please enter a valid number for learning rate.");
        return;
    }

    if (isNaN(stepNum) || stepNum <= 0) {
        alert("Please enter a positive integer for Number of Training Steps.");
        return;
    }

    // Clear previous results
    const resultsDiv = document.getElementById("results");
    resultsDiv.style.display = "none";
    const resultImg = document.getElementById("result_gif");
    resultImg.style.display = "none";
    resultImg.src = "";

    // Display a loading message or spinner (optional)
    const loadingMessage = document.getElementById("loading-message");
    loadingMessage.style.display = "block";

    // Disable the submit button to prevent multiple submissions
    const submitButton = document.querySelector("button[type='submit']");
    submitButton.disabled = true;

    // If all validations pass, submit the form
    fetch("/run_experiment", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ activation: activation, lr: lr, step_num: stepNum })
    })
    .then(response => response.json())
    .then(data => {
        // Hide the loading message
        loadingMessage.style.display = "none";

        // Re-enable the submit button
        submitButton.disabled = false;

        // Show and set images if they exist
        if (data.result_gif) {
            resultsDiv.style.display = "block";
            resultImg.src = `/${data.result_gif}`;
            resultImg.style.display = "block";
        } else {
            alert("No results were generated.");
        }
    })
    .catch(error => {
        console.error("Error running experiment:", error);
        alert("An error occurred while running the experiment.");
        loadingMessage.style.display = "none";

        // Re-enable the submit button
        submitButton.disabled = false;
    });
});
