// Custom Form Validation and Interactivity
document.addEventListener("DOMContentLoaded", function () {
    // Bootstrap form validation listener
    const forms = document.querySelectorAll(".needs-validation");

    Array.from(forms).forEach(function (form) {
        form.addEventListener("submit", function (event) {
            let isValid = true;
            
            // Check HTML5 default validation
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
                isValid = false;
            }

            // Perform additional logic checks
            const nitrogen = document.getElementById("nitrogen");
            const phosphorous = document.getElementById("phosphorous");
            const potassium = document.getElementById("potassium");
            const temperature = document.getElementById("temperature");
            const humidity = document.getElementById("humidity");
            const ph = document.getElementById("ph");
            const rainfall = document.getElementById("rainfall");

            // Helper function to mark invalid
            const markInvalid = (inputElement, feedbackMessage) => {
                inputElement.classList.add("is-invalid");
                const feedbackNode = inputElement.parentNode.querySelector(".invalid-feedback");
                if (feedbackNode) {
                    feedbackNode.textContent = feedbackMessage;
                }
                isValid = false;
            };

            // Reset custom invalid states
            [nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall].forEach(el => {
                if (el) el.classList.remove("is-invalid");
            });

            // Validate negative numbers
            const inputs = [
                { el: nitrogen, name: "Nitrogen" },
                { el: phosphorous, name: "Phosphorous" },
                { el: potassium, name: "Potassium" },
                { el: temperature, name: "Temperature" },
                { el: humidity, name: "Humidity" },
                { el: ph, name: "pH" },
                { el: rainfall, name: "Rainfall" }
            ];

            inputs.forEach(item => {
                if (item.el) {
                    const val = parseFloat(item.el.value);
                    if (isNaN(val)) {
                        markInvalid(item.el, `${item.name} must be a number.`);
                    } else if (val < 0 && item.name !== "Temperature") {
                        // Temperature can be negative in winter environments, but others cannot
                        markInvalid(item.el, `${item.name} cannot be negative.`);
                    }
                }
            });

            // Specific limits validation
            if (ph && (parseFloat(ph.value) < 0 || parseFloat(ph.value) > 14)) {
                markInvalid(ph, "pH level must be between 0.0 and 14.0.");
            }

            if (humidity && parseFloat(humidity.value) > 100) {
                markInvalid(humidity, "Humidity percentage cannot exceed 100%.");
            }

            if (!isValid) {
                event.preventDefault();
                event.stopPropagation();
            }

            form.classList.add("was-validated");
        }, false);
    });
});

// Resets form fields and validation indicators
function resetForm() {
    const form = document.getElementById("cropForm");
    if (form) {
        form.reset();
        form.classList.remove("was-validated");
        
        // Remove individual invalid indicators
        const inputs = form.querySelectorAll(".form-control");
        inputs.forEach(input => {
            input.classList.remove("is-invalid");
        });
    }
}
