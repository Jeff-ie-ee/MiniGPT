const generateButton = document.getElementById("generateButton");

const promptInput = document.getElementById("prompt");
const temperatureInput = document.getElementById("temperature");
const topKInput = document.getElementById("topK");
const tokensInput = document.getElementById("tokens");

const output = document.getElementById("output");
const status = document.getElementById("status");


generateButton.addEventListener("click", async () => {

    const prompt = promptInput.value.trim();

    if (!prompt) {
        output.textContent = "Please enter a prompt.";
        return;
    }

    const temperature = parseFloat(temperatureInput.value);
    const topK = parseInt(topKInput.value);
    const maxNewTokens = parseInt(tokensInput.value);

    generateButton.disabled = true;
    status.textContent = "Generating...";
    output.textContent = "";

    try {

        const response = await fetch(
            "http://127.0.0.1:8000/generate",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    prompt: prompt,
                    temperature: temperature,
                    top_k: topK,
                    max_new_tokens: maxNewTokens
                })
            }
        );

        const data = await response.json();

        if (data.error) {
            output.textContent = data.error;
        } else {
            output.textContent = data.generated_text;
        }

        status.textContent = "Generation complete.";

    } catch (error) {

        console.error(error);

        output.textContent =
            "Could not connect to MiniGPT API.";

        status.textContent =
            "Make sure the FastAPI server is running.";

    } finally {

        generateButton.disabled = false;
    }
});