document.getElementById('review-form').addEventListener('submit', async function (event) {
  event.preventDefault(); // it prevents normal form submit

  const reviewInput = document.getElementById('review-input');
  const resultDiv = document.getElementById('prediction-result');
  const review = reviewInput.value.trim();

  if (!review) {
    resultDiv.innerHTML = "⚠️ Please enter a review.";
    resultDiv.style.color = "orange";
    return;
  }

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'  // Sending JSON
      },
      body: JSON.stringify({ review: review }) // Convert to JSON
    });

    const data = await response.json();
    const prediction = data.result.toLowerCase();

    if (prediction === 'good') {
      resultDiv.innerHTML = "✅ Good";
      resultDiv.style.color = "green";
    } else if (prediction === 'bad') {
      resultDiv.innerHTML = "❌ Bad";
      resultDiv.style.color = "red";
    } else {
      resultDiv.innerHTML = `⚠️ ${prediction}`;
      resultDiv.style.color = "orange";
    }

  } catch (error) {
    resultDiv.innerHTML = "❌ Error connecting to server.";
    resultDiv.style.color = "red";
    console.error("Error:", error);
  }
});
