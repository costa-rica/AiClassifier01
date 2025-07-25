from transformers import pipeline
import torch

# Check if Apple MPS (Metal) is available; fallback to CPU
device = 0 if torch.backends.mps.is_available() else -1

# Initialize zero-shot classifier
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=device
)

# Test text (replace with article excerpt)
# text = "A defective toy manufactured in China caused injuries to children in California."
text = "A vehicle hit a girl riding an electric scooter Monday night on Tunxis Hill Cutoff and Knapps Highway in Fairfield. Police say the vehicle was going through a green light at the time of impact. The girl was taken to a hospital for injuries and released. Police say she and the driver received citations. The driver was cited for having an obstructed windshield view. The scooter operator was in violation of improper use of the highway by a pedestrian."

# Candidate labels for classification
labels = ["Occurred in the United States", "Occurred outside the United States"]

# Run classification
result = classifier(text, candidate_labels=labels)

print("\n--- Classification Result ---")
print(result)