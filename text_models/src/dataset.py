from datasets import load_dataset
import os


def download_dataset():
    print("Downloading AG News dataset...")

    # Download dataset
    dataset = load_dataset("ag_news")

    # Create folder if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Save dataset
    dataset.save_to_disk("data/ag_news")

    print("\nDataset downloaded successfully!")
    print(dataset)

    print("\nTraining Samples:", len(dataset["train"]))
    print("Testing Samples:", len(dataset["test"]))

    print("\nFirst Training Sample:")
    print(dataset["train"][0])


if __name__ == "__main__":
    download_dataset()