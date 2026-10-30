from datasets import load_from_disk


def explore_dataset():
    # Load the dataset
    dataset = load_from_disk("data/ag_news")

    # Print dataset information
    print(dataset)

    print("\nTraining Samples:", len(dataset["train"]))
    print("Testing Samples:", len(dataset["test"]))

    # Label names
    labels = {
        0: "World",
        1: "Sports",
        2: "Business",
        3: "Science/Technology"
    }

    print("\nFirst 5 Training Samples\n")

    for i in range(5):
        sample = dataset["train"][i]

        print("=" * 80)
        print("Sample:", i + 1)
        print("Category:", labels[sample["label"]])
        print("News:")
        print(sample["text"])


if __name__ == "__main__":
    explore_dataset()