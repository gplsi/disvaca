from datasets import load_dataset


def get_words():
    dataset = load_dataset("gplsi/es_vaca", data_files="data.csv")
    data = dataset["train"]
    return {
        "va": [
            subword.lower()
            for word in data["VA"]
            if word is not None
            for subword in word.split("||")
        ],
        "ca": [
            subword.lower()
            for word in data["CA"]
            if word is not None
            for subword in word.split("||")
        ],
    }
