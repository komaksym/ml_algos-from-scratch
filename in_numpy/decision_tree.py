import math
from collections import Counter


def calculate_entropy(labels):
    label_counts = Counter(labels)
    label_total = len(labels)
    entropy = -sum(
        count / label_total * math.log2(count / label_total)
        for count in label_counts.values()
    )
    return entropy


def calculate_information_gain(examples, attr, target_attr):
    total_entropy = calculate_entropy([example[target_attr] for example in examples])
    values = set(example[attr] for example in examples)
    attr_entropy = 0

    for value in values:
        value_subset = [
            example[target_attr] for example in examples if example[attr] == value
        ]
        value_entropy = calculate_entropy(value_subset)
        attr_entropy += (len(value_subset) / len(examples)) * value_entropy

    return total_entropy - attr_entropy


def majority_class(examples, target_attr):
    return Counter(example[target_attr] for example in examples).most_common(1)[0][0]


def learn_decision_tree(
    examples: list[dict], attributes: list[str], target_attr: str
) -> dict:
    if not examples:
        return "No examples"
    if all(example[target_attr] == examples[0][target_attr] for example in examples):
        return examples[0][target_attr]
    if not attributes:
        return majority_class(examples, target_attr)

    candidates = {
        attr: calculate_information_gain(examples, attr, target_attr)
        for attr in attributes
    }
    best_attr = max(candidates, key=candidates.get)
    tree = {best_attr: {}}

    for value in set(example[best_attr] for example in examples):
        subset = [example for example in examples if example[best_attr] == value]
        new_attributes = attributes.copy()
        new_attributes.remove(best_attr)
        subtree = learn_decision_tree(subset, new_attributes, target_attr)
        tree[best_attr][value] = subtree

    return tree


examples = [
    {
        "Outlook": "Sunny",
        "Temperature": "Hot",
        "Humidity": "High",
        "Wind": "Weak",
        "PlayTennis": "No",
    },
    {
        "Outlook": "Sunny",
        "Temperature": "Hot",
        "Humidity": "High",
        "Wind": "Strong",
        "PlayTennis": "Yes",
    },
    {
        "Outlook": "Overcast",
        "Temperature": "Hot",
        "Humidity": "High",
        "Wind": "Weak",
        "PlayTennis": "Yes",
    },
    {
        "Outlook": "Rain",
        "Temperature": "Mild",
        "Humidity": "High",
        "Wind": "Weak",
        "PlayTennis": "Yes",
    },
]

attributes = ["Outlook", "Temperature", "Humidity", "Wind"]
target_attr = "PlayTennis"

print(learn_decision_tree(examples, attributes, target_attr))
