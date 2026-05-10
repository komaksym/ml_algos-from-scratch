import math
from collections import Counter


def calculate_entropy(labels):
    label_counts = Counter(labels)
    total_count = len(labels)
    entropy = -sum(
        (count / total_count) * math.log2(count / total_count)
        for count in label_counts.values()
    )
    return entropy


def calculate_information_gain(examples, attr, target_attr):
    # Total entropy calc
    total_entropy = calculate_entropy([example[target_attr] for example in examples])
    # Get all the possible values of the searched attribute
    values = set(example[attr] for example in examples)
    attr_entropy = 0

    # Iterate value by value of the searched attribute
    for value in values:
        # Group labels by values of the searched attribute
        value_subset = [
            example[target_attr] for example in examples if example[attr] == value
        ]
        # Calculate entrope for this single value
        value_entropy = calculate_entropy(value_subset)
        # Compute entropy for a single attribute of all values
        attr_entropy += (len(value_subset) / len(examples)) * value_entropy

    # Information gain from splitting on this attribute
    return total_entropy - attr_entropy


def majority_class(examples, target_attr):
    """Find the most common target in examples"""
    return Counter([example[target_attr] for example in examples]).most_common(1)[0][0]


def learn_decision_tree(examples, attributes, target_attr):
    # No examples left
    if not examples:
        return "No examples"
    # All examples match the same target
    if all(example[target_attr] == examples[0][target_attr] for example in examples):
        return examples[0][target_attr]
    # No attributes left to split on
    if not attributes:
        return majority_class(examples, target_attr)

    # Information gain computed for every attribute
    candidates = {
        attr: calculate_information_gain(examples, attr, target_attr)
        for attr in attributes
    }
    # Splitting on the best attribute, /w highest Information Gain
    best_attr = max(candidates, key=candidates.get)
    # Declaring a subtree
    tree = {best_attr: {}}

    # Iterate over the unique values of the chosen attr
    for value in set(example[best_attr] for example in examples):
        # Group examples by values of the best_attr
        subset = [example for example in examples if example[best_attr] == value]
        # Copy attributes and remove to go further with a subtree
        new_attributes = attributes.copy()
        new_attributes.remove(best_attr)
        # Recurse
        subtree = learn_decision_tree(subset, new_attributes, target_attr)
        # Assign to a base case value
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
