def merge_tokens(corpus, merge_pair):
    new_corpus = {}

    for word, freq in corpus.items():
        tokens = word.split()
        cur_word = ""
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == merge_pair:
                merged_token = tokens[i] + tokens[i + 1]
                cur_word += merged_token + " "
                i += 2
            else:
                cur_word += tokens[i] + " "
                i += 1
        new_corpus[cur_word] = new_corpus.get(cur_word, 0) + freq

    return new_corpus


def find_most_frequent_merge(token_pairs):
    most_frequent_pair = max(token_pairs, key=token_pairs.get)
    return most_frequent_pair


def find_token_pairs(corpus):
    token_pairs = {}
    merge_possible = False

    for word, freq in corpus.items():
        if not merge_possible:
            merge_possible = True
        tokens = word.split()  # P.S. "a b " might need to be ["a", "b", " "]
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            token_pairs[pair] = token_pairs.get(pair, 0) + freq

    return token_pairs, merge_possible


def byte_pair_encoding(corpus: dict, num_merges: int) -> list:
    """
    Train a BPE tokenizer on the given corpus.

    Args:
        corpus: Dictionary mapping space-separated token sequences to their frequencies.
        Example: {"l o w </w>": 5, "n e w </w>": 6}
        num_merges: Number of merge operations to perform.

    Returns:
        List of tuples, where each tuple contains the two tokens that were merged.
        Example: [('l', 'o'), ('lo', 'w')]
    """
    merges = []

    for _ in range(num_merges):
        token_pairs, merge_possible = find_token_pairs(corpus)
        merge_pair = find_most_frequent_merge(token_pairs)

        if not merge_possible:
            return merges

        corpus = merge_tokens(corpus, merge_pair)
        merges.append(merge_pair)

    return merges


if __name__ == "__main__":
    result = byte_pair_encoding(
        {"h u g </w>": 10, "p u g </w>": 5, "p u g s </w>": 5}, 2
    )
    print(result)
