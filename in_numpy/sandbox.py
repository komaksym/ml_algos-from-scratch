"""
Given a list of words and an integer k, return the k most frequent words.

"""


def k_most_frequent(words, k):
    freqs = {}

    for w in words:
        if w not in freqs:
            freqs[w] = 1
        else:
            freqs[w] = freqs[w] + 1

    srted = sorted(freqs, key=freqs.get)[:2]
    print(srted)


if __name__ == "__main__":
    words = ["i", "love", "ai", "i", "love", "coding"]
    k = 2

    k_most_frequent(words, k)
