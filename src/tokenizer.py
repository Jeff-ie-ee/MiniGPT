class CharacterTokenizer:

    def __init__(self, text):
        # Get all unique characters
        chars = sorted(list(set(text)))

        # Character → integer
        self.stoi = {
            ch: i for i, ch in enumerate(chars)
        }

        # Integer → character
        self.itos = {
            i: ch for ch, i in self.stoi.items()
        }

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, tokens):
        return ''.join(self.itos[i] for i in tokens)


if __name__ == "__main__":

    text = "I love AI"

    tokenizer = CharacterTokenizer(text)

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    print("Vocabulary:")
    print(tokenizer.stoi)

    print("\nEncoded:")
    print(encoded)

    print("\nDecoded:")
    print(decoded)