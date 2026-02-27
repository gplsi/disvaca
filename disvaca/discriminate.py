import spacy


class VACADiscriminator:
    def __init__(self, words, tokenizer):
        self.words = words
        self.tokenizer = tokenizer

    @classmethod
    def load(cls, words=None, learned=True, es_vaca=True):
        """Load the discriminator with the given words."""
        tokenizer = cls.get_tokenizer()

        if words is None:
            words = {}

        if learned:
            from disvaca.learned_words import get_words as get_learned_words

            words = cls.merge_words(words, get_learned_words())

        if es_vaca:
            from disvaca.es_vaca import get_words as get_esvaca_words

            words = cls.merge_words(words, get_esvaca_words())

        return cls(words, tokenizer)

    @classmethod
    def merge_words(cls, words1, words2):
        """Merge two dictionaries of words, combining the sets for each language."""
        merged = {}
        for language in set(words1) | set(words2):
            merged[language] = set(words1.get(language, [])) | set(
                words2.get(language, [])
            )
        return merged

    @staticmethod
    def get_tokenizer():
        """Get the SpaCy tokenizer for Catalan (also works for Valencian)."""
        return spacy.load("ca_core_news_sm")

    def get_words(self, input):
        """Get words from the text."""
        doc = self.tokenizer(input)
        return [token.text.lower() for token in doc if token.is_alpha]

    def get_words_if_needed(self, input):
        """Get words from the text if it's a string, otherwise return it as is."""
        return self.get_words(input) if isinstance(input, str) else input

    def is_va(self, input):
        """Check if the text contains any Valencian words."""
        return self.check_language(input, "va")

    def is_ca(self, input):
        """Check if the text contains any Catalan words."""
        return self.check_language(input, "ca")

    def check_language(self, input, language):
        """Check if the text contains any words from the specified language."""
        words = self.get_words_if_needed(input)
        return any((word in self.words[language]) for word in map(str.lower, words))

    def count_language_words(self, input, language):
        """Count the number of words from the specified language in the text."""
        words = self.get_words_if_needed(input)
        return sum(1 for word in map(str.lower, words) if word in self.words[language])

    def get_languages(self, input):
        """Get the languages present in the text."""
        words = self.get_words_if_needed(input)
        return {
            language for language in self.words if self.check_language(words, language)
        }

    def get_ratios(self, input):
        """Get the ratio of words belonging to each language in the text.
        """
        words = self.get_words_if_needed(input)
        
        lang_counts = {
            language: self.count_language_words(words, language)
            for language in self.words
        
        }
        total_matched = sum(lang_counts.values())

        if total_matched == 0:
            return {language: 0.0 for language in self.words}

        return {
            language: lang_counts[language] / total_matched
            for language in self.words
        }
