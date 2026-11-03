import unittest

from llm.corpus import brown_to_text, chunk_paragraphs, normalize, split_of, strip_gutenberg


class CorpusTests(unittest.TestCase):
    def test_brown_tags_removed_and_punctuation_attached(self):
        raw = "\t\tThe/at Fulton/np-tl County/nn-tl said/vbd ``/`` no/at ''/'' ./."
        self.assertEqual(brown_to_text(raw), 'The Fulton County said " no ".')

    def test_gutenberg_header_footer_removed(self):
        raw = "license junk\n*** START OF THIS PROJECT ***\nbody line\n*** END OF THIS PROJECT ***\nfooter"
        self.assertEqual(strip_gutenberg(raw), "body line")

    def test_normalize_collapses_blank_lines(self):
        self.assertEqual(normalize("a \r\n\r\n\r\n\r\nb  "), "a\n\nb")

    def test_split_is_deterministic_and_mostly_train(self):
        self.assertEqual(split_of("gutenberg/austen-emma.txt"), split_of("gutenberg/austen-emma.txt"))
        ids = [f"doc{i}" for i in range(2000)]
        counts = {s: sum(split_of(i) == s for i in ids) for s in ("train", "dev", "test")}
        self.assertGreater(counts["train"], 1700)
        self.assertGreater(counts["dev"], 0)
        self.assertGreater(counts["test"], 0)

    def test_chunk_paragraphs(self):
        text = "\n\n".join(f"p{i}" for i in range(5))
        self.assertEqual(chunk_paragraphs(text, size=2), ["p0\n\np1", "p2\n\np3", "p4"])


if __name__ == "__main__":
    unittest.main()
