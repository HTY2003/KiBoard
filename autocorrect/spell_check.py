import json
import re
from collections import Counter

class Spell_Checker:
    def __init__(self):
        with open("english_zipf.json") as f:
            data = json.load(f)
            f.close()
        self.WORDS = Counter(data)

    def _P(self, word):
        "Probability of `word`."
        N = sum(self.WORDS.values())
        return self.WORDS[word] / N

    def _candidates(self, word):
        "Generate possible spelling corrections for word."
        return (self._known([word]) | self._known(self._edits1(word)) | self._known(self._edits2(word)) | self._known(self._edits3(word)) | set([word]))

    def _known(self, words):
        "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if w in self.WORDS)

    def _edits1(self, word):
        "All edits that are one edit away from `word`."
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        final = set(deletes + transposes + replaces + inserts)
        return final
    def _edits2(self, word):
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self._edits1(word) for e2 in self._edits1(e1))

    def _edits3(self, word):
        return (e3 for e1 in self._edits1(word) for e2 in self._edits1(e1) for e3 in self._edits1(e2))

    def correction(self, word, n=1):
        "Most probable spelling correction for word."
        words = self._candidates(word)
        print(words)
        final = []
        for i in range(n):
            if len(words) > 0:
                word = max(words, key=self._P)
                final.append(word)
                words.remove(word)
        return final

import json
import re
from collections import Counter

class Spell_Checker:
    def __init__(self):
        with open("english_zipf.json") as f:
            data = json.load(f)
            f.close()
        self.WORDS = Counter(data)

    def _P(self, word):
        "Probability of `word`."
        N = sum(self.WORDS.values())
        return self.WORDS[word] / N

    def _candidates(self, word):
        "Generate possible spelling corrections for word."
        return (self._known([word]) | self._known(self._edits1(word)) | self._known(self._edits2(word)) | self._known(self._edits3(word)) | set([word]))

    def _known(self, words):
        "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if w in self.WORDS)

    def _edits1(self, word):
        "All edits that are one edit away from `word`."
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        final = set(deletes + transposes + replaces + inserts)
        return final
    def _edits2(self, word):
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self._edits1(word) for e2 in self._edits1(e1))

    def _edits3(self, word):
        return (e3 for e1 in self._edits1(word) for e2 in self._edits1(e1) for e3 in self._edits1(e2))

    def correction(self, word, n=1):
        "Most probable spelling correction for word."
        words = self._candidates(word)
        print(words)
        final = []
        for i in range(n):
            if len(words) > 0:
                word = max(words, key=self._P)
                final.append(word)
                words.remove(word)
        return final

import json
import re
from collections import Counter

class Spell_Checker_2:
    def __init__(self):
        with open("english_zipf.json") as f:
            data = json.load(f)
            f.close()
        self.WORDS = Counter(data)
        self.insert = 1
        self.replace = 1
        self.tpose = 1
        self.delete = 1

    def _P(self, word):
        N = sum(self.WORDS.values())
        return self.WORDS[word] / N

    def _candidates(self, word):
        return (self._known([word]) | self._known(self._edits1(word)) | self._known(self._edits2(word)) | self._known(self._edits3(word)) | set([word]))

    def _known(self, words):
        return set(w for w in words if w in self.WORDS)

    def _edits1(self, word):
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        final = set(deletes + transposes + replaces + inserts)
        return final

    def _edits2(self, word):
        return (e2 for e1 in self._edits1(word) for e2 in self._edits1(e1))

    def _edits3(self, word):
        words = _edits2(word)
        repeats = 

    def correction(self, word, n=1):
        words = self._candidates(word)
        print(words)
        final = []
        for i in range(n):
            if len(words) > 0:
                word = max(words, key=self._P)
                final.append(word)
                words.remove(word)
        return final

a = Spell_Checker_2()
print(a.correction("help", 3))
