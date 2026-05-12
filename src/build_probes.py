"""Build the extended 50-item post-1930 probe battery.

Reads the 20-item seed in ``datasets/post1930_probes/seed_probes.jsonl`` and
adds 30 hand-curated items that span:
- post-1930 factual / event completions  (kind=completion)
- post-1930 in-context-learning probes for Python and JavaScript (kind=icl)
- a small "pre-1930 control" subset (kind=completion, year < 1930) so the
  knowledge-cutoff effect can be measured against an in-distribution
  baseline.

Writes the union to ``datasets/post1930_probes/probes.jsonl``.
"""

from __future__ import annotations

from pathlib import Path

import json

REPO = Path(__file__).resolve().parents[1]
SEED = REPO / "datasets" / "post1930_probes" / "seed_probes.jsonl"
OUT = REPO / "datasets" / "post1930_probes" / "probes.jsonl"


# ---------------------------------------------------------------------------
# Additional items — 30 total: 14 post-1930 facts, 8 ICL items, 8 pre-1930
# controls. The expected_modern field is what a contemporary model would say.
# Pre-1930 controls have year < 1930 — Talkie should plausibly do well on these.
# ---------------------------------------------------------------------------

EXTRA = [
    # ---------------------- post-1930 events / inventions ------------------
    {"id": "extra-fact-1", "year": 1957, "category": "event",
     "concept": "Sputnik launch",
     "prompt": "On October 4, 1957, the Soviet Union launched the first artificial satellite, named",
     "expected_modern": "Sputnik 1", "kind": "completion"},
    {"id": "extra-fact-2", "year": 1969, "category": "event",
     "concept": "Apollo 11 landing date",
     "prompt": "Apollo 11 landed on the Moon on July",
     "expected_modern": "20, 1969", "kind": "completion"},
    {"id": "extra-fact-3", "year": 1977, "category": "invention",
     "concept": "personal computer",
     "prompt": "The Apple II personal computer was released in",
     "expected_modern": "1977", "kind": "completion"},
    {"id": "extra-fact-4", "year": 1981, "category": "invention",
     "concept": "MS-DOS operating system",
     "prompt": "The IBM Personal Computer, released in 1981, shipped with an operating system called",
     "expected_modern": "MS-DOS", "kind": "completion"},
    {"id": "extra-fact-5", "year": 1986, "category": "event",
     "concept": "Chernobyl disaster",
     "prompt": "The Chernobyl nuclear disaster occurred in",
     "expected_modern": "1986", "kind": "completion"},
    {"id": "extra-fact-6", "year": 2007, "category": "invention",
     "concept": "iPhone",
     "prompt": "Apple released the first iPhone in the year",
     "expected_modern": "2007", "kind": "completion"},
    {"id": "extra-fact-7", "year": 1996, "category": "discovery",
     "concept": "cloned sheep",
     "prompt": "Dolly the cloned sheep was born in",
     "expected_modern": "1996", "kind": "completion"},
    {"id": "extra-fact-8", "year": 2001, "category": "event",
     "concept": "September 11 attacks",
     "prompt": "The World Trade Center towers in New York were destroyed in attacks on September",
     "expected_modern": "11, 2001", "kind": "completion"},
    {"id": "extra-fact-9", "year": 1957, "category": "scientific-concept",
     "concept": "DNA replication (Meselson-Stahl)",
     "prompt": "The Meselson-Stahl experiment demonstrated that DNA replication is",
     "expected_modern": "semiconservative", "kind": "completion"},
    {"id": "extra-fact-10", "year": 1990, "category": "scientific-concept",
     "concept": "Hubble Space Telescope",
     "prompt": "The Hubble Space Telescope was launched into orbit in",
     "expected_modern": "1990", "kind": "completion"},
    {"id": "extra-fact-11", "year": 1989, "category": "neologism",
     "concept": "World Wide Web (proposal)",
     "prompt": "The protocol that became the World Wide Web is built on top of an application-layer protocol called",
     "expected_modern": "HTTP", "kind": "completion"},
    {"id": "extra-fact-12", "year": 1971, "category": "invention",
     "concept": "email",
     "prompt": "The character used to separate a username from a domain in email addresses is the",
     "expected_modern": "@ symbol", "kind": "completion"},
    {"id": "extra-fact-13", "year": 2003, "category": "discovery",
     "concept": "Human Genome Project",
     "prompt": "The Human Genome Project announced the essentially complete human genome sequence in the year",
     "expected_modern": "2003", "kind": "completion"},
    {"id": "extra-fact-14", "year": 2020, "category": "event",
     "concept": "COVID-19 pandemic",
     "prompt": "The World Health Organization declared COVID-19 a pandemic in March of",
     "expected_modern": "2020", "kind": "completion"},

    # ---------------------- in-context-learning probes ---------------------
    {"id": "extra-icl-py-1", "year": 1991, "category": "icl-python",
     "concept": "Python lists",
     "prompt": "# Python\nnums = [4, 1, 7, 2]\nnums.sort()\nprint(nums)\n# Output: [1, 2, 4, 7]\n\nletters = ['c', 'a', 'b']\nletters.sort()\nprint(letters)\n# Output:",
     "expected_modern": "['a', 'b', 'c']", "kind": "icl"},
    {"id": "extra-icl-py-2", "year": 1991, "category": "icl-python",
     "concept": "Python conditional",
     "prompt": "# Python\nx = 7\nif x > 5:\n    print('big')\nelse:\n    print('small')\n# Output: big\n\nx = 3\nif x > 5:\n    print('big')\nelse:\n    print('small')\n# Output:",
     "expected_modern": "small", "kind": "icl"},
    {"id": "extra-icl-py-3", "year": 1991, "category": "icl-python",
     "concept": "Python string concatenation",
     "prompt": "# Python\na = 'foo'\nb = 'bar'\nprint(a + b)\n# Output: foobar\n\nx = 'hello'\ny = 'world'\nprint(x + y)\n# Output:",
     "expected_modern": "helloworld", "kind": "icl"},
    {"id": "extra-icl-py-4", "year": 1991, "category": "icl-python",
     "concept": "Python while loop",
     "prompt": "# Python\nn = 0\nwhile n < 3:\n    print(n)\n    n = n + 1\n# Output:\n# 0\n# 1\n# 2\n\nn = 0\nwhile n < 2:\n    print(n)\n    n = n + 1\n# Output:\n#",
     "expected_modern": "0", "kind": "icl"},
    {"id": "extra-icl-js-1", "year": 1995, "category": "icl-language",
     "concept": "JavaScript arrow function",
     "prompt": "// JavaScript\nconst sq = x => x * x;\nconsole.log(sq(4));\n// Output: 16\n\nconst cube = x => x * x * x;\nconsole.log(cube(3));\n// Output:",
     "expected_modern": "27", "kind": "icl"},
    {"id": "extra-icl-js-2", "year": 1995, "category": "icl-language",
     "concept": "JavaScript template literal",
     "prompt": "// JavaScript\nconst name = 'Alice';\nconsole.log(`Hello, ${name}!`);\n// Output: Hello, Alice!\n\nconst name2 = 'Bob';\nconsole.log(`Hello, ${name2}!`);\n// Output:",
     "expected_modern": "Hello, Bob!", "kind": "icl"},
    {"id": "extra-icl-mini-1", "year": 1991, "category": "icl-python",
     "concept": "post-1930 keyword 'def'",
     "prompt": "# Python defines functions with the keyword 'def'.\n# Example:\ndef add(a, b):\n    return a + b\n\nadd(2, 3)\n# Output: 5\n\n# Now define a function 'sub' that subtracts b from a.\ndef sub(a, b):\n    return",
     "expected_modern": "a - b", "kind": "icl"},
    {"id": "extra-icl-mini-2", "year": 1991, "category": "icl-python",
     "concept": "Python boolean operators",
     "prompt": "# Python\nprint(True and False)\n# Output: False\nprint(True or False)\n# Output: True\nprint(not True)\n# Output:",
     "expected_modern": "False", "kind": "icl"},

    # ---------------------- pre-1930 controls (in-distribution) -----------
    {"id": "ctrl-pre-1", "year": 1492, "category": "history-pre1930-control",
     "concept": "Columbus voyage",
     "prompt": "In 1492, Christopher Columbus sailed across the Atlantic Ocean and reached",
     "expected_modern": "the Americas", "kind": "completion"},
    {"id": "ctrl-pre-2", "year": 1865, "category": "history-pre1930-control",
     "concept": "Lincoln assassination",
     "prompt": "President Abraham Lincoln was assassinated by John Wilkes Booth in the year",
     "expected_modern": "1865", "kind": "completion"},
    {"id": "ctrl-pre-3", "year": 1859, "category": "history-pre1930-control",
     "concept": "Origin of Species",
     "prompt": "Charles Darwin published \"On the Origin of Species\" in the year",
     "expected_modern": "1859", "kind": "completion"},
    {"id": "ctrl-pre-4", "year": 1789, "category": "history-pre1930-control",
     "concept": "French Revolution",
     "prompt": "The French Revolution began in the year",
     "expected_modern": "1789", "kind": "completion"},
    {"id": "ctrl-pre-5", "year": 1879, "category": "invention-pre1930-control",
     "concept": "incandescent light bulb",
     "prompt": "Thomas Edison patented a practical incandescent electric light bulb in the year",
     "expected_modern": "1879", "kind": "completion"},
    {"id": "ctrl-pre-6", "year": 1905, "category": "physics-pre1930-control",
     "concept": "special relativity",
     "prompt": "Albert Einstein published his theory of special relativity in the year",
     "expected_modern": "1905", "kind": "completion"},
    {"id": "ctrl-pre-7", "year": 1903, "category": "invention-pre1930-control",
     "concept": "Wright brothers flight",
     "prompt": "The Wright brothers made the first powered, controlled airplane flight near Kitty Hawk in the year",
     "expected_modern": "1903", "kind": "completion"},
    {"id": "ctrl-pre-8", "year": 1914, "category": "history-pre1930-control",
     "concept": "World War I start",
     "prompt": "World War I began in the year",
     "expected_modern": "1914", "kind": "completion"},
]


def main() -> None:
    seed_items: list[dict] = []
    with SEED.open() as f:
        for line in f:
            line = line.strip()
            if line:
                seed_items.append(json.loads(line))
    all_items = seed_items + EXTRA

    # Sanity: unique ids
    seen = set()
    for d in all_items:
        if d["id"] in seen:
            raise ValueError(f"duplicate id {d['id']}")
        seen.add(d["id"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for d in all_items:
            f.write(json.dumps(d) + "\n")

    n_total = len(all_items)
    n_post = sum(1 for d in all_items if d["year"] >= 1930)
    n_pre = sum(1 for d in all_items if d["year"] < 1930)
    n_icl = sum(1 for d in all_items if d["kind"] == "icl")
    print(f"Wrote {OUT} — {n_total} probes "
          f"({n_post} post-1930, {n_pre} pre-1930 controls, {n_icl} ICL).")


if __name__ == "__main__":
    main()
