import json, subprocess, sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/leaf_data")
TASKS = {
    "leaf-add": ("calc.py", "def add(a, b):\n    return a - b\n",
                 "test_calc.py", "from calc import add\nassert add(2, 3) == 5\nassert add(-1, 1) == 0\nprint('ok')\n",
                 "`calc.add(a, b)` returns the wrong result: `add(2, 3)` should be 5."),
    "leaf-reverse": ("text.py", "def reverse_words(s):\n    return ' '.join(s.split())\n",
                     "test_text.py", "from text import reverse_words\nassert reverse_words('a b c') == 'c b a'\nassert reverse_words('hi') == 'hi'\nprint('ok')\n",
                     "`text.reverse_words` should reverse the order of the words: 'a b c' -> 'c b a'. It currently returns the input unchanged."),
    "leaf-fizz": ("fizz.py", "def fizzbuzz(n):\n    out = []\n    for i in range(1, n):\n        if i % 15 == 0:\n            out.append('FizzBuzz')\n        elif i % 3 == 0:\n            out.append('Fizz')\n        elif i % 5 == 0:\n            out.append('Buzz')\n        else:\n            out.append(str(i))\n    return out\n",
                  "test_fizz.py", "from fizz import fizzbuzz\nassert len(fizzbuzz(15)) == 15\nassert fizzbuzz(15)[-1] == 'FizzBuzz'\nprint('ok')\n",
                  "`fizz.fizzbuzz(n)` should return n items for 1..n inclusive, but it is missing the last one."),
    "leaf-max": ("stats.py", "def largest(xs):\n    best = xs[0]\n    for x in xs:\n        if x < best:\n            best = x\n    return best\n",
                 "test_stats.py", "from stats import largest\nassert largest([3, 9, 2]) == 9\nassert largest([-5, -1]) == -1\nprint('ok')\n",
                 "`stats.largest(xs)` returns the smallest element instead of the largest."),
}
images, rows = {}, []
for name, (src, code, test, test_code, issue) in TASKS.items():
    repo = root / "repos" / name
    repo.mkdir(parents=True, exist_ok=True)
    (repo / src).write_text(code)
    (repo / test).write_text(test_code)
    g = ["git", "-C", str(repo), "-c", "user.name=leaf", "-c", "user.email=leaf@example.com"]
    if not (repo / ".git").exists():
        subprocess.run(g + ["init", "-q"], check=True)
        subprocess.run(g + ["add", "-A"], check=True)
        subprocess.run(g + ["commit", "-qm", "init"], check=True)
    images[name] = {"source": str(repo), "workdir": "/testbed"}
    rows.append({
        "prompt": [{"role": "user", "content": issue}],
        "label": name,
        "metadata": {"instance_id": name, "image": name, "workdir": "/testbed", "problem_statement": issue,
                     "eval_cmd": f"python3 {test}", "eval_user": "agent"},
    })
(root / "images.json").write_text(json.dumps(images, indent=2))
(root / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
print(f"wrote {len(rows)} tasks under {root}")
