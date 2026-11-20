import json

from hypecut.captions import Word, build_ass, chunk_words
from hypecut.signals import chat_activity, load_chat


def test_load_twitchdownloader(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"comments": [
        {"content_offset_seconds": 12.5, "message": {"body": "KEKW"}},
        {"content_offset_seconds": 13.0, "message": {"body": "hi"}},
    ]}))
    assert load_chat(p) == [(12.5, "KEKW"), (13.0, "hi")]


def test_load_chat_downloader_and_csv(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps([{"time_in_seconds": 3, "message": "W"}]))
    assert load_chat(p) == [(3.0, "W")]
    c = tmp_path / "c.csv"
    c.write_text("seconds,message\n4,no way, bro\n")
    assert load_chat(c) == [(4.0, "no way, bro")]


def test_chat_delay_and_hype_weight():
    act, toks = chat_activity([(15, "KEKW KEKW"), (15, "hello there")], n_sec=20, delay=5)
    assert act[10] > 2.0 and act[15] == 0
    assert toks[10]["kekw"] == 2


def test_chunking_and_ass(tmp_path):
    words = [Word(i * 0.3, i * 0.3 + 0.25, w) for i, w in enumerate("bro what is he doing {lol}".split())]
    chunks = chunk_words(words)
    assert all(len(c) <= 3 for c in chunks)
    ass = build_ass(words, tmp_path / "a.ass").read_text()
    assert "PlayResY: 1920" in ass and "\\k" in ass
    assert "{LOL}" not in ass and "(LOL)" in ass
