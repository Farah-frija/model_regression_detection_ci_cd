from email_classifier import ClassificationResult, ClassifierError
from email_classifier.__main__ import main


def _result():
    return ClassificationResult(
        prompt_version="0.1.0",
        model="gemini-3.5-flash-lite",
        category="billing",
        summary="Customer was charged twice.",
        latency_ms=1234.0,
        prompt_tokens=100,
        completion_tokens=20,
    )


async def _fake_classify(text, **kwargs):
    return _result()


async def _failing_classify(text, **kwargs):
    raise ClassifierError("No Gemini API key found.")


def test_prints_result_and_returns_zero(monkeypatch, capsys):
    monkeypatch.setattr("email_classifier.__main__.classify_email", _fake_classify)

    code = main(["I", "was", "charged", "twice"])

    out = capsys.readouterr().out
    assert code == 0
    assert "Category:    billing" in out
    assert "Customer was charged twice." in out
    assert "100 in, 20 out" in out


def test_reads_email_from_stdin_when_no_arguments(monkeypatch, capsys):
    seen = {}

    async def spy(text, **kwargs):
        seen["text"] = text
        return _result()

    monkeypatch.setattr("email_classifier.__main__.classify_email", spy)
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("from stdin\n"))

    assert main([]) == 0
    assert seen["text"] == "from stdin"


def test_empty_input_returns_one(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("   \n"))

    assert main([]) == 1
    assert "No email text given" in capsys.readouterr().err


def test_classifier_error_returns_one_and_prints_to_stderr(monkeypatch, capsys):
    monkeypatch.setattr("email_classifier.__main__.classify_email", _failing_classify)

    code = main(["hello"])

    captured = capsys.readouterr()
    assert code == 1
    assert "No Gemini API key found." in captured.err
    assert captured.out == ""
