"""Small ordinary tests for bounded public-data I/O; no network or training."""

import json
from zipfile import ZipFile

import pytest

from football_model.wyscout_data import iter_json_array, iter_match_events, unpack_member


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 11, 64, 65536])
@pytest.mark.parametrize("ascii_escapes", [False, True])
def test_tiny_chunks_preserve_unicode_escapes_and_nested_values(tmp_path, chunk_size, ascii_escapes):
    rows = [
        {"matchId": 1, "name": "阿森纳 ⚽ é", "text": 'comma, bracket ] brace } quote " slash \\',
         "nested": {"array": [1, -2.25e-12, True, False, None, {"inner": "汉字\n\t"}]}},
        {},
        {"matchId": 2, "escaped": "\\u4e2d", "number": 12345678901234567890},
    ]
    path = tmp_path / "unicode.json"
    path.write_text(" \n" + json.dumps(rows, ensure_ascii=ascii_escapes) + " \r\n", encoding="utf-8-sig")
    assert list(iter_json_array(path, chunk_size=chunk_size)) == rows


@pytest.mark.parametrize("text", ["[]", " \n[ \r\n ]\t", "\ufeff[]"])
@pytest.mark.parametrize("chunk_size", [1, 2, 16])
def test_empty_array_including_bom_and_whitespace(tmp_path, text, chunk_size):
    path = tmp_path / "empty.json"
    path.write_text(text, encoding="utf-8")
    assert list(iter_json_array(path, chunk_size=chunk_size)) == []
    assert list(iter_match_events(path)) == []


@pytest.mark.parametrize("text", [
    "", "{}", "null", "[", "[{}", '[{"a":', '[{"a":"unterminated}]',
    "[{},]", "[{} {}]", "[{},,{}]", "[{}]garbage", "[{}],[]", "[]{}",
])
@pytest.mark.parametrize("chunk_size", [1, 7])
def test_malformed_or_truncated_arrays_reject_instead_of_returning_partial_success(tmp_path, text, chunk_size):
    path = tmp_path / "invalid.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        list(iter_json_array(path, chunk_size=chunk_size))


@pytest.mark.parametrize("element", ["1", "-42", "1.234e5", "true", "false", "null", '"text"', "[]", "[{}]"])
@pytest.mark.parametrize("chunk_size", [1, 2, 7])
def test_array_elements_must_be_objects_even_when_scalar_tokens_cross_chunks(tmp_path, element, chunk_size):
    path = tmp_path / "scalar.json"
    path.write_text("[" + element + "]", encoding="utf-8")
    with pytest.raises(ValueError, match="object"):
        list(iter_json_array(path, chunk_size=chunk_size))


@pytest.mark.parametrize("chunk_size", [0, -1, True, 1.0, None, "1"])
def test_chunk_size_requires_a_positive_exact_integer(tmp_path, chunk_size):
    path = tmp_path / "input.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="Positive integer"):
        list(iter_json_array(path, chunk_size=chunk_size))


def test_reader_yields_first_object_before_later_malformed_input_is_consumed(tmp_path):
    path = tmp_path / "later_error.json"
    path.write_text('[{"id":1},NOT_JSON]', encoding="utf-8")
    iterator = iter_json_array(path, chunk_size=1)
    assert next(iterator) == {"id": 1}
    with pytest.raises(ValueError):
        list(iterator)


def test_match_groups_preserve_source_order_without_sorting_or_cross_match_accumulation(tmp_path):
    rows = [{"matchId": 20, "id": 9}, {"matchId": 20, "id": 7},
            {"matchId": 3, "id": 6}, {"matchId": 1, "id": 2}, {"matchId": 1, "id": 4}]
    path = tmp_path / "grouped.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    groups = list(iter_match_events(path))
    assert groups == [(20, rows[:2]), (3, rows[2:3]), (1, rows[3:])]
    assert groups[0][1] is not groups[1][1]


def test_noncontiguous_repeated_match_group_rejects_explicitly(tmp_path):
    path = tmp_path / "repeated.json"
    path.write_text(json.dumps([{"matchId": 1}, {"matchId": 2}, {"matchId": 1}]), encoding="utf-8")
    with pytest.raises(ValueError, match="not contiguous"):
        list(iter_match_events(path))


@pytest.mark.parametrize("row", [{}, {"matchId": None}, {"matchId": 0}, {"matchId": -1},
                                  {"matchId": True}, {"matchId": "1"}, {"matchId": 1.0}])
def test_grouping_requires_a_positive_exact_integer_match_identity(tmp_path, row):
    path = tmp_path / "missing_match.json"
    path.write_text(json.dumps([row]), encoding="utf-8")
    with pytest.raises(ValueError, match="positive event match ID"):
        list(iter_match_events(path))


def make_archive(path, members):
    with ZipFile(path, "w") as zipped:
        for name, payload in members:
            zipped.writestr(name, payload)
    return path


@pytest.mark.parametrize("member", ["nested/events_England.json", "nested\\events_England.json",
                                   "../../events_England.json"])
def test_unpack_copies_only_selected_member_to_explicit_destination(tmp_path, member):
    archive = make_archive(tmp_path / "source.zip", [
        (member, b'[{"matchId":1}]'),
        ("nested/events_France.json", b"not selected"),
        ("../../outside.txt", b"must not be extracted"),
        ("__MACOSX/events_England.json", b"metadata ignored"),
    ])
    raw_archive = archive.read_bytes()
    destination = tmp_path / "selected.json"
    result = unpack_member(archive, "events_England.json", destination)
    assert destination.read_bytes() == b'[{"matchId":1}]'
    # ZipFile normalizes Windows separators when constructing ZipInfo.
    assert result["member"].replace("\\", "/") == member.replace("\\", "/")
    assert result["local_path"] == str(destination)
    assert result["bytes"] == len(destination.read_bytes())
    assert result["raw_content_changed"] is False
    assert archive.read_bytes() == raw_archive
    assert sorted(p.name for p in tmp_path.iterdir()) == ["selected.json", "source.zip"]


@pytest.mark.parametrize("members", [
    [("events_France.json", b"[]")],
    [("__MACOSX/events_England.json", b"[]")],
    [("events_England.json/", b"")],
    [("a/events_England.json", b"[]"), ("b/events_England.json", b"[]")],
    [("events_England.json", b"[]"), ("../events_England.json", b"[]")],
])
def test_missing_or_ambiguous_zip_basename_never_creates_destination(tmp_path, members):
    archive = make_archive(tmp_path / "source.zip", members)
    destination = tmp_path / "selected.json"
    with pytest.raises(ValueError, match="unambiguous"):
        unpack_member(archive, "events_England.json", destination)
    assert not destination.exists()


def test_duplicate_exact_zip_member_names_also_reject(tmp_path):
    with pytest.warns(UserWarning, match="Duplicate name"):
        archive = make_archive(tmp_path / "source.zip", [("events_England.json", b"[]"),
                                                         ("events_England.json", b"[]")])
    destination = tmp_path / "selected.json"
    with pytest.raises(ValueError, match="unambiguous"):
        unpack_member(archive, "events_England.json", destination)
    assert not destination.exists()


def test_identical_existing_destination_is_reused_without_overwriting(tmp_path):
    archive = make_archive(tmp_path / "source.zip", [("events_England.json", b"[]")])
    destination = tmp_path / "selected.json"
    destination.write_bytes(b"[]")
    before = destination.stat().st_mtime_ns
    unpack_member(archive, "events_England.json", destination)
    assert destination.read_bytes() == b"[]"
    assert destination.stat().st_mtime_ns == before


def test_wrong_size_existing_destination_is_rejected_without_overwriting(tmp_path):
    archive = make_archive(tmp_path / "source.zip", [("events_England.json", b"[]")])
    destination = tmp_path / "selected.json"
    destination.write_bytes(b"wrong size")
    with pytest.raises(ValueError, match="size|match|disagree|content|differs"):
        unpack_member(archive, "events_England.json", destination)
    assert destination.read_bytes() == b"wrong size"


def test_same_size_wrong_content_existing_destination_is_rejected_without_overwriting(tmp_path):
    archive = make_archive(tmp_path / "source.zip", [("events_England.json", b"abc")])
    destination = tmp_path / "selected.json"
    destination.write_bytes(b"xyz")
    with pytest.raises(ValueError):
        unpack_member(archive, "events_England.json", destination)
    assert destination.read_bytes() == b"xyz"
