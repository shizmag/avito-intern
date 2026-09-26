from avito_candidate_generation.end_to_end import run_end_to_end

def test_end_to_end_manifest():
    assert run_end_to_end(".")["status"] == "PASS"
