from avito_candidate_generation.verify import verify_repository

def test_repository_verifier():
    result=verify_repository(".")
    assert result["status"] == "PASS"
