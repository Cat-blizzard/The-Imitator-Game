import pytest
from data_generation_methods.r2h_pipeline.providers import GeminiClient, ProviderError


def test_explicit_gateway_bearer_does_not_change_native_defaults(monkeypatch):
    monkeypatch.setenv('TEST_RELAY_KEY','test-not-a-real-key')
    cfg={'model':'gemini-3.1-pro-preview','api_key_env':'TEST_RELAY_KEY',
         'base_url':'https://api.echoflowai.cc/v1beta','auth_scheme':'bearer'}
    relay=GeminiClient(cfg)
    assert relay._headers()['Authorization']=='Bearer test-not-a-real-key'
    assert 'x-goog-api-key' not in relay._headers()
    native=GeminiClient({'model':'gemini-3.1-pro-preview','api_key_env':'TEST_RELAY_KEY'})
    assert native._headers()['x-goog-api-key']=='test-not-a-real-key'
    assert 'Authorization' not in native._headers()
    assert 'test-not-a-real-key' not in relay.base_url


def test_arbitrary_auth_scheme_refused():
    with pytest.raises(ProviderError,match='invalid_auth_scheme'):
        GeminiClient({'model':'test','auth_scheme':'query_key'})
