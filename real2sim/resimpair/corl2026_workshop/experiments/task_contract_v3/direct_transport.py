"""One HTTPS request to the same configured host without inherited proxies.

TLS verification and redirect rejection are retained. No retries or fallback
POST are performed. Public read-only probes verified this route is reachable.
"""
from urllib.request import Request, build_opener, ProxyHandler


def call(method, url, *, headers, json, timeout):
    from data_generation_methods.r2h_pipeline.providers import _NoRedirect, _json_bytes, ProviderError
    body = None if json is None else _json_bytes(json)
    request = Request(url, data=body, headers=headers, method=method)
    with build_opener(_NoRedirect(), ProxyHandler({})).open(request, timeout=timeout) as response:
        raw = response.read(128*1024*1024+1)
    if len(raw)>128*1024*1024: raise ProviderError("response_too_large")
    try: decoded = __import__("json").loads(raw)
    except (ValueError,UnicodeError): raise ProviderError("invalid_json_response") from None
    if not isinstance(decoded,dict): raise ProviderError("invalid_json_response")
    return decoded
