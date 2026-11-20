# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Compare configured and direct HTTPS transport without changing environment/cache."""

import argparse
import datetime
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tarfile
import time
import traceback
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener, getproxies, proxy_bypass

URL = ('https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/'
       'flagtune-xgb-manifest-v1.0.0.tar.gz')
MAX_BYTES = 16 * 1024 * 1024


class HTTPSOnlyRedirect(HTTPRedirectHandler):

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parts = urlsplit(newurl)
        if parts.scheme.lower() != 'https' or not parts.netloc:
            raise HTTPError(newurl, code, 'FlagTune redirects must remain HTTPS', headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def redact(text):
    # Never log configured proxy URLs or environment values. Also redact URL
    # userinfo if an underlying exception unexpectedly includes it.
    return re.sub(r'([a-zA-Z][a-zA-Z0-9+.-]*://)[^/@\s]+@', r'\1<redacted>@', text)


def probe(direct):
    started = time.monotonic()
    row = {
        'route': 'explicit_direct' if direct else 'as_configured', 'source_url': URL, 'method': 'GET',
        'timeout_seconds': 20, 'environment_changed': False, 'cache_written': False
    }
    handlers = [HTTPSOnlyRedirect()]
    if direct:
        handlers.insert(0, ProxyHandler({}))
    try:
        request = Request(URL, headers={'User-Agent': 'FlagTune/manifest'})
        with build_opener(*handlers).open(request, timeout=20) as response:
            if response.status != 200:
                raise ValueError('manifest GET did not return HTTP 200')
            payload = response.read(MAX_BYTES + 1)
            if len(payload) > MAX_BYTES:
                raise ValueError('remote Manifest archive is too large')
            if payload[:2] != b'\x1f\x8b':
                raise ValueError('response is not a gzip archive')
            with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
                candidates = [member for member in archive.getmembers() if member.name == 'manifest.json']
                if len(candidates) != 1 or not candidates[0].isfile() or candidates[0].size > 4 * 1024 * 1024:
                    raise ValueError('response has no unique bounded manifest.json')
                manifest_bytes = archive.extractfile(candidates[0]).read()
                manifest = json.loads(manifest_bytes)
                if manifest.get('schema_version') != 1 or not isinstance(manifest.get('packages'), dict):
                    raise ValueError('response does not contain the expected Manifest schema')
            row.update({
                'status': int(response.status), 'final_url': response.geturl(), 'bytes': len(payload), 'sha256':
                hashlib.sha256(payload).hexdigest(), 'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
                'ok': True
            })
    except Exception:
        row.update({'ok': False, 'traceback': redact(traceback.format_exc())})
    row['duration_seconds'] = time.monotonic() - started
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    host = urlsplit(URL).hostname
    report = {
        'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'sys_executable': sys.executable, 'source_url':
        URL, 'proxy_configured_for_https': bool(getproxies().get('https')), 'configured_proxy_bypass_for_host':
        proxy_bypass(host), 'trials': [probe(False), probe(True)], 'diagnostic_only': True, 'models_selected_or_loaded':
        False, 'environment_changed': False, 'cache_written': False
    }
    report['both_payloads_identical'] = (all(r['ok'] for r in report['trials'])
                                         and report['trials'][0]['sha256'] == report['trials'][1]['sha256'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
