#!/bin/bash
# =====================================================================
# push_kaggle.sh — Push the winnex-madhava-sec benchmark(s) to Kaggle
# =====================================================================
#
# Two notebooks:
#   (default)  build/kaggle-benchmark-real   — Native C++ engine benchmark
#              compiles libmadhava_sec.so on Kaggle (g++), real dataset,
#              5-fold CV. Best proof of the NATIVE core.
#
#   --pip      build/kaggle-pip-benchmark    — Pure-Python pip benchmark
#              installs winnex-madhava-sec from PyPI, real dataset, 5-fold
#              CV via the vectorized batch API (v3.1.0). Best proof of the
#              PYPI PRODUCT alone (no C++).
#
# Prerequisites:
#   1. Kaggle API token with KERNEL WRITE scope:
#      https://www.kaggle.com/settings -> API -> Create New Token
#      (the default token is datasets-only; kernels push needs the
#      "Create a new token" one with kernel permissions, or re-auth)
#   2. kaggle CLI:  pip install kaggle
#   3. Credentials at ~/.kaggle/kaggle.json
#
# Usage:
#   ./push_kaggle.sh              # native C++ benchmark
#   ./push_kaggle.sh --pip        # pip-only benchmark
#   KAGGLE_API_TOKEN=<token> ./push_kaggle.sh [--pip]   # bearer auth
# =====================================================================
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

MODE="${1:-native}"
if [ "$MODE" = "--pip" ]; then
    BUILD_DIR="../build/kaggle-pip-benchmark"
    KERNEL_ID="kleniopadilha/winnex-madhava-sec-pip-benchmark"
    echo "=== Building the pip-only benchmark notebook ==="
    python3 deploy_pip_benchmark.py --build-only
else
    BUILD_DIR="../build/kaggle-benchmark-real"
    KERNEL_ID="kleniopadilha/winnex-madhava-sec-benchmark-real"
    echo "=== Building the native C++ benchmark notebook ==="
    python3 deploy_kaggle_benchmark.py --build-only
fi

echo ""
echo "=== Testing Kaggle auth ==="
kaggle datasets list -s "test" -p 1 >/dev/null 2>&1 || {
    echo "ERROR: Kaggle datasets API not authenticated."
    echo "Fix ~/.kaggle/kaggle.json and retry."
    exit 1
}
echo "  datasets API: OK"

echo ""
echo "=== Pushing kernel ($KERNEL_ID) ==="
if [ -n "$KAGGLE_API_TOKEN" ]; then
    echo "  using KAGGLE_API_TOKEN (bearer)"
    KAGGLE_API_TOKEN="$KAGGLE_API_TOKEN" python3 - <<PY
import os, kagglesdk.kaggle_http_client as khc
TOKEN = os.environ["KAGGLE_API_TOKEN"]
def patched(self):
    if self._signed_in is not None: return
    self._session.auth = khc.KaggleHttpClient.BearerAuth(TOKEN)
    self._signed_in = True
khc.KaggleHttpClient._try_fill_auth = patched
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
r = api.kernels_push("$BUILD_DIR")
print("Push OK:", r.url, "Version:", r.version_number)
PY
else
    echo "  using ~/.kaggle/kaggle.json (username+key)"
    kaggle kernels push -p "$BUILD_DIR"
fi

echo ""
echo "Done. View at:"
echo "  https://www.kaggle.com/code/$KERNEL_ID"
