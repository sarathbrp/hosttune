#!/usr/bin/env bash
# Run MLflow tracking server on System2 in a dedicated shell.
# Usage: ./scripts/run_mlflow.sh [system2_host]
#
# Opens an SSH session that runs the MLflow server on the remote host.
# UI is accessible at http://<host>:5000 from your browser.

set -euo pipefail

HOST="${1:-d21-h24-000-r650.rdu2.scalelab.redhat.com}"
PORT=5000

echo "Starting MLflow tracking server on ${HOST}:${PORT}"
echo "UI: http://${HOST}:${PORT}"
echo "Press Ctrl+C to stop."
echo ""

ssh -t "root@${HOST}" << 'ENDSSH'
set -euo pipefail

VENV="/opt/hosttune/venv"
BACKEND_STORE="sqlite:////opt/hosttune/artifacts/mlflow.db"
ARTIFACT_ROOT="/opt/hosttune/artifacts/mlflow-artifacts"
PORT=5000

mkdir -p /opt/hosttune/artifacts/mlflow-artifacts

if ! "${VENV}/bin/python" -c "import mlflow" 2>/dev/null; then
    echo "Installing mlflow..."
    "${VENV}/bin/pip" install mlflow --quiet
fi

echo "MLflow version: $("${VENV}/bin/mlflow" --version)"
echo ""

"${VENV}/bin/mlflow" server \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --backend-store-uri "${BACKEND_STORE}" \
    --default-artifact-root "${ARTIFACT_ROOT}" \
    --allowed-hosts "*"
ENDSSH
