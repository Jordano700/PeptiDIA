#!/usr/bin/env bash
# Launch PeptiDIA in Docker on the first available port (starting at 8501),
# so it never fails just because another app is using a port.
cd "$(dirname "$0")" || exit 1

port="${PEPTIDIA_PORT:-8501}"

port_in_use() {
  # returns success (0) if something is already listening on port $1
  (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && { exec 3>&-; return 0; }
  return 1
}

while port_in_use "$port"; do
  echo "Port $port is in use, trying $((port + 1))..."
  port=$((port + 1))
done

export PEPTIDIA_PORT="$port"
echo ""
echo "🧬 Starting PeptiDIA  ->  http://localhost:$port"
echo "   (Ctrl+C to stop; 'docker compose down' to remove the container)"
echo ""
exec docker compose up
