#!/bin/bash
#############################################################################
# RunPod Quick Start Script
#############################################################################
# Use this to quickly start/stop/restart the backend after initial deployment
#
# Usage:
#   ./scripts/runpod_start.sh [start|stop|restart|status|logs]
#############################################################################

set -e

ACTION=${1:-start}

case $ACTION in
    start)
        echo "🚀 Starting Temporal Trading backend..."
        source venv/bin/activate
        export $(grep -v '^#' .env | xargs) 2>/dev/null || true

        # Kill any existing instances
        pkill -f "uvicorn backend.main:app" || true

        # Start in background
        nohup uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} > /tmp/temporal-trading.log 2>&1 &

        sleep 2
        PID=$(pgrep -f "uvicorn backend.main:app" || echo "")

        if [ -z "$PID" ]; then
            echo "❌ Failed to start backend"
            echo "Check logs: tail -f /tmp/temporal-trading.log"
            exit 1
        else
            echo "✅ Backend started (PID: $PID)"
            echo "📊 Health check: http://localhost:${PORT:-8000}/health"
            echo "📝 Logs: tail -f /tmp/temporal-trading.log"
        fi
        ;;

    stop)
        echo "🛑 Stopping Temporal Trading backend..."
        pkill -f "uvicorn backend.main:app" || true
        echo "✅ Backend stopped"
        ;;

    restart)
        echo "🔄 Restarting Temporal Trading backend..."
        $0 stop
        sleep 2
        $0 start
        ;;

    status)
        PID=$(pgrep -f "uvicorn backend.main:app" || echo "")
        if [ -z "$PID" ]; then
            echo "❌ Backend not running"
            exit 1
        else
            echo "✅ Backend running (PID: $PID)"

            # Check GPU usage
            if command -v nvidia-smi &> /dev/null; then
                echo ""
                echo "🎮 GPU Status:"
                nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
            fi

            # Try to hit health endpoint
            PORT=${PORT:-8000}
            if command -v curl &> /dev/null; then
                echo ""
                echo "🏥 Health Check:"
                curl -s http://localhost:$PORT/health 2>/dev/null | python3 -m json.tool || echo "Unable to reach health endpoint"
            fi
        fi
        ;;

    logs)
        echo "📝 Showing logs (Ctrl+C to exit)..."
        tail -f /tmp/temporal-trading.log
        ;;

    systemd-start)
        echo "🚀 Starting via systemd..."
        sudo systemctl start temporal-trading
        sleep 2
        sudo systemctl status temporal-trading --no-pager
        ;;

    systemd-stop)
        echo "🛑 Stopping via systemd..."
        sudo systemctl stop temporal-trading
        ;;

    systemd-status)
        sudo systemctl status temporal-trading --no-pager
        ;;

    systemd-logs)
        echo "📝 Showing systemd logs (Ctrl+C to exit)..."
        sudo journalctl -u temporal-trading -f
        ;;

    *)
        echo "Usage: $0 [start|stop|restart|status|logs]"
        echo ""
        echo "Commands:"
        echo "  start          - Start backend in background"
        echo "  stop           - Stop backend"
        echo "  restart        - Restart backend"
        echo "  status         - Show backend status"
        echo "  logs           - Show and follow logs"
        echo ""
        echo "Systemd commands (if installed):"
        echo "  systemd-start  - Start via systemd"
        echo "  systemd-stop   - Stop via systemd"
        echo "  systemd-status - Check systemd status"
        echo "  systemd-logs   - Follow systemd logs"
        exit 1
        ;;
esac
