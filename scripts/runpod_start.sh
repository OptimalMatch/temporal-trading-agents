#!/bin/bash
#############################################################################
# RunPod All-in-One Management Script
#############################################################################
# Manages MongoDB, Backend, and Frontend (nginx) services
#
# Usage:
#   ./scripts/runpod_start.sh [start|stop|restart|status|logs]
#############################################################################

set -e

ACTION=${1:-start}

# Get instance IP
INSTANCE_IP=$(hostname -I | awk '{print $1}')

case $ACTION in
    start)
        echo "🚀 Starting all services..."

        # Start MongoDB
        if command -v mongod &> /dev/null; then
            if sudo systemctl is-active --quiet mongod; then
                echo "   MongoDB already running"
            else
                sudo systemctl start mongod
                echo "   ✅ MongoDB started"
            fi
        fi

        # Start nginx
        if command -v nginx &> /dev/null; then
            if sudo systemctl is-active --quiet nginx; then
                echo "   nginx already running"
            else
                sudo systemctl start nginx
                echo "   ✅ nginx started"
            fi
        fi

        # Start backend
        source venv/bin/activate 2>/dev/null || true
        export $(grep -v '^#' .env | xargs) 2>/dev/null || true

        # Kill any existing instances
        pkill -f "uvicorn backend.main:app" || true

        # Start in background
        nohup uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} > /tmp/temporal-trading.log 2>&1 &

        sleep 2
        PID=$(pgrep -f "uvicorn backend.main:app" || echo "")

        if [ -z "$PID" ]; then
            echo "   ❌ Failed to start backend"
            echo "   Check logs: tail -f /tmp/temporal-trading.log"
            exit 1
        else
            echo "   ✅ Backend started (PID: $PID)"
        fi

        echo ""
        echo "✅ All services started!"
        echo "   Frontend: http://${INSTANCE_IP}/"
        echo "   API Docs: http://${INSTANCE_IP}/docs"
        echo "   Logs: tail -f /tmp/temporal-trading.log"
        ;;

    stop)
        echo "🛑 Stopping all services..."

        # Stop backend
        pkill -f "uvicorn backend.main:app" || true
        echo "   ✅ Backend stopped"

        # Stop nginx
        if command -v nginx &> /dev/null; then
            sudo systemctl stop nginx || true
            echo "   ✅ nginx stopped"
        fi

        # Stop MongoDB (optional - usually keep running)
        # sudo systemctl stop mongod || true
        # echo "   ✅ MongoDB stopped"

        echo "✅ Services stopped (MongoDB kept running)"
        ;;

    restart)
        echo "🔄 Restarting all services..."
        $0 stop
        sleep 2
        $0 start
        ;;

    status)
        echo "📊 Service Status:"
        echo ""

        # Check MongoDB
        if command -v mongod &> /dev/null; then
            if sudo systemctl is-active --quiet mongod; then
                echo "   ✅ MongoDB: Running"
            else
                echo "   ❌ MongoDB: Not running"
            fi
        else
            echo "   ⚠️  MongoDB: Not installed"
        fi

        # Check nginx
        if command -v nginx &> /dev/null; then
            if sudo systemctl is-active --quiet nginx; then
                echo "   ✅ nginx: Running"
            else
                echo "   ❌ nginx: Not running"
            fi
        else
            echo "   ⚠️  nginx: Not installed"
        fi

        # Check backend
        PID=$(pgrep -f "uvicorn backend.main:app" || echo "")
        if [ -z "$PID" ]; then
            echo "   ❌ Backend: Not running"
            exit 1
        else
            echo "   ✅ Backend: Running (PID: $PID)"

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
        echo "All-in-One Service Management:"
        echo "  start          - Start all services (MongoDB, nginx, backend)"
        echo "  stop           - Stop all services (keeps MongoDB running)"
        echo "  restart        - Restart all services"
        echo "  status         - Show status of all services + GPU"
        echo "  logs           - Show and follow backend logs"
        echo ""
        echo "Systemd commands (if installed):"
        echo "  systemd-start  - Start backend via systemd"
        echo "  systemd-stop   - Stop backend via systemd"
        echo "  systemd-status - Check backend systemd status"
        echo "  systemd-logs   - Follow backend systemd logs"
        echo ""
        echo "Services managed:"
        echo "  - MongoDB (database)"
        echo "  - nginx (frontend + proxy)"
        echo "  - Backend API (FastAPI + PyTorch)"
        exit 1
        ;;
esac
