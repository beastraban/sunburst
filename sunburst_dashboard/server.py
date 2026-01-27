#!/usr/bin/env python3
"""
SunBURST Dashboard Server
=========================
Local WebSocket server for real-time SunBURST monitoring.

Usage:
    # Terminal 1: Start dashboard
    python -m sunburst_dashboard.server
    
    # Terminal 2: Run SunBURST with dashboard enabled
    from sunburst_dashboard import DashboardClient
    dashboard = DashboardClient()
    result = pipeline.compute_evidence(log_L, bounds, callbacks=dashboard)

Or standalone:
    python server.py
    # Then open http://localhost:8080 in browser
"""

import asyncio
import json
import webbrowser
import http.server
import socketserver
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Set
import os

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    print("WARNING: 'websockets' not installed. Run: pip install websockets")

# Configuration
HTTP_PORT = 8080
WS_PORT = 8765
DASHBOARD_HTML = Path(__file__).parent / "dashboard.html"

# Global state
connected_clients: Set = set()
run_history: list = []
current_run: Optional[Dict[str, Any]] = None


class DashboardMessage:
    """Message types for dashboard communication."""
    
    # Pipeline status
    PIPELINE_START = "pipeline_start"
    PIPELINE_END = "pipeline_end"
    MODULE_START = "module_start"
    MODULE_END = "module_end"
    
    # Progress updates
    PROGRESS = "progress"
    RAYS_UPDATE = "rays_update"
    PEAKS_UPDATE = "peaks_update"
    
    # Results
    RESULT = "result"
    ERROR = "error"
    
    # History
    HISTORY = "history"
    
    @staticmethod
    def create(msg_type: str, data: dict) -> str:
        """Create a JSON message."""
        return json.dumps({
            "type": msg_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        })


async def broadcast(message: str):
    """Send message to all connected clients."""
    if connected_clients:
        await asyncio.gather(
            *[client.send(message) for client in connected_clients],
            return_exceptions=True
        )


async def handle_client(websocket):
    """Handle a WebSocket client connection."""
    connected_clients.add(websocket)
    print(f"[Dashboard] Client connected. Total: {len(connected_clients)}")
    
    try:
        # Send current state on connect
        if run_history:
            await websocket.send(DashboardMessage.create(
                DashboardMessage.HISTORY,
                {"runs": run_history[-20:]}  # Last 20 runs
            ))
        
        if current_run:
            await websocket.send(DashboardMessage.create(
                DashboardMessage.PIPELINE_START,
                current_run
            ))
        
        # Keep connection alive and handle incoming messages
        async for message in websocket:
            # Handle any client messages (e.g., requests for history)
            try:
                data = json.loads(message)
                if data.get("type") == "get_history":
                    await websocket.send(DashboardMessage.create(
                        DashboardMessage.HISTORY,
                        {"runs": run_history[-20:]}
                    ))
            except json.JSONDecodeError:
                pass
                
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_clients.discard(websocket)
        print(f"[Dashboard] Client disconnected. Total: {len(connected_clients)}")


class DashboardClient:
    """
    Client for sending updates to the dashboard.
    Use as callbacks in SunBURST pipeline.
    
    Example:
        dashboard = DashboardClient()
        
        # In your pipeline code:
        dashboard.on_pipeline_start(dim=64, bounds=bounds)
        dashboard.on_module_start("CarryTiger")
        dashboard.on_progress(rays_cast=100, total_rays=1000)
        dashboard.on_peaks_found(peaks, likelihoods)
        dashboard.on_module_end("CarryTiger", time=0.5)
        dashboard.on_result(log_evidence=-92.31, n_peaks=3, time=1.2)
    """
    
    def __init__(self, host: str = "localhost", port: int = WS_PORT):
        self.uri = f"ws://{host}:{port}"
        self._loop = None
        self._connection = None
        self._run_id = None
        
    async def _connect(self):
        """Establish WebSocket connection."""
        if not HAS_WEBSOCKETS:
            return
        try:
            self._connection = await websockets.connect(self.uri)
        except Exception as e:
            print(f"[Dashboard] Could not connect: {e}")
            self._connection = None
    
    async def _send(self, message: str):
        """Send message to server."""
        if self._connection:
            try:
                await self._connection.send(message)
            except Exception:
                self._connection = None
    
    def _sync_send(self, msg_type: str, data: dict):
        """Synchronous send for use in non-async code."""
        if not HAS_WEBSOCKETS:
            return
            
        message = DashboardMessage.create(msg_type, data)
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self._send(message))
            else:
                loop.run_until_complete(self._send(message))
        except RuntimeError:
            # Create new loop if none exists
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._connect())
            loop.run_until_complete(self._send(message))
    
    def on_pipeline_start(self, dim: int, bounds=None, config: dict = None):
        """Called when pipeline starts."""
        self._run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._sync_send(DashboardMessage.PIPELINE_START, {
            "run_id": self._run_id,
            "dim": dim,
            "bounds": bounds.tolist() if hasattr(bounds, 'tolist') else bounds,
            "config": config or {}
        })
    
    def on_pipeline_end(self, success: bool = True):
        """Called when pipeline ends."""
        self._sync_send(DashboardMessage.PIPELINE_END, {
            "run_id": self._run_id,
            "success": success
        })
    
    def on_module_start(self, module_name: str):
        """Called when a module starts."""
        self._sync_send(DashboardMessage.MODULE_START, {
            "run_id": self._run_id,
            "module": module_name
        })
    
    def on_module_end(self, module_name: str, time_seconds: float, **extra):
        """Called when a module ends."""
        self._sync_send(DashboardMessage.MODULE_END, {
            "run_id": self._run_id,
            "module": module_name,
            "time": time_seconds,
            **extra
        })
    
    def on_progress(self, **kwargs):
        """Called for progress updates."""
        self._sync_send(DashboardMessage.PROGRESS, {
            "run_id": self._run_id,
            **kwargs
        })
    
    def on_rays_update(self, rays_cast: int, total_rays: int, 
                       ray_samples: list = None):
        """Called when rays are cast."""
        data = {
            "run_id": self._run_id,
            "rays_cast": rays_cast,
            "total_rays": total_rays
        }
        if ray_samples is not None:
            # Send subset for visualization (first 2 dims)
            data["samples"] = ray_samples[:1000] if len(ray_samples) > 1000 else ray_samples
        self._sync_send(DashboardMessage.RAYS_UPDATE, data)
    
    def on_peaks_found(self, peaks, likelihoods=None, widths=None):
        """Called when peaks are found/updated."""
        peaks_list = peaks.tolist() if hasattr(peaks, 'tolist') else peaks
        data = {
            "run_id": self._run_id,
            "n_peaks": len(peaks_list),
            "peaks_2d": [[p[0], p[1]] if len(p) >= 2 else [p[0], 0] 
                        for p in peaks_list]  # First 2 dims for visualization
        }
        if likelihoods is not None:
            ll = likelihoods.tolist() if hasattr(likelihoods, 'tolist') else likelihoods
            data["likelihoods"] = ll
        if widths is not None:
            w = widths.tolist() if hasattr(widths, 'tolist') else widths
            data["widths"] = w
        self._sync_send(DashboardMessage.PEAKS_UPDATE, data)
    
    def on_result(self, log_evidence: float, n_peaks: int, 
                  time_seconds: float, **extra):
        """Called with final result."""
        self._sync_send(DashboardMessage.RESULT, {
            "run_id": self._run_id,
            "log_evidence": log_evidence,
            "n_peaks": n_peaks,
            "time": time_seconds,
            **extra
        })
    
    def on_error(self, error_message: str, module: str = None):
        """Called on error."""
        self._sync_send(DashboardMessage.ERROR, {
            "run_id": self._run_id,
            "error": error_message,
            "module": module
        })


# Internal message handler for server
async def handle_internal_message(message: str):
    """Handle messages from SunBURST client."""
    global current_run, run_history
    
    try:
        data = json.loads(message)
        msg_type = data.get("type")
        msg_data = data.get("data", {})
        
        if msg_type == DashboardMessage.PIPELINE_START:
            current_run = msg_data
        elif msg_type == DashboardMessage.PIPELINE_END:
            current_run = None
        elif msg_type == DashboardMessage.RESULT:
            run_history.append(msg_data)
            if len(run_history) > 100:
                run_history = run_history[-100:]
        
        # Broadcast to all dashboard clients
        await broadcast(message)
        
    except json.JSONDecodeError:
        pass


async def internal_client_handler(websocket):
    """Handle internal connections from SunBURST."""
    print("[Dashboard] SunBURST client connected")
    try:
        async for message in websocket:
            await handle_internal_message(message)
    except websockets.exceptions.ConnectionClosed:
        pass
    print("[Dashboard] SunBURST client disconnected")


def serve_http():
    """Serve the dashboard HTML file."""
    
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(DASHBOARD_HTML.parent), **kwargs)
        
        def do_GET(self):
            if self.path == "/" or self.path == "/index.html":
                self.path = "/dashboard.html"
            return super().do_GET()
        
        def log_message(self, format, *args):
            pass  # Suppress HTTP logs
    
    with socketserver.TCPServer(("", HTTP_PORT), Handler) as httpd:
        print(f"[Dashboard] HTTP server at http://localhost:{HTTP_PORT}")
        httpd.serve_forever()


async def main():
    """Run the dashboard server."""
    if not HAS_WEBSOCKETS:
        print("ERROR: Please install websockets: pip install websockets")
        return
    
    # Check if dashboard.html exists
    if not DASHBOARD_HTML.exists():
        print(f"ERROR: {DASHBOARD_HTML} not found")
        print("Make sure dashboard.html is in the same directory as server.py")
        return
    
    # Start HTTP server in background thread
    http_thread = threading.Thread(target=serve_http, daemon=True)
    http_thread.start()
    
    # Start WebSocket server
    print(f"[Dashboard] WebSocket server at ws://localhost:{WS_PORT}")
    
    async with websockets.serve(handle_client, "localhost", WS_PORT):
        print(f"\n{'='*60}")
        print("  SunBURST Dashboard Running!")
        print(f"  Open http://localhost:{HTTP_PORT} in your browser")
        print(f"{'='*60}\n")
        
        # Open browser
        webbrowser.open(f"http://localhost:{HTTP_PORT}")
        
        # Keep running
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Dashboard] Shutting down...")
