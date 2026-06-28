import json, io, os, threading, socket, time, uuid, math
from flask import Flask, jsonify, render_template, send_file, request, redirect, url_for
import qrcode
from parking_graph import ParkingGraph

PORT = int(os.environ.get("PORT", 5000))

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

LOCAL_IP   = get_local_ip()
PUBLIC_URL = os.environ.get("PUBLIC_URL", f"http://{LOCAL_IP}:{PORT}")

app   = Flask(__name__)
_lock = threading.Lock()
graph = ParkingGraph("bounding_boxes.json")

# reservations: { session_id: spot_id }
_reservations: dict[str, int] = {}

def _reserved_ids():
    return set(_reservations.values())

def _find_route(exclude_ids: set[int]):
    """A* to nearest free+unreserved spot."""
    candidates = [s for s in graph.free_spots()
                  if s.id not in exclude_ids]
    if not candidates:
        return None
    best_path, best_dist, best_spot = None, math.inf, None
    for spot in candidates:
        path = graph.astar(graph.entry_id, spot.id)
        if not path:
            continue
        dist = sum(graph._dist(path[i], path[i+1]) for i in range(len(path)-1))
        if dist < best_dist:
            best_dist, best_path, best_spot = dist, path, spot
    if not best_spot:
        return None
    return {
        "spot_id":  best_spot.id,
        "distance": round(best_dist, 1),
        "waypoints": [{"x": graph.spots[n].cx, "y": graph.spots[n].cy}
                      for n in best_path if n != graph.entry_id],
    }

@app.route("/")
def index():
    return render_template("index.html", public_url=PUBLIC_URL)

@app.route("/guide")
def guide():
    sid     = request.args.get("session")
    want_new = request.args.get("new") == "1"

    if not sid:
        sid = str(uuid.uuid4())[:8]
        return redirect(url_for("guide", session=sid))

    with _lock:
        if want_new and sid in _reservations:
            old = _reservations.pop(sid)
            exclude = _reserved_ids() | {old}
        elif sid in _reservations:
            exclude = _reserved_ids()
        else:
            exclude = _reserved_ids()

        if sid not in _reservations:
            route = _find_route(exclude)
            if route:
                _reservations[sid] = route["spot_id"]
        else:
            # rebuild route for existing spot
            spot_id = _reservations[sid]
            path = graph.astar(graph.entry_id, spot_id)
            dist = sum(graph._dist(path[i], path[i+1])
                       for i in range(len(path)-1)) if path else 0
            route = {
                "spot_id":  spot_id,
                "distance": round(dist, 1),
                "waypoints": [{"x": graph.spots[n].cx, "y": graph.spots[n].cy}
                               for n in (path or []) if n != graph.entry_id],
            }

        lot = graph.to_json()

    return render_template("guide.html",
                           public_url=PUBLIC_URL,
                           session_id=sid,
                           route_json=json.dumps(route),
                           lot_json=json.dumps(lot))

@app.route("/api/state")
def api_state():
    with _lock:
        lot = graph.to_json()
    return jsonify({
        "lot":            lot,
        "reserved":       list(_reserved_ids()),
        "free_count":     lot["free_count"],
        "occupied_count": lot["occupied_count"],
        "total":          lot["total"],
    })

@app.route("/api/update", methods=["POST"])
def api_update():
    """Called by yolo_bridge.py every N frames with list of occupied spot IDs."""
    data     = request.get_json(force=True)
    occupied = set(data.get("occupied", []))
    with _lock:
        graph.update_occupancy(occupied)
    return jsonify({"ok": True, "free": len(graph.free_spots())})

@app.route("/qr")
def qr_code():
    img = qrcode.make(f"{PUBLIC_URL}/guide")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

@app.route("/qr/download")
def qr_download():
    img = qrcode.make(f"{PUBLIC_URL}/guide")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png",
                     as_attachment=True, download_name="parking_qr.png")

@app.route("/dev/seed")
def dev_seed():
    ids = [s.id for s in graph.spots.values() if s.id != graph.entry_id]
    with _lock:
        graph.update_occupancy(set(ids[len(ids)//2:]))
    return redirect(url_for("index"))

if __name__ == "__main__":
    print(f"\n🅿  Server ready")
    print(f"   Operator  →  http://{LOCAL_IP}:{PORT}")
    print(f"   Phone QR  →  http://{LOCAL_IP}:{PORT}/qr")
    print(f"   Customer  →  http://{LOCAL_IP}:{PORT}/guide\n")
    app.run(host="0.0.0.0", port=PORT, debug=False)
