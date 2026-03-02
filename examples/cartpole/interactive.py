"""Generate a self-contained interactive HTML demo of the trained CartPole PPO agent.

The script loads the trained checkpoint, extracts actor weights, and embeds them
into an HTML file with CartPole physics, neural-net inference, and canvas rendering
all implemented in JavaScript.  No server needed — open the HTML in any browser.

Usage:
    cd examples/cartpole
    python interactive.py
    # then open logs/cartpole_ppo/interactive.html
"""

import json
import os

import torch

CHECKPOINT_PATH = os.path.join("logs", "cartpole_ppo", "model_299.pt")
OUTPUT_PATH = os.path.join("logs", "cartpole_ppo", "interactive.html")


def extract_actor_weights(checkpoint_path: str) -> dict:
    """Load checkpoint and return actor MLP weights as JSON-serializable lists."""
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    sd = ckpt["actor_state_dict"]

    weights = {}
    for idx in (0, 2, 4):
        weights[f"w{idx}"] = sd[f"mlp.{idx}.weight"].tolist()
        weights[f"b{idx}"] = sd[f"mlp.{idx}.bias"].tolist()
    return weights


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Interactive CartPole — PPO Agent</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#1a1a2e;color:#e0e0e0;font-family:'Segoe UI',system-ui,sans-serif;display:flex;flex-direction:column;align-items:center;min-height:100vh;padding:20px}
h1{font-size:1.4rem;margin-bottom:8px;color:#e94560}
#info{font-size:.85rem;color:#888;margin-bottom:12px}
canvas{background:#0f3460;border-radius:8px;border:2px solid #16213e}
#controls{margin-top:14px;display:flex;gap:10px;flex-wrap:wrap;justify-content:center}
button{padding:8px 18px;border:none;border-radius:6px;font-size:.9rem;cursor:pointer;font-weight:600;transition:background .15s}
#btnDisturb{background:#e94560;color:#fff}#btnDisturb:hover{background:#c73650}
#btnToggle{background:#0f3460;color:#e0e0e0;border:1px solid #e94560}#btnToggle:hover{background:#16213e}
#btnReset{background:#533483;color:#fff}#btnReset:hover{background:#3e2563}
#sliderWrap{display:flex;align-items:center;gap:6px;font-size:.85rem}
#speedSlider{width:100px}
#readout{margin-top:10px;font-family:'Courier New',monospace;font-size:.82rem;line-height:1.6;text-align:left;background:#16213e;padding:10px 16px;border-radius:6px;min-width:340px}
.failed-flash{position:absolute;font-size:2.5rem;font-weight:900;color:#e94560;pointer-events:none;animation:flashOut .8s forwards}
@keyframes flashOut{0%{opacity:1;transform:scale(1)}100%{opacity:0;transform:scale(1.4)}}
#wrapper{position:relative;display:inline-block}
</style>
</head>
<body>
<h1>Interactive CartPole — PPO Agent</h1>
<p id="info">Trained with rsl_rl PPO &middot; Everything runs locally in your browser</p>
<div id="wrapper">
<canvas id="cv" width="600" height="300"></canvas>
</div>
<div id="controls">
<button id="btnDisturb">Disturbance</button>
<button id="btnToggle">Controller: ON</button>
<button id="btnReset">Reset</button>
<div id="sliderWrap">
<label for="speedSlider">Speed:</label>
<input id="speedSlider" type="range" min="1" max="10" value="1">
<span id="speedVal">1x</span>
</div>
</div>
<div id="readout"></div>

<script>
"use strict";

// ── Neural network weights (injected from Python) ──
const W = __WEIGHTS_JSON__;

// ── CartPole physics constants (matching gymnasium CartPole-v1) ──
const GRAVITY   = 9.8;
const MASSCART  = 1.0;
const MASSPOLE  = 0.1;
const TOTAL_MASS = MASSCART + MASSPOLE;
const LENGTH    = 0.5;   // half-pole length
const POLEMASS_LENGTH = MASSPOLE * LENGTH;
const FORCE_MAG = 10.0;
const TAU       = 0.02;  // seconds per step
const THETA_THRESHOLD = 12 * Math.PI / 180;
const X_THRESHOLD     = 2.4;

// ── State ──
let state = [0, 0, 0, 0]; // x, x_dot, theta, theta_dot
let step  = 0;
let controllerOn = true;

// ── ELU activation ──
function elu(x) { return x > 0 ? x : Math.exp(x) - 1; }

// ── Forward pass: 3-layer MLP with ELU activations on hidden layers ──
function forward(obs) {
    // Layer 0: Linear + ELU
    let h = new Array(64);
    for (let i = 0; i < 64; i++) {
        let s = W.b0[i];
        for (let j = 0; j < 4; j++) s += W.w0[i][j] * obs[j];
        h[i] = elu(s);
    }
    // Layer 2: Linear + ELU
    let h2 = new Array(64);
    for (let i = 0; i < 64; i++) {
        let s = W.b2[i];
        for (let j = 0; j < 64; j++) s += W.w2[i][j] * h[j];
        h2[i] = elu(s);
    }
    // Layer 4: Linear (output)
    let s = W.b4[0];
    for (let j = 0; j < 64; j++) s += W.w4[0][j] * h2[j];
    return s;
}

// ── Physics step (Euler integration, matches gymnasium) ──
function physicsStep(action) {
    let [x, xd, th, thd] = state;
    const force = action === 1 ? FORCE_MAG : -FORCE_MAG;
    const costh = Math.cos(th);
    const sinth = Math.sin(th);
    const temp = (force + POLEMASS_LENGTH * thd * thd * sinth) / TOTAL_MASS;
    const thacc = (GRAVITY * sinth - costh * temp) /
                  (LENGTH * (4/3 - MASSPOLE * costh * costh / TOTAL_MASS));
    const xacc = temp - POLEMASS_LENGTH * thacc * costh / TOTAL_MASS;
    x  += TAU * xd;
    xd += TAU * xacc;
    th += TAU * thd;
    thd += TAU * thacc;
    state = [x, xd, th, thd];
    step++;
}

// ── Canvas drawing ──
const cv  = document.getElementById("cv");
const ctx = cv.getContext("2d");
const W_PX = cv.width, H_PX = cv.height;
const SCALE = W_PX / (X_THRESHOLD * 2 + 1); // pixels per metre
const GROUND_Y = H_PX * 0.7;

function worldToCanvas(wx) { return W_PX / 2 + wx * SCALE; }

function draw() {
    ctx.clearRect(0, 0, W_PX, H_PX);

    // Track
    ctx.strokeStyle = "#e0e0e040";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, GROUND_Y);
    ctx.lineTo(W_PX, GROUND_Y);
    ctx.stroke();

    // Bounds markers
    ctx.strokeStyle = "#e9456080";
    ctx.setLineDash([6, 4]);
    for (const bx of [-X_THRESHOLD, X_THRESHOLD]) {
        const px = worldToCanvas(bx);
        ctx.beginPath(); ctx.moveTo(px, GROUND_Y - 80); ctx.lineTo(px, GROUND_Y + 10); ctx.stroke();
    }
    ctx.setLineDash([]);

    const cx = worldToCanvas(state[0]);
    const cy = GROUND_Y;

    // Cart
    const cw = 50, ch = 26;
    ctx.fillStyle = "#16213e";
    ctx.strokeStyle = "#e94560";
    ctx.lineWidth = 2;
    ctx.fillRect(cx - cw/2, cy - ch/2, cw, ch);
    ctx.strokeRect(cx - cw/2, cy - ch/2, cw, ch);

    // Wheels
    ctx.fillStyle = "#e94560";
    for (const dx of [-14, 14]) {
        ctx.beginPath(); ctx.arc(cx + dx, cy + ch/2, 5, 0, Math.PI * 2); ctx.fill();
    }

    // Pole
    const poleLen = LENGTH * 2 * SCALE;
    const px = cx + Math.sin(state[2]) * poleLen;
    const py = cy - Math.cos(state[2]) * poleLen;
    ctx.strokeStyle = "#e0c040";
    ctx.lineWidth = 6;
    ctx.lineCap = "round";
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(px, py); ctx.stroke();

    // Pole tip
    ctx.fillStyle = "#e0c040";
    ctx.beginPath(); ctx.arc(px, py, 5, 0, Math.PI * 2); ctx.fill();

    // Pivot dot
    ctx.fillStyle = "#e94560";
    ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI * 2); ctx.fill();

    // Controller status badge
    ctx.font = "bold 13px 'Segoe UI', sans-serif";
    ctx.fillStyle = controllerOn ? "#4ecca3" : "#e94560";
    ctx.fillText(controllerOn ? "CONTROLLER ON" : "CONTROLLER OFF", 10, 22);

}

// ── Readout panel ──
const readoutEl = document.getElementById("readout");
function updateReadout() {
    const deg = (state[2] * 180 / Math.PI).toFixed(2);
    readoutEl.innerHTML =
        `x: ${state[0].toFixed(4)}  &nbsp; x_dot: ${state[1].toFixed(4)}<br>` +
        `theta: ${deg}°  &nbsp; theta_dot: ${state[3].toFixed(4)}<br>` +
        `step: ${step}  &nbsp; controller: ${controllerOn ? "ON" : "OFF"}`;
}

// ── Controls ──
const btnDisturb = document.getElementById("btnDisturb");
const btnToggle  = document.getElementById("btnToggle");
const btnReset   = document.getElementById("btnReset");
const speedSlider = document.getElementById("speedSlider");
const speedVal   = document.getElementById("speedVal");

btnDisturb.addEventListener("click", () => {
    const impulse = (Math.random() > 0.5 ? 1 : -1) * (1.5 + Math.random());
    state[3] += impulse;
});

btnToggle.addEventListener("click", () => {
    controllerOn = !controllerOn;
    btnToggle.textContent = "Controller: " + (controllerOn ? "ON" : "OFF");
    btnToggle.style.background = controllerOn ? "#0f3460" : "#e94560";
});

btnReset.addEventListener("click", resetState);

speedSlider.addEventListener("input", () => {
    speedVal.textContent = speedSlider.value + "x";
});

function resetState() {
    state = [0, 0, 0, 0];
    step = 0;
}

// ── Main loop ──
function tick() {
    const stepsPerFrame = parseInt(speedSlider.value);
    for (let i = 0; i < stepsPerFrame; i++) {
        let action;
        if (controllerOn) {
            const out = forward(state);
            action = out > 0 ? 1 : 0;
        } else {
            action = 0; // no force (push left = -10N, but effectively idle visually)
        }
        physicsStep(action);
    }
    draw();
    updateReadout();
    requestAnimationFrame(tick);
}

requestAnimationFrame(tick);
</script>
</body>
</html>
"""


def main():
    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    weights = extract_actor_weights(CHECKPOINT_PATH)
    weights_json = json.dumps(weights)

    html = HTML_TEMPLATE.replace("__WEIGHTS_JSON__", weights_json)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write(html)

    print(f"Saved interactive demo to {OUTPUT_PATH}")
    print("Open it in a browser to see the trained PPO agent in action.")


if __name__ == "__main__":
    main()
