#!/usr/bin/env python3
"""Deterministic generator for tests/fixtures/ops/*.json (schemas: tests/fixtures/README.md).

    make_ops.py            write the fixtures
    make_ops.py --check    regenerate in memory, fail unless byte-identical to the files on disk

All expected values come from the float64 oracle in ref_qwen35.py. Inputs (activations) are
rounded to f32 and weights to bf16, so a C test reads exactly the numbers the oracle used;
expected outputs are float64 (compare an f32 engine with a tolerance, see the README).
"""

import argparse
import json
import math
import os
import random
import struct
import sys

import quant_ref
import ref_qwen35 as ref
import safetensors as st

SEED = 20261113
EPS = 1e-6


def f32(v):
    return st.round_f32(v)


class R:
    def __init__(self, seed):
        self.r = random.Random(seed)

    def act(self, n, std=1.0, mean=0.0):
        return [f32(self.r.gauss(mean, std)) for _ in range(n)]

    def wt(self, n, std):
        return [st.round_bf16(self.r.gauss(0.0, std)) for _ in range(n)]

    def mat(self, rows, cols, std):
        return [self.wt(cols, std) for _ in range(rows)]


def rmsnorm(rng):
    cases = []
    for n in (1, 7, 16, 33, 64, 256):
        x, w = rng.act(n, 2.0), rng.wt(n, 0.3)
        cases.append({"x": x, "w": w, "y": ref.rmsnorm_zero_centred(x, w, EPS)})
    cases.append({"x": [0.0] * 8, "w": rng.wt(8, 0.3), "y": [0.0] * 8})
    x = rng.act(16, 1e-4)
    w = rng.wt(16, 0.3)
    cases.append({"x": x, "w": w, "y": ref.rmsnorm_zero_centred(x, w, EPS)})
    return {"op": "rmsnorm_zero_centred", "formula": "y = x * rsqrt(mean(x^2) + eps) * (1 + w)",
            "eps": EPS, "cases": cases}


def rmsnorm_gated(rng):
    cases = []
    for n in (1, 7, 16, 33, 128):
        x, z, w = rng.act(n, 2.0), rng.act(n, 2.0), [st.round_bf16(1.0 + v) for v in rng.wt(n, 0.2)]
        cases.append({"x": x, "z": z, "w": w, "y": ref.rmsnorm_gated(x, z, w, EPS)})
    return {"op": "rmsnorm_gated", "formula": "y = x * rsqrt(mean(x^2) + eps) * w * silu(z)  (plain w)",
            "eps": EPS, "cases": cases}


def l2norm(rng):
    cases = []
    for n in (1, 8, 16, 33, 128):
        x = rng.act(n, 3.0)
        cases.append({"x": x, "y": ref.l2norm(x, EPS)})
    cases.append({"x": [0.0] * 16, "y": [0.0] * 16})
    x = rng.act(16, 1e-4)
    cases.append({"x": x, "y": ref.l2norm(x, EPS)})
    return {"op": "l2norm", "formula": "y = x * rsqrt(sum(x^2) + eps)", "eps": EPS, "cases": cases}


def rope(rng):
    groups = []
    for head_dim, rot, theta, positions in (
            (32, 8, 1e7, (0, 1, 2, 7, 100, 4093)),
            (256, 64, 1e7, (0, 1, 31, 1000, 100000)),
            (16, 16, 100.0, (0, 3, 50)),
            (24, 6, 10000.0, (5, 17))):
        cases = []
        for pos in positions:
            x = rng.act(head_dim, 1.5)
            cases.append({"pos": pos, "x": x, "y": ref.rope_apply(x, pos, rot, theta)})
        groups.append({"head_dim": head_dim, "rot_dim": rot, "theta": theta, "cases": cases})
    return {"op": "rope_partial",
            "formula": "i<rot/2: f=theta^(-2i/rot), a=pos*f; y[i]=x[i]cos a - x[i+rot/2]sin a; "
                       "y[i+rot/2]=x[i+rot/2]cos a + x[i]sin a; y[j>=rot]=x[j]",
            "groups": groups}


def conv1d(rng):
    cases = []
    for channels, kernel, length, split in ((20, 4, 9, 5), (8, 4, 6, 1), (24, 3, 7, 4)):
        weight = rng.mat(channels, kernel, 0.6)
        xs = [rng.act(channels, 1.5) for _ in range(length)]
        state = [[0.0] * channels for _ in range(kernel - 1)]
        ys, state_at_split = [], None
        for t, x in enumerate(xs):
            if t == split:
                state_at_split = [list(s) for s in state]
            ys.append(ref.conv1d_silu_step(state, x, weight))
        cases.append({"channels": channels, "kernel": kernel, "weight": weight, "x": xs, "y": ys,
                      "split": split, "state_at_split": state_at_split})
    return {"op": "causal_conv1d_silu",
            "formula": "y[t][c] = silu(sum_j weight[c][j] * xin[t-(K-1)+j][c]), xin[<0] = 0; "
                       "weight[c][K-1] multiplies the current input",
            "cases": cases}


def delta_rule(rng):
    cases = []
    for nk, nv, dk, dv, steps in ((2, 4, 16, 16, 10), (1, 3, 8, 12, 6), (2, 2, 16, 16, 5)):
        seq = []
        for _ in range(steps):
            q, k = rng.act(nk * dk, 1.5), rng.act(nk * dk, 1.5)
            v = rng.act(nv * dv, 1.0)
            g = [f32(-rng.r.uniform(0.01, 2.0)) for _ in range(nv)]
            beta = [f32(ref.sigmoid(rng.r.gauss(0, 1.5))) for _ in range(nv)]
            seq.append((q, k, v, g, beta))
        outs, state = ref.gated_delta_rule(seq, nk, nv, dk, dv)
        cases.append({
            "num_k_heads": nk, "num_v_heads": nv, "dk": dk, "dv": dv,
            "steps": [{"q": q, "k": k, "v": v, "g": g, "beta": b, "o": o}
                      for (q, k, v, g, b), o in zip(seq, outs)],
            "final_state": [[row for row in head] for head in state]})
    return {"op": "gated_delta_rule",
            "formula": "per step: qn = l2norm(q_head)/sqrt(dk), kn = l2norm(k_head); value head i uses key "
                       "head i // (nv/nk); S *= exp(g[i]); kv = S^T kn; delta = (v - kv)*beta[i]; "
                       "S += kn outer delta; o = S^T qn. S[dk][dv] zero at step 0. q,k are raw (pre-l2norm).",
            "cases": cases}


def mini_cfg(hidden, inter, heads, kv_heads, head_dim, partial, theta, lk, lv, lkd, lvd, conv_k, layer_type):
    return ref.Config({"text_config": {
        "hidden_size": hidden, "intermediate_size": inter, "num_hidden_layers": 1,
        "layer_types": [layer_type], "vocab_size": 8, "rms_norm_eps": EPS,
        "num_attention_heads": heads, "num_key_value_heads": kv_heads, "head_dim": head_dim,
        "rope_parameters": {"rope_type": "default", "rope_theta": theta, "partial_rotary_factor": partial},
        "linear_num_key_heads": lk, "linear_num_value_heads": lv, "linear_key_head_dim": lkd,
        "linear_value_head_dim": lvd, "linear_conv_kernel_dim": conv_k, "hidden_act": "silu",
        "attn_output_gate": True}})


class DictWeights:
    def __init__(self, d):
        self.d = d

    def get(self, name):
        return self.d[name]


def layer_model(cfg, weights):
    m = object.__new__(ref.Model)
    m.cfg, m.w = cfg, DictWeights(weights)
    return m


def full_attention(rng):
    hidden, heads, kvh, hd, rot, theta, length = 32, 4, 2, 16, 8, 100.0, 8
    cfg = mini_cfg(hidden, 8, heads, kvh, hd, rot / hd, theta, 1, 1, 1, 1, 4, "full_attention")
    assert cfg.rot == rot
    w = {"q_proj.weight": rng.mat(heads * hd * 2, hidden, 1.4 / math.sqrt(hidden)),
         "k_proj.weight": rng.mat(kvh * hd, hidden, 1.4 / math.sqrt(hidden)),
         "v_proj.weight": rng.mat(kvh * hd, hidden, 1.0 / math.sqrt(hidden)),
         "o_proj.weight": rng.mat(hidden, heads * hd, 1.0 / math.sqrt(heads * hd)),
         "q_norm.weight": rng.wt(hd, 0.3), "k_norm.weight": rng.wt(hd, 0.3)}
    m = layer_model(cfg, w)
    ls = {"k": [], "v": []}
    xs = [rng.act(hidden, 1.0) for _ in range(length)]
    ys = [m._full_attn("", x, ls, pos) for pos, x in enumerate(xs)]
    return {"op": "full_attention_layer",
            "note": "mixer only: x is the already input_layernorm'ed hidden state; y is o_proj output "
                    "(no residual). q_proj rows per head h: [h*2*head_dim, +head_dim) = query, then "
                    "[+head_dim, +2*head_dim) = gate. Position of x[t] is t.",
            "hidden": hidden, "heads": heads, "kv_heads": kvh, "head_dim": hd, "rot_dim": rot,
            "theta": theta, "eps": EPS, "weights": w, "x": xs, "y": ys}


def linear_attention(rng):
    hidden, lk, lv, lkd, lvd, ck, length = 32, 2, 4, 8, 8, 4, 8
    cfg = mini_cfg(hidden, 8, 2, 1, 8, 0.5, 1e7, lk, lv, lkd, lvd, ck, "linear_attention")
    key_dim, value_dim = lk * lkd, lv * lvd
    conv_dim = 2 * key_dim + value_dim
    w = {"in_proj_qkv.weight": rng.mat(conv_dim, hidden, 1.0 / math.sqrt(hidden)),
         "in_proj_z.weight": rng.mat(value_dim, hidden, 1.0 / math.sqrt(hidden)),
         "in_proj_b.weight": rng.mat(lv, hidden, 1.5 / math.sqrt(hidden)),
         "in_proj_a.weight": rng.mat(lv, hidden, 1.5 / math.sqrt(hidden)),
         "conv1d.weight": rng.mat(conv_dim, ck, 0.5),
         "dt_bias": [st.round_bf16(rng.r.gauss(-0.3, 0.5)) for _ in range(lv)],
         "A_log": [f32(math.log(rng.r.uniform(0.05, 1.0))) for _ in range(lv)],
         "norm.weight": [st.round_bf16(1.0 + v) for v in rng.wt(lvd, 0.2)],
         "out_proj.weight": rng.mat(hidden, value_dim, 1.0 / math.sqrt(value_dim))}
    m = layer_model(cfg, w)
    ls = m.new_state()["layers"][0]
    xs = [rng.act(hidden, 1.0) for _ in range(length)]
    ys = [m._linear_attn("", x, ls) for x in xs]
    return {"op": "linear_attention_layer",
            "note": "mixer only: x is the already input_layernorm'ed hidden state; y is out_proj output "
                    "(no residual). conv1d.weight is [conv_dim][kernel] (the [C,1,K] tensor squeezed); "
                    "A_log and dt_bias are per value head. State (conv, S) is zero before x[0].",
            "hidden": hidden, "num_k_heads": lk, "num_v_heads": lv, "key_head_dim": lkd,
            "value_head_dim": lvd, "conv_kernel": ck, "eps": EPS, "weights": w, "x": xs, "y": ys}


def mlp(rng):
    hidden, inter, length = 32, 48, 4
    cfg = mini_cfg(hidden, inter, 2, 1, 8, 0.5, 1e7, 1, 1, 1, 1, 4, "full_attention")
    w = {"gate_proj.weight": rng.mat(inter, hidden, 1.0 / math.sqrt(hidden)),
         "up_proj.weight": rng.mat(inter, hidden, 1.0 / math.sqrt(hidden)),
         "down_proj.weight": rng.mat(hidden, inter, 1.0 / math.sqrt(inter))}
    m = layer_model(cfg, w)
    xs = [rng.act(hidden, 1.0) for _ in range(length)]
    return {"op": "mlp", "formula": "y = down(silu(gate(x)) * up(x))",
            "hidden": hidden, "intermediate": inter, "weights": w, "x": xs,
            "y": [m._mlp("", x) for x in xs]}


def bf16_bits(v):
    return st.float_to_bf16_bits(v)


def q8_case(name, x):
    assert len(x) == 32, name
    scale, q = quant_ref.quantize_group(x)
    return {"name": name, "x": x, "x_bf16_bits": [bf16_bits(v) for v in x], "scale": scale,
            "scale_f32_bits": struct.unpack("<I", struct.pack("<f", scale))[0], "q": q}


def q8(rng):
    groups = []
    ties = [0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 3.5, -3.5, 4.5, 125.5, 126.5, -126.5, 127.0, 1.0, 0.0,
            -0.0, 2.0, -2.0, 0.25, 0.75, -0.75, 62.5, 63.5, 64.5, -64.5, 100.0, -100.0, 7.5, 8.5,
            -8.5, 0.125, 5.5]
    groups.append(q8_case("ties_at_half_scale_1", ties))
    groups.append(q8_case("zero_group", [0.0] * 32))
    groups.append(q8_case("negative_zero_group", [-0.0] * 32))
    groups.append(q8_case("negative_zero_with_values", [-0.0, 1.0, -0.0, -2.0] + [0.5] * 28))
    groups.append(q8_case("max_negative_at_clamp", [-3.0] + [st.round_bf16(0.01 * i) for i in range(31)]))
    groups.append(q8_case("max_positive_at_clamp", [1.0] + [-1.0, 0.5, -0.5, 0.999] + [0.0] * 27))
    groups.append(q8_case("scale_not_exact", [st.round_bf16(0.3 * (i - 15)) for i in range(32)]))
    groups.append(q8_case("tiny_values", [st.round_bf16(1e-30 * (i + 1)) for i in range(32)]))
    groups.append(q8_case("bf16_subnormal_scale", [st.bf16_to_floats(b"\x01\x00")[0]] + [0.0] * 31))
    groups.append(q8_case("huge_values", [st.round_bf16(1e30 * (i - 16)) for i in range(32)]))
    for i in range(6):
        groups.append(q8_case("random_%d" % i, [st.round_bf16(rng.r.gauss(0, 0.05 * (i + 1))) for _ in range(32)]))
    rows, cols = 3, 96
    vals = [st.round_bf16(rng.r.gauss(0, 0.1)) for _ in range(rows * cols)]
    vals[40] = 0.0
    vals[64:96] = [0.0] * 32  # row 0 third group is all zero
    q, scales = quant_ref.quantize_matrix(vals, rows, cols)
    matrix = {"rows": rows, "cols": cols, "x": vals, "q": q, "scale": scales,
              "scale_f32_bits": [struct.unpack("<I", struct.pack("<f", s))[0] for s in scales]}
    return {"op": "q8_quantize",
            "formula": "absmax over the group; scale = (float)(absmax/127.0) (double division, one round to f32); "
                       "scale==0 -> q=0; else q = clamp(round_half_even(x/scale in double), -127, 127). "
                       "x values are bf16-exact; -0.0 gives q=0.",
            "group": 32, "groups": groups, "matrix": matrix}


def build():
    rng = R(SEED)
    out = {}
    for name, fn in (("rmsnorm", rmsnorm), ("rmsnorm_gated", rmsnorm_gated), ("l2norm", l2norm),
                     ("rope", rope), ("conv1d_silu", conv1d), ("gated_delta_rule", delta_rule),
                     ("full_attention_layer", full_attention),
                     ("linear_attention_layer", linear_attention), ("mlp", mlp), ("q8_quantize", q8)):
        text = json.dumps(fn(rng), separators=(",", ":"), allow_nan=False) + "\n"
        out[name + ".json"] = text.encode("ascii")
    return out


def default_dir():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "tests", "fixtures", "ops")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=default_dir())
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    files = build()
    if a.check:
        bad = []
        for name, data in files.items():
            path = os.path.join(a.out, name)
            if not os.path.exists(path):
                bad.append("missing " + name)
            else:
                with open(path, "rb") as f:
                    if f.read() != data:
                        bad.append("differs " + name)
        if os.path.isdir(a.out):
            bad += ["unexpected " + n for n in sorted(os.listdir(a.out)) if n not in files]
        if bad:
            print("make_ops --check FAILED: " + "; ".join(bad), file=sys.stderr)
            return 1
        print("make_ops --check OK: %d files byte-identical" % len(files))
        return 0
    os.makedirs(a.out, exist_ok=True)
    for name, data in files.items():
        with open(os.path.join(a.out, name), "wb") as f:
            f.write(data)
        print("%s %d bytes" % (name, len(data)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
