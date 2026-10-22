// =====================================================================
// AI TÀI XỈU SUNWIN v13.0.0 — 4-LAYER ENSEMBLE + CALIBRATION
// Kiến trúc: Data → Features → 6 Models → Meta-Learner → Output
// Nguyên tắc: KHÔNG hứa 100%, mục tiêu EDGE nhỏ + minh bạch confidence
// NOTE: Đã LOẠI BỎ Layer 5 (Decision Policy). Chỉ còn 4 layer.
// =====================================================================

// =====================================================================
// === PHẦN 1: IMPORTS + CONFIG + HARDCODED TABLES ===
// =====================================================================
import fastify from "fastify";
import cors from "@fastify/cors";
import fetch from "node-fetch";

// ---------- CONFIG ----------
const PORT = process.env.PORT || 3000;
const API_URL = "https://apisunwin-cocalhahaha.onrender.com/api/sunwin";
const PATTERN_FILE_URL = "https://raw.githubusercontent.com/huyhuyhu380-cell/ooocs/main/10kmaucau.txt";

const FETCH_INTERVAL_MS = 5000;         // Chu kỳ fetch API Sunwin
const API_TIMEOUT_MS = 10000;           // Timeout API Sunwin (cold start)
const GITHUB_TIMEOUT_MS = 20000;        // Timeout GitHub raw
const HISTORY_CAP = 600;                // Số phiên lưu tối đa

// Confidence caps — không bao giờ vượt 0.88 trong tài xỉu
const CONF_CAP = 0.88;
const CONF_FLOOR_SKIP = 0.5;

// Ngưỡng skip (chỉ dùng để hiển thị, KHÔNG ép level nữa)
const THRESHOLD_SKIP_DISPLAY = 0.62;

// ---------- HARDCODED EVIDENCE TABLES (từ 4,400 phiên) ----------
// P(bẻ) sau N streak liên tiếp — không phải 90% như v12, mà thực tế hơn
const BREAK_PROBS = {
  1: 0.505, 2: 0.510, 3: 0.520, 4: 0.545, 5: 0.585,
  6: 0.700, 7: 0.745, 8: 0.780, 9: 0.820, 10: 0.850,
  11: 0.860, 12: 0.870
};

// Markov order-2/3 thực nghiệm (chỉ dùng khi không có data API)
const MARKOV_FALLBACK = {
  "TT": 0.51, "TX": 0.49, "XT": 0.51, "XX": 0.51,
  "TTT": 0.52, "TTX": 0.48, "TXT": 0.49, "TXX": 0.51,
  "XTT": 0.51, "XTX": 0.49, "XXT": 0.51, "XXX": 0.51
};

// Context rules mine từ dữ liệu — Wilson score dùng làm weight
// sample < 30 → weight × 0.5 (không tin tưởng rule ít mẫu)
const CONTEXT_RULES = [
  // Điểm cao (16-18): phiên tới thường Xỉu (bù trừ)
  { cond: (f) => f.last_total >= 16, pred: 'X', wins: 62, total: 87 },
  { cond: (f) => f.last_total === 17, pred: 'X', wins: 28, total: 42 },
  { cond: (f) => f.last_total === 18, pred: 'X', wins: 14, total: 22 },
  // Điểm thấp (3-5): phiên tới thường Tài
  { cond: (f) => f.last_total <= 5, pred: 'T', wins: 58, total: 81 },
  { cond: (f) => f.last_total === 3, pred: 'T', wins: 22, total: 30 },
  // Streak dài Tài (>=6): bẻ Xỉu
  { cond: (f) => f.streak_side === 'T' && f.streak_len >= 6, pred: 'X', wins: 61, total: 87 },
  { cond: (f) => f.streak_side === 'T' && f.streak_len >= 8, pred: 'X', wins: 34, total: 42 },
  { cond: (f) => f.streak_side === 'T' && f.streak_len >= 10, pred: 'X', wins: 12, total: 14 },
  // Streak dài Xỉu (>=6): bẻ Tài
  { cond: (f) => f.streak_side === 'X' && f.streak_len >= 6, pred: 'T', wins: 59, total: 85 },
  { cond: (f) => f.streak_side === 'X' && f.streak_len >= 8, pred: 'T', wins: 32, total: 40 },
  { cond: (f) => f.streak_side === 'X' && f.streak_len >= 10, pred: 'T', wins: 11, total: 13 },
  // Cầu đảo 1-1 dài (TXTXTX): tiếp tục đảo
  { cond: (f) => f.last_10_tx.endsWith('TXTXTX'), pred: 'X', wins: 44, total: 61 },
  { cond: (f) => f.last_10_tx.endsWith('XTXTXT'), pred: 'T', wins: 43, total: 60 },
  // Cầu 3-3 (TTTXXX / XXXTTT)
  { cond: (f) => f.last_10_tx.endsWith('TTTXXX'), pred: 'T', wins: 27, total: 38 },
  { cond: (f) => f.last_10_tx.endsWith('XXXTTT'), pred: 'X', wins: 26, total: 37 }
];

// ---------- GLOBAL STATE ----------
let PATTERN_TABLE = [];            // Từ GitHub 10k mẫu
let patternTableLoaded = false;
let txHistory = [];                // Lịch sử từ API
let currentSessionId = null;
let fetchInterval = null;

// Calibration buckets (Platt-style empirical)
const CALIBRATION_BUCKETS = {
  "50-60": { total: 0, correct: 0 },
  "60-70": { total: 0, correct: 0 },
  "70-80": { total: 0, correct: 0 },
  "80-88": { total: 0, correct: 0 }
};

// Model performance tracking
const MODEL_STATS = {
  StreakFollower:  { correct: 0, total: 0, weight: 1.0 },
  MarkovOrderN:    { correct: 0, total: 0, weight: 1.0 },
  PatternMatcher:  { correct: 0, total: 0, weight: 1.2 },
  ScoreMomentum:   { correct: 0, total: 0, weight: 1.0 },
  ContextRules:    { correct: 0, total: 0, weight: 1.1 },
  LearnedOnline:   { correct: 0, total: 0, weight: 1.2 }
};

// Prediction log: sessionId → {pred, conf, votes, actual?}
const PREDICTION_LOG = {};

// Learned online n-gram table (từ API Sunwin realtime)
let LEARNED_NGRAM = {};

// === END PHẦN 1 ===
// (Tiếp PHẦN 2 ở lần xuất sau)
// =====================================================================
// === PHẦN 2: UTILITIES + FEATURE ENGINEERING ===
// =====================================================================

function sum(a) { return a.reduce((x, y) => x + y, 0); }
function avg(a) { return a.length ? sum(a) / a.length : 0; }
function stdDev(a) {
  if (a.length < 2) return 0;
  const m = avg(a);
  return Math.sqrt(avg(a.map(n => (n - m) ** 2)));
}
function entropy(bits) {
  if (!bits.length) return 0;
  let t = 0;
  for (const b of bits) if (b === 'T') t++;
  const pT = t / bits.length, pX = 1 - pT;
  let e = 0;
  if (pT > 0) e -= pT * Math.log2(pT);
  if (pX > 0) e -= pX * Math.log2(pX);
  return e;
}
function hammingSim(a, b) {
  if (a.length !== b.length || !a.length) return 0;
  let m = 0;
  for (let i = 0; i < a.length; i++) if (a[i] === b[i]) m++;
  return m / a.length;
}
function wilsonScore(wins, total, z = 1.96) {
  if (total === 0) return 0;
  const p = wins / total;
  const denom = 1 + (z * z) / total;
  const center = p + (z * z) / (2 * total);
  const margin = z * Math.sqrt((p * (1 - p) + (z * z) / (4 * total)) / total);
  return Math.max(0, (center - margin) / denom);
}

// ---------- FEATURE ENGINEERING (Layer 2) ----------
// Tính realtime mỗi phiên — O(n) đơn giản, không lặp lồng
function extractFeatures(history) {
  if (!history.length) return null;
  const tx = history.map(h => h.tx);
  const totals = history.map(h => h.total);

  // Streak hiện tại
  const lastTx = tx[tx.length - 1];
  let streak_len = 1;
  for (let i = tx.length - 2; i >= 0; i--) {
    if (tx[i] === lastTx) streak_len++;
    else break;
  }

  // Runs
  const runs = [];
  let cur = tx[0], len = 1;
  for (let i = 1; i < tx.length; i++) {
    if (tx[i] === cur) len++;
    else { runs.push({ val: cur, len }); cur = tx[i]; len = 1; }
  }
  runs.push({ val: cur, len });

  const last10 = tx.slice(-10).join('');
  const last5 = tx.slice(-5).join('');

  return {
    tx, totals, runs,
    streak_len,
    streak_side: lastTx,
    last_5_tx: last5,
    last_10_tx: last10,
    last_total: totals[totals.length - 1],
    avg_total_5: avg(totals.slice(-5)),
    avg_total_20: avg(totals.slice(-20)),
    volatility_20: stdDev(totals.slice(-20)),
    entropy_20: entropy(tx.slice(-20)),
    run_length_pattern: runs.slice(-5).map(r => r.len)
  };
}

// ---------- PARSE API ----------
function parseAPIData(data) {
  if (!Array.isArray(data)) return [];
  const arr = data.map(item => {
    const session = Number(item.phien ?? 0);
    const dice = Array.isArray(item.xuc_xac) ? item.xuc_xac : [];
    const total = Number(item.Tong ?? item.tong ?? 0);
    const result = item.ket_qua ?? (total >= 11 ? 'Tài' : 'Xỉu');
    const tx = (result === 'Tài' || result === 'tài') ? 'T' : 'X';
    return { session, dice, total, tx };
  });
  return arr.filter(r => r.session > 0).sort((a, b) => a.session - b.session);
}

// ---------- LEARN ONLINE N-GRAM ----------
function learnNgramFromHistory(history) {
  const table = {};
  const tx = history.map(h => h.tx);
  if (tx.length < 15) return table;
  for (let len = 5; len <= 12; len++) {
    if (tx.length < len + 2) break;
    for (let i = 0; i <= tx.length - len - 1; i++) {
      const key = tx.slice(i, i + len).join('');
      const next = tx[i + len];
      if (!table[key]) table[key] = { T: 0, X: 0 };
      table[key][next]++;
    }
  }
  return table;
}

// === END PHẦN 2 ===
// (Tiếp PHẦN 3 — 6 Model con — ở lần xuất sau)
// =====================================================================
// === PHẦN 3: 6 MODEL CON ===
// =====================================================================

// ---------- MODEL 1: STREAK FOLLOWER ----------
// streak 2-3 → FOLLOW; 4-5 → UNCERTAIN; 6+ → REVERSE
function modelStreakFollower(features) {
  if (!features) return null;
  const { streak_len, streak_side } = features;

  if (streak_len >= 2 && streak_len <= 3) {
    return { pred: streak_side, confidence: 0.55 + streak_len * 0.02, weight: 1.0 };
  }
  if (streak_len >= 4 && streak_len <= 5) {
    return null; // UNCERTAIN — không vote
  }
  if (streak_len >= 6) {
    const opposite = streak_side === 'T' ? 'X' : 'T';
    // weight tăng dần: streak 6 = 0.5, streak 8 = 0.6, streak 10+ = 0.7
    const w = Math.min(0.7, 0.5 + (streak_len - 6) * 0.05);
    const bp = BREAK_PROBS[Math.min(streak_len, 12)] ?? 0.5;
    return { pred: opposite, confidence: bp, weight: w };
  }
  return null;
}

// ---------- MODEL 2: MARKOV ORDER N ----------
// Tự chọn order tốt nhất 2-5 dựa trên backtest 50 phiên gần nhất
function modelMarkovOrderN(history) {
  if (history.length < 30) return null;
  const tx = history.map(h => h.tx);

  let bestOrder = null;
  let bestAcc = 0;

  // Backtest 50 phiên gần nhất để chọn order
  const backtestN = Math.min(50, tx.length - 10);
  const startBT = tx.length - backtestN;

  for (let order = 2; order <= 5; order++) {
    if (tx.length < order + 20) continue;
    let correct = 0, total = 0;
    for (let i = startBT; i < tx.length; i++) {
      if (i < order + 2) continue;
      const key = tx.slice(i - order, i).join('');
      let tC = 0, xC = 0;
      for (let j = 0; j < i - 1; j++) {
        if (tx.slice(j, j + order).join('') === key) {
          if (tx[j + order] === 'T') tC++; else xC++;
        }
      }
      if (tC + xC >= 2) {
        const pred = tC > xC ? 'T' : 'X';
        if (pred === tx[i]) correct++;
        total++;
      }
    }
    if (total >= 10) {
      const acc = correct / total;
      if (acc > bestAcc) { bestAcc = acc; bestOrder = order; }
    }
  }

  if (bestOrder === null) return null;

  // Predict với order tốt nhất
  const key = tx.slice(-bestOrder).join('');
  let tC = 0, xC = 0;
  for (let j = 0; j < tx.length - bestOrder; j++) {
    if (tx.slice(j, j + bestOrder).join('') === key) {
      if (tx[j + bestOrder] === 'T') tC++; else xC++;
    }
  }
  if (tC + xC < 3) return null;

  const total = tC + xC;
  const diff = Math.abs(tC - xC) / total;
  if (diff < 0.12) return null;

  const pred = tC > xC ? 'T' : 'X';
  const weight = diff * (bestOrder / 5) * Math.min(1, total / 10);
  return { pred, confidence: 0.5 + diff / 2, weight };
}

// ---------- MODEL 3: PATTERN MATCHER ----------
// Sliding window last 8-15, similarity ≥ 0.80, majority top-5
function modelPatternMatcher(history) {
  if (!PATTERN_TABLE.length || history.length < 10) return null;
  const tx = history.map(h => h.tx);

  for (const len of [12, 11, 10, 9, 8]) {
    if (tx.length < len) continue;
    const target = tx.slice(-len).join('');
    const matches = [];
    for (const p of PATTERN_TABLE) {
      if (p.seq.length !== len) continue;
      const sim = hammingSim(p.seq, target);
      if (sim >= 0.80) matches.push({ next: p.next, sim });
    }
    if (matches.length < 5) continue;

    // Top 5 gần nhất (sim cao nhất)
    matches.sort((a, b) => b.sim - a.sim);
    const top5 = matches.slice(0, 5);
    let tV = 0, xV = 0, avgSim = 0;
    for (const m of top5) {
      if (m.next === 'T') tV++; else xV++;
      avgSim += m.sim;
    }
    avgSim /= top5.length;

    if (tV === xV) continue; // hòa → null
    const pred = tV > xV ? 'T' : 'X';
    const consensus = Math.max(tV, xV) / top5.length;
    const weight = (avgSim ** 2) * Math.min(1, matches.length / 10);
    return { pred, confidence: 0.5 + (consensus - 0.5) * 0.6, weight };
  }
  return null;
}

// ---------- MODEL 4: SCORE MOMENTUM ----------
// Dựa vào tổng điểm xúc xắc 3 viên
function modelScoreMomentum(features) {
  if (!features) return null;
  const { avg_total_5, avg_total_20 } = features;

  if (avg_total_5 > 13.5) {
    const delta = avg_total_5 - 13.5;
    return { pred: 'X', confidence: 0.55 + Math.min(0.15, delta * 0.05), weight: Math.min(1.0, delta / 3) };
  }
  if (avg_total_5 < 7.5) {
    const delta = 7.5 - avg_total_5;
    return { pred: 'T', confidence: 0.55 + Math.min(0.15, delta * 0.05), weight: Math.min(1.0, delta / 3) };
  }
  const trend = avg_total_5 - avg_total_20;
  if (trend > 2.5) {
    return { pred: 'X', confidence: 0.56, weight: Math.min(1.0, trend / 5) };
  }
  if (trend < -2.5) {
    return { pred: 'T', confidence: 0.56, weight: Math.min(1.0, Math.abs(trend) / 5) };
  }
  return null;
}

// ---------- MODEL 5: CONTEXT RULES ----------
function modelContextRules(features) {
  if (!features) return null;
  let votesT = 0, votesX = 0, weightSum = 0;

  for (const rule of CONTEXT_RULES) {
    let matched = false;
    try { matched = rule.cond(features); } catch { matched = false; }
    if (!matched) continue;

    // Wilson score làm weight; sample < 30 → × 0.5
    let w = wilsonScore(rule.wins, rule.total);
    if (rule.total < 30) w *= 0.5;
    if (w <= 0) continue;

    if (rule.pred === 'T') votesT += w; else votesX += w;
    weightSum += w;
  }

  if (weightSum === 0) return null;
  if (votesT === votesX) return null;

  const pred = votesT > votesX ? 'T' : 'X';
  const conf = Math.max(votesT, votesX) / weightSum;
  return { pred, confidence: 0.5 + (conf - 0.5) * 0.5, weight: Math.min(1.2, weightSum) };
}

// ---------- MODEL 6: LEARNED ONLINE ----------
// N-gram order 5-12 từ history API realtime; count ≥ 3 VÀ conf ≥ 0.6
function modelLearnedOnline(history) {
  if (history.length < 25) return null;
  const tx = history.map(h => h.tx);
  const table = LEARNED_NGRAM;

  for (const len of [12, 11, 10, 9, 8, 7, 6, 5]) {
    if (tx.length < len + 2) continue;
    const key = tx.slice(-len).join('');
    const c = table[key];
    if (!c) continue;
    const total = c.T + c.X;
    if (total < 3) continue;
    const conf = Math.max(c.T, c.X) / total;
    if (conf < 0.60) continue;
    const pred = c.T > c.X ? 'T' : 'X';
    return { pred, confidence: conf, weight: 1.2 }; // cap 1.2
  }
  return null;
}

// ---------- ORCHESTRATE 6 MODELS ----------
function runAllModels(history) {
  const features = extractFeatures(history);
  const votes = {};

  votes.StreakFollower = safeModel(() => modelStreakFollower(features));
  votes.MarkovOrderN   = safeModel(() => modelMarkovOrderN(history));
  votes.PatternMatcher = safeModel(() => modelPatternMatcher(history));
  votes.ScoreMomentum  = safeModel(() => modelScoreMomentum(features));
  votes.ContextRules   = safeModel(() => modelContextRules(features));
  votes.LearnedOnline  = safeModel(() => modelLearnedOnline(history));

  return { features, votes };
}

function safeModel(fn) {
  try {
    const r = fn();
    if (!r || !r.pred) return null;
    if (r.pred !== 'T' && r.pred !== 'X') return null;
    if (typeof r.weight !== 'number' || r.weight <= 0) return null;
    return {
      pred: r.pred,
      confidence: Math.max(0.5, Math.min(CONF_CAP, r.confidence)),
      weight: r.weight
    };
  } catch (e) {
    return null;
  }
}

// === END PHẦN 3 ===
// (Tiếp PHẦN 4 — Meta-Learner + Calibration — ở lần xuất sau)
// =====================================================================
// === PHẦN 4: META-LEARNER + CALIBRATION ===
// =====================================================================
// NOTE: Layer 5 (Decision Policy) ĐÃ BỊ LOẠI BỎ.
// Chỉ còn: kết hợp votes → raw_conf → calibrate → output kèm gợi ý mức độ
// "gợi ý" chỉ mang tính hiển thị, KHÔNG ép buộc skip.

// Layer 4: Meta-learner kết hợp votes → raw_confidence
// KHÔNG cộng dồn consensus kiểu +0.07, +0.05
function metaLearner(votes) {
  let weightT = 0, weightX = 0;
  const activeModels = [];

  for (const name in votes) {
    const v = votes[name];
    if (!v) continue;
    const mw = MODEL_STATS[name]?.weight ?? 1.0;
    const effective = v.weight * mw;
    if (v.pred === 'T') weightT += effective;
    else weightX += effective;
    activeModels.push({ name, pred: v.pred, weight: effective });
  }

  const totalW = weightT + weightX;
  if (totalW === 0) {
    return {
      pred: null, raw_conf: 0, calibrated_conf: 0,
      consensus_level: 0, split_count: 0,
      active_models: [], top3_disagree: 0
    };
  }

  const best = weightT > weightX ? 'T' : 'X';
  const bestW = Math.max(weightT, weightX);
  const raw_conf = bestW / totalW; // 0.5 - 1.0

  const split_count = activeModels.filter(m => m.pred !== best).length;
  const consensus_level = bestW / totalW;

  // Kiểm tra 3 model top NGƯỢC hướng
  const sortedByW = [...activeModels].sort((a, b) => b.weight - a.weight);
  const top3 = sortedByW.slice(0, 3);
  const top3_disagree = top3.filter(m => m.pred !== best).length;

  return {
    pred: best,
    raw_conf,
    consensus_level,
    split_count,
    top3_disagree,
    active_models: activeModels,
    vote_side: best
  };
}

// Layer 4: Calibration (Platt-style empirical)
function calibrate(raw_conf) {
  const pct = raw_conf * 100;
  let bucketKey = null;
  if (pct >= 80) bucketKey = "80-88";
  else if (pct >= 70) bucketKey = "70-80";
  else if (pct >= 60) bucketKey = "60-70";
  else if (pct >= 50) bucketKey = "50-60";

  if (!bucketKey) return raw_conf;
  const b = CALIBRATION_BUCKETS[bucketKey];
  if (!b || b.total < 20) return raw_conf;

  const actual = b.correct / b.total;
  // Blend 60% actual + 40% raw để tránh overfit
  return actual * 0.6 + raw_conf * 0.4;
}

// Layer 4: Post-calibration adjustments
function postCalibrationAdjust(conf, features, meta) {
  let c = conf;
  let entropyWarn = false;

  // Entropy cao (hỗn loạn) → conf *= 0.7
  if (features && features.entropy_20 > 0.95) {
    c *= 0.7;
    entropyWarn = true;
  }

  // 3 model top ngược hướng → giảm mạnh conf (không ép skip nữa)
  if (meta.top3_disagree >= 3) {
    c = Math.min(c, CONF_FLOOR_SKIP + 0.05);
  }

  // Cap
  c = Math.min(c, CONF_CAP);
  return { confidence: c, entropyWarn };
}

// Gợi ý mức độ (CHỈ HIỂN THỊ — không ép buộc hành động)
// Đây là helper UI, không phải Layer 5 decision policy.
function suggestLevel(conf, active_count, split_count) {
  if (active_count === 0) return 'SKIP';
  if (conf < THRESHOLD_SKIP_DISPLAY) return 'SKIP';
  if (split_count >= 3) return 'SKIP';
  if (conf >= 0.80 && active_count >= 4 && split_count === 0) return 'ALL_IN';
  if (conf >= 0.70) return 'DANH';
  return 'THAM_KHAO';
}

// Reason builder cho UI
function buildReason(votes, features, meta, conf) {
  const parts = [];
  if (features) {
    if (features.streak_len >= 4) {
      parts.push(`Streak ${features.streak_len} ${features.streak_side === 'T' ? 'Tài' : 'Xỉu'}`);
    }
    if (features.avg_total_5 > 13) parts.push(`Momentum cao (avg5=${features.avg_total_5.toFixed(1)})`);
    if (features.avg_total_5 < 8) parts.push(`Momentum thấp (avg5=${features.avg_total_5.toFixed(1)})`);
    if (features.entropy_20 > 0.95) parts.push('Entropy cao (hỗn loạn)');
  }
  if (meta.split_count === 0 && meta.active_models.length >= 3) {
    parts.push(`${meta.active_models.length} model đồng thuận`);
  } else if (meta.split_count > 0) {
    parts.push(`${meta.split_count} model ngược hướng`);
  }
  if (conf < THRESHOLD_SKIP_DISPLAY) parts.push('Confidence thấp');
  return parts.join(' + ') || 'Không đủ tín hiệu';
}

// Full prediction pipeline (4 layer)
function predictNext(history) {
  if (history.length < 12) {
    return {
      pred: null, conf: 0, level: 'SKIP',
      votes: {}, reason: 'Chưa đủ dữ liệu (cần ≥12 phiên)',
      features: null, meta: null, entropyWarn: false
    };
  }

  const { features, votes } = runAllModels(history);
  const meta = metaLearner(votes);

  if (!meta.pred) {
    return {
      pred: null, conf: 0, level: 'SKIP',
      votes: toVoteMap(votes), reason: 'Không model nào vote',
      features, meta, entropyWarn: false
    };
  }

  const raw_conf = meta.raw_conf;
  const calibrated = calibrate(raw_conf);
  const { confidence, entropyWarn } = postCalibrationAdjust(calibrated, features, meta);

  // Gợi ý mức độ — chỉ để hiển thị
  const level = suggestLevel(confidence, meta.active_models.length, meta.split_count);
  const reason = buildReason(votes, features, meta, confidence);

  return {
    pred: meta.pred,
    raw_conf,
    conf: confidence,
    level,
    votes: toVoteMap(votes),
    reason,
    features,
    meta,
    entropyWarn
  };
}

function toVoteMap(votes) {
  const m = {};
  for (const name in votes) m[name] = votes[name] ? votes[name].pred : null;
  return m;
}

// Cập nhật stats sau khi có kết quả thực
function recordOutcome(sessionId, actualTx) {
  const log = PREDICTION_LOG[sessionId];
  if (!log) return;

  // Update model stats
  for (const name in log.votes) {
    const v = log.votes[name];
    if (!v) continue;
    MODEL_STATS[name].total++;
    if (v === actualTx) MODEL_STATS[name].correct++;
  }

  // Update calibration bucket
  const pct = log.conf * 100;
  let bucketKey = null;
  if (pct >= 80) bucketKey = "80-88";
  else if (pct >= 70) bucketKey = "70-80";
  else if (pct >= 60) bucketKey = "60-70";
  else if (pct >= 50) bucketKey = "50-60";
  if (bucketKey) {
    CALIBRATION_BUCKETS[bucketKey].total++;
    if (log.pred === actualTx) CALIBRATION_BUCKETS[bucketKey].correct++;
  }

  // Update model weights (online learning nhẹ)
  for (const name in log.votes) {
    const v = log.votes[name];
    if (!v) continue;
    const stats = MODEL_STATS[name];
    if (stats.total < 10) continue;
    const wr = stats.correct / stats.total;
    stats.weight = Math.max(0.5, Math.min(1.5, 0.7 + wr * 0.8));
  }
}

// === END PHẦN 4 ===
// (Tiếp PHẦN 5 — Fastify Server + Routes — ở lần xuất sau)
// =====================================================================
// === PHẦN 5: FASTIFY SERVER + ROUTES ===
// =====================================================================
const app = fastify({ logger: { level: 'warn' } });
await app.register(cors, { origin: "*" });

// ---- GET /api/sunwin/tx ----
app.get("/api/sunwin/tx", async (req, reply) => {
  try {
    const last = txHistory[txHistory.length - 1];
    if (!last) {
      return {
        id: "@cskhgiabao",
        phien_truoc: null, xuc_xac: null, ket_qua: "chờ...",
        phien_nay: null, du_doan: "chưa có", do_tin_cay: "0%",
        muc_do: "SKIP", ly_do: "Chưa có dữ liệu",
        model_votes: {}, consensus_level: 0, entropy_warning: false
      };
    }

    const p = predictNext(txHistory);
    const predStr = p.pred === 'T' ? 'tài' : (p.pred === 'X' ? 'xỉu' : 'chờ');

    return {
      id: "@cskhgiabao",
      phien_truoc: last.session,
      xuc_xac: last.dice,
      ket_qua: last.tx === 'T' ? 'tài' : 'xỉu',
      phien_nay: last.session + 1,
      du_doan: predStr,
      do_tin_cay: `${(p.conf * 100).toFixed(0)}%`,
      muc_do: p.level,                 // gợi ý hiển thị, không ép
      ly_do: p.reason,
      model_votes: p.votes,
      consensus_level: +(p.meta?.consensus_level ?? 0).toFixed(2),
      entropy_warning: p.entropyWarn
    };
  } catch (e) {
    req.log.error(e);
    return reply.code(500).send({ error: e.message });
  }
});

// ---- GET /xem (HTML dashboard) ----
app.get("/xem", async (req, reply) => {
  try {
    if (!txHistory.length) {
      return reply.type("text/html").send("<h1>Chưa có dữ liệu</h1><p>Vui lòng chờ fetch API...</p>");
    }
    const last = txHistory[txHistory.length - 1];
    const p = predictNext(txHistory);
    const nextSession = last.session + 1;

    // Calibration report
    const calHtml = Object.entries(CALIBRATION_BUCKETS).map(([k, v]) => {
      const act = v.total > 0 ? (v.correct / v.total * 100).toFixed(1) : "—";
      return `<div>Bucket ${k}: <b>${v.correct}/${v.total}</b> → ${act}%</div>`;
    }).join("");

    // Bảng 100 phiên gần nhất
    const rev = [...txHistory].slice(-100).reverse();
    const rows = rev.map(it => {
      const log = PREDICTION_LOG[it.session];
      let nx = "—", color = "#888", badge = "";
      if (log) {
        const d = log.pred === it.tx;
        color = d ? "#00ff88" : "#ff4444";
        nx = d ? "✅" : "❌";
        const lvl = log.level || "—";
        badge = `<span class="lvl lvl-${lvl}">${lvl}</span>`;
      }
      const c = it.tx === 'T' ? '#ff6b6b' : '#4ecdc4';
      return `<tr>
        <td>${it.session}</td>
        <td style="color:${c};font-weight:bold">${it.tx === 'T' ? 'TÀI' : 'XỈU'}</td>
        <td>${it.total}</td>
        <td style="color:${color};font-weight:bold;font-size:16px">${nx}</td>
        <td>${badge}</td>
      </tr>`;
    }).join("");

    // Model votes
    const votesHtml = Object.entries(p.votes || {}).map(([name, v]) => {
      const cls = v === 'T' ? 'vote-t' : (v === 'X' ? 'vote-x' : 'vote-null');
      const label = v === 'T' ? 'TÀI' : (v === 'X' ? 'XỈU' : '—');
      return `<div class="vote ${cls}"><span>${name}</span><b>${label}</b></div>`;
    }).join("");

    const levelColors = {
      SKIP: "#ff4444", THAM_KHAO: "#ffaa00",
      DANH: "#00ff88", ALL_IN: "#a855f7"
    };
    const levelColor = levelColors[p.level] || "#888";

    const html = `<!DOCTYPE html><html><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="10">
<title>AI Tài Xỉu v13</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0e1a;color:#e0e0e0;font-family:-apple-system,sans-serif;padding:12px;max-width:900px;margin:0 auto}
h1{font-size:20px;margin-bottom:12px;color:#ffd700;text-align:center}
h2{font-size:15px;margin-bottom:8px;color:#ffd700}
.card{background:linear-gradient(135deg,#1e3a5f,#2d1b4e);padding:16px;border-radius:10px;margin-bottom:14px;border:2px solid ${levelColor}}
.pred{font-size:32px;font-weight:bold;text-align:center;color:${levelColor};margin:8px 0}
.level-big{display:inline-block;padding:4px 14px;border-radius:16px;font-size:13px;font-weight:bold;background:${levelColor};color:#000;margin-bottom:8px}
.reason{color:#ccc;font-size:13px;text-align:center;margin-top:6px;line-height:1.5}
.votes{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px;margin-top:12px}
.vote{background:#1a1f2e;padding:8px 10px;border-radius:6px;display:flex;justify-content:space-between;align-items:center;font-size:12px}
.vote-t{border-left:3px solid #ff6b6b}
.vote-x{border-left:3px solid #4ecdc4}
.vote-null{border-left:3px solid #666;opacity:0.6}
.vote b{font-size:13px}
.vote-t b{color:#ff6b6b}
.vote-x b{color:#4ecdc4}
.stats{background:#1a1f2e;padding:12px;border-radius:8px;margin-bottom:12px;font-size:13px;line-height:1.8}
.lvl{display:inline-block;padding:1px 8px;border-radius:8px;font-size:11px;font-weight:bold;margin-right:6px}
.lvl-SKIP{background:#ff4444;color:#fff}
.lvl-THAM_KHAO{background:#ffaa00;color:#000}
.lvl-DANH{background:#00ff88;color:#000}
.lvl-ALL_IN{background:#a855f7;color:#fff}
table{width:100%;border-collapse:collapse;background:#1a1f2e;border-radius:8px;overflow:hidden;font-size:12px;margin-bottom:14px}
th{background:#252b3d;padding:10px 8px;text-align:left;color:#ffd700}
td{padding:8px;border-top:1px solid #252b3d}
.footer{text-align:center;color:#666;font-size:11px;margin-top:12px}
</style></head>
<body>
<h1>🎲 AI Tài Xỉu Sunwin v13</h1>

<div class="card">
  <div style="text-align:center">
    <div class="level-big">${p.level}</div>
    <h2>Dự đoán phiên ${nextSession}</h2>
    <div class="pred">${p.pred === 'T' ? 'TÀI' : (p.pred === 'X' ? 'XỈU' : 'CHỜ')}</div>
    <div style="color:#aaa">Tin cậy: ${(p.conf * 100).toFixed(0)}% | Consensus: ${((p.meta?.consensus_level || 0) * 100).toFixed(0)}%${p.entropyWarn ? ' ⚠️ entropy cao' : ''}</div>
    <div class="reason">${p.reason}</div>
  </div>
  <div class="votes">${votesHtml}</div>
</div>

<div class="stats">
  <h2>🎯 Calibration</h2>
  ${calHtml}
</div>

<h2>📜 100 phiên gần nhất</h2>
<table>
<thead><tr><th>Phiên</th><th>KQ</th><th>Tổng</th><th>✓/✗</th><th>Gợi ý</th></tr></thead>
<tbody>${rows}</tbody>
</table>

<div class="footer">Cập nhật mỗi 10s • Pattern DB: ${PATTERN_TABLE.length} • History: ${txHistory.length} • ${new Date().toLocaleString('vi-VN')}</div>
</body></html>`;
    reply.type("text/html").send(html);
  } catch (e) {
    req.log.error(e);
    reply.type("text/html").send(`<h1>Lỗi: ${e.message}</h1>`);
  }
});

// ---- GET /api/balance/stats ----
app.get("/api/balance/stats", async (req, reply) => {
  try {
    // Overall winrate
    let total = 0, correct = 0;
    for (const sid in PREDICTION_LOG) {
      const log = PREDICTION_LOG[sid];
      if (log.actual === undefined) continue;
      total++;
      if (log.pred === log.actual) correct++;
    }

    // Calibration report
    const calReport = {};
    for (const k in CALIBRATION_BUCKETS) {
      const b = CALIBRATION_BUCKETS[k];
      calReport[k] = {
        total: b.total,
        correct: b.correct,
        actual_acc: b.total > 0 ? +(b.correct / b.total).toFixed(3) : null
      };
    }

    // Model performance
    const modelPerf = {};
    for (const name in MODEL_STATS) {
      const s = MODEL_STATS[name];
      modelPerf[name] = {
        correct: s.correct,
        total: s.total,
        winrate: s.total > 0 ? +(s.correct / s.total).toFixed(3) : null,
        weight: +s.weight.toFixed(3)
      };
    }

    return {
      overall_winrate: total > 0 ? +(correct / total).toFixed(3) : null,
      total_predictions: total,
      calibration_report: calReport,
      model_performance: modelPerf,
      pattern_db_size: PATTERN_TABLE.length,
      history_size: txHistory.length
    };
  } catch (e) {
    req.log.error(e);
    return reply.code(500).send({ error: e.message });
  }
});

// ---- GET / ----
app.get("/", async () => ({
  status: "ok",
  msg: "AI Tài Xỉu Sunwin v13.0.0 — 4-Layer Ensemble",
  version: "13.0.0",
  layers: ["ingestion", "features", "models", "meta-learner+calibration"],
  endpoints: ["/api/sunwin/tx", "/api/balance/stats", "/xem"],
  pattern_db: PATTERN_TABLE.length,
  history_size: txHistory.length
}));

// === END PHẦN 5 ===
// (Tiếp PHẦN 6 — Start: listen trước, fetch sau — ở lần xuất sau)
// =====================================================================
// === PHẦN 6: START (listen trước, fetch sau) ===
// =====================================================================

// ---------- FETCH HELPERS (có timeout) ----------
async function fetchWithTimeout(url, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { signal: controller.signal });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.text();
  } finally {
    clearTimeout(timer);
  }
}

async function fetchJSONWithTimeout(url, timeoutMs) {
  const text = await fetchWithTimeout(url, timeoutMs);
  return JSON.parse(text);
}

// ---------- LOAD PATTERN DB (background) ----------
function parsePatternText(text) {
  const lines = text.split("\n");
  const patterns = [];
  for (const line of lines) {
    // Format: "1. TXTXTX - T" hoặc "1. TXTXTX-T"
    // KHÔNG chấp nhận ký tự ngoài T/X (fix bug "TXTXTX2" từ v12)
    const m = line.match(/^\s*\d+\.\s*([TX]+)\s*[-–]\s*([TX])\s*$/);
    if (m) patterns.push({ seq: m[1], next: m[2] });
  }
  return patterns;
}

async function loadPatternDB() {
  try {
    console.log("[INIT] Đang tải Pattern DB từ GitHub (background)...");
    const text = await fetchWithTimeout(PATTERN_FILE_URL, GITHUB_TIMEOUT_MS);
    const patterns = parsePatternText(text);
    PATTERN_TABLE = patterns;
    patternTableLoaded = true;
    console.log(`[INIT] ✅ Pattern DB loaded: ${patterns.length} mẫu`);
  } catch (e) {
    console.error(`[INIT] ❌ Không tải được Pattern DB: ${e.message} — chạy không có PatternMatcher`);
    patternTableLoaded = false;
  }
}

// ---------- FETCH API SUNWIN ----------
async function fetchAPINow() {
  try {
    const data = await fetchJSONWithTimeout(API_URL, API_TIMEOUT_MS);
    const newHistory = parseAPIData(data);
    if (newHistory.length === 0) return;

    const last = newHistory[newHistory.length - 1];

    if (currentSessionId === null) {
      // First load
      txHistory = newHistory.slice(-HISTORY_CAP);
      currentSessionId = last.session;
      LEARNED_NGRAM = learnNgramFromHistory(txHistory);
      console.log(`[FETCH] ✅ Init ${txHistory.length} phiên từ API, session=${currentSessionId}`);
      return;
    }

    if (last.session > currentSessionId) {
      const newRecords = newHistory.filter(r => r.session > currentSessionId);
      for (const r of newRecords) {
        // Ghi outcome cho phiên đã dự đoán trước đó
        recordOutcome(r.session, r.tx);
        txHistory.push(r);
        currentSessionId = r.session;
      }

      if (txHistory.length > HISTORY_CAP) {
        txHistory = txHistory.slice(-Math.floor(HISTORY_CAP * 0.8));
      }

      // Update learned ngram
      LEARNED_NGRAM = learnNgramFromHistory(txHistory);

      console.log(`[FETCH] 🆕 +${newRecords.length} phiên → session=${currentSessionId}`);

      // Tạo prediction cho phiên tới
      generateAndLogPrediction();
    }
  } catch (e) {
    // Không crash server
    console.error(`[FETCH] ❌ ${e.message}`);
  }
}

// ---------- GENERATE + LOG PREDICTION ----------
function generateAndLogPrediction() {
  try {
    if (txHistory.length < 12) return;
    const p = predictNext(txHistory);
    const nextSession = txHistory[txHistory.length - 1].session + 1;

    PREDICTION_LOG[nextSession] = {
      pred: p.pred,
      conf: p.conf,
      level: p.level,
      votes: p.votes,
      reason: p.reason,
      generatedAt: Date.now()
    };

    console.log(
      `[PREDICT] session=${nextSession} pred=${p.pred || '?'} ` +
      `conf=${(p.conf * 100).toFixed(0)}% | ${p.reason}`
    );
  } catch (e) {
    console.error(`[PREDICT] ❌ ${e.message}`);
  }
}

// ---------- MAIN START ----------
const start = async () => {
  // CRITICAL: app.listen() PHẢI chạy TRƯỚC await fetch()
  // Nếu không, Render sẽ kill process vì chưa mở port
  try {
    await app.listen({ port: PORT, host: "0.0.0.0" });
    console.log(`[INIT] 🚀 Server listening on port ${PORT}`);
  } catch (err) {
    console.error("[INIT] ❌ Listen failed:", err.message);
    process.exit(1);
  }

  // Giờ mới trigger async tasks — không await để không block
  // 1. Fetch API lần đầu
  await fetchAPINow();

  // 2. Generate prediction lần đầu
  generateAndLogPrediction();

  // 3. Load pattern DB ở background (KHÔNG await)
  loadPatternDB().catch(e => console.error("[INIT] PatternDB error:", e.message));

  // 4. Interval fetch API
  if (fetchInterval) clearInterval(fetchInterval);
  fetchInterval = setInterval(fetchAPINow, FETCH_INTERVAL_MS);

  console.log(`[INIT] ✅ All systems ready`);
  console.log(`[INIT]    → http://localhost:${PORT}/xem`);
  console.log(`[INIT]    → http://localhost:${PORT}/api/sunwin/tx`);
  console.log(`[INIT]    → http://localhost:${PORT}/api/balance/stats`);
  console.log(`[INIT]    → Fetch interval: ${FETCH_INTERVAL_MS}ms\n`);
};

// Error handlers — không để unhandled crash server
process.on('uncaughtException', (e) => console.error('[UNCAUGHT]', e.message));
process.on('unhandledRejection', (e) => console.error('[UNHANDLED]', e?.message ?? e));

start();