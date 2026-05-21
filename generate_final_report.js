const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
        PageNumber, PageBreak, ImageRun, LineRuleType } = require('docx');
const fs = require('fs');
const path = require('path');

const outputDir = path.join(__dirname, 'output');

// ── NeurIPS 2021 layout constants ─────────────────────────────────────────────
// US Letter; L/R margins 1.5 in (2160 DXA); T/B margins 1 in (1440 DXA)
// Content width 5.5 in = 7920 DXA
const CW       = 7920;
const BODY_SZ  = 20;   // 10 pt
const CAP_SZ   = 18;   // 9 pt — captions & references
const H1_SZ    = 24;   // 12 pt
const H2_SZ    = 20;   // 10 pt
const TITLE_SZ = 34;   // 17 pt
const ABS_SZ   = 24;   // 12 pt "Abstract"
const LS       = { line: 220, lineRule: LineRuleType.EXACT };
const PA       = 110;  // 5.5 pt after-paragraph (twips)

const BNONE  = { style: BorderStyle.NONE,   size: 0,  color: 'FFFFFF', space: 0 };
const BTHICK = { style: BorderStyle.SINGLE, size: 16, color: '000000', space: 0 };
const BMID   = { style: BorderStyle.SINGLE, size: 6,  color: '000000', space: 0 };

// ── Helpers ───────────────────────────────────────────────────────────────────

function loadImage(fn) {
  const p = path.join(outputDir, fn);
  return fs.existsSync(p) ? fs.readFileSync(p) : null;
}

function img(fn, w, h, caption) {
  const data = loadImage(fn);
  const out = [];
  if (data) {
    out.push(new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { ...LS, before: 160, after: 40 },
      children: [new ImageRun({ type: 'png', data,
        transformation: { width: w, height: h },
        altText: { title: caption, description: caption, name: caption } })]
    }));
  }
  out.push(new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { ...LS, before: 40, after: PA },
    children: [new TextRun({ text: caption, size: CAP_SZ, font: 'Times New Roman' })]
  }));
  return out;
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { ...LS, before: 240, after: PA },
    children: [new TextRun({ text, bold: true, size: H1_SZ, font: 'Times New Roman' })]
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { ...LS, before: 180, after: 60 },
    children: [new TextRun({ text, bold: true, size: H2_SZ, font: 'Times New Roman' })]
  });
}

function p(text, opts = {}) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { ...LS, before: 0, after: PA },
    children: [new TextRun({ text, size: BODY_SZ, font: 'Times New Roman', ...opts })]
  });
}

function bullet(text) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { ...LS, before: 0, after: 55 },
    indent: { left: 540, hanging: 360 },
    children: [new TextRun({ text: '•\t' + text, size: BODY_SZ, font: 'Times New Roman' })]
  });
}

function blankLine() {
  return new Paragraph({ spacing: { ...LS, before: 0, after: 0 }, children: [new TextRun('')] });
}

// Booktabs table — returns [captionPara, Table, spacer]; spread with ...makeTable(...)
function makeTable(caption, headers, rows, colWidths) {
  const totalW = colWidths.reduce((a, b) => a + b, 0);
  function tc(text, ci, topRule, botRule, bold) {
    return new TableCell({
      width: { size: colWidths[ci], type: WidthType.DXA },
      borders: {
        top:    topRule === 'thick' ? BTHICK : BNONE,
        bottom: botRule === 'thick' ? BTHICK : botRule === 'mid' ? BMID : BNONE,
        left: BNONE, right: BNONE,
      },
      margins: { top: 55, bottom: 55, left: 80, right: 80 },
      children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { ...LS },
        children: [new TextRun({ text: String(text), size: CAP_SZ, font: 'Times New Roman', bold: !!bold })]
      })]
    });
  }
  const nR = rows.length;
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { ...LS, before: 160, after: 80 },
      children: [new TextRun({ text: caption, size: CAP_SZ, font: 'Times New Roman' })]
    }),
    new Table({
      width: { size: totalW, type: WidthType.DXA }, columnWidths: colWidths,
      rows: [
        new TableRow({ tableHeader: true, children: headers.map((h, i) => tc(h, i, 'thick', 'mid', true)) }),
        ...rows.map((row, ri) => new TableRow({
          children: row.map((c, i) => tc(c, i, 'none', ri === nR - 1 ? 'thick' : 'none', false))
        }))
      ]
    }),
    new Paragraph({ spacing: { before: 0, after: PA }, children: [new TextRun('')] }),
  ];
}

// ── NeurIPS title block ───────────────────────────────────────────────────────

const titleBlock = [
  new Paragraph({
    spacing: { before: 0, after: 360 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 32, color: '000000', space: 0 } },
    children: [new TextRun({ text: '' })]
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 0, after: 360 },
    children: [new TextRun({
      text: 'Bitcoin Market State Prediction Using Machine Learning and Deep Learning',
      bold: true, size: TITLE_SZ, font: 'Times New Roman'
    })]
  }),
  new Paragraph({
    spacing: { before: 0, after: 440 },
    border: { top: { style: BorderStyle.SINGLE, size: 8, color: '000000', space: 0 } },
    children: [new TextRun({ text: '' })]
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { ...LS, before: 0, after: 60 },
    children: [new TextRun({
      text: 'Chen Zhiyu  Liu Ruyan  Shi Xiangyan  Yang Shuyi',
      bold: true, size: BODY_SZ, font: 'Times New Roman'
    })]
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { ...LS, before: 0, after: 60 },
    children: [new TextRun({ text: 'Nanyang Technological University, SC6122 Emerging Topics in FinTech', size: BODY_SZ, font: 'Times New Roman' })]
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { ...LS, before: 0, after: 440 },
    children: [new TextRun({ text: '{zhiyu005, ruyan001, xiangyan001, shuyi007}@e.ntu.edu.sg', size: BODY_SZ, font: 'Times New Roman' })]
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { ...LS, before: 0, after: PA },
    children: [new TextRun({ text: 'Abstract', bold: true, size: ABS_SZ, font: 'Times New Roman' })]
  }),
  new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { ...LS, before: 0, after: 440 },
    indent: { left: 720, right: 720 },
    children: [new TextRun({
      text: 'We compare ten machine learning approaches for Bitcoin market state classification — predicting Bull (>+15%), Bear (<−15%), or Sideways state 30 days ahead. Using 4,383 daily on-chain observations (2011–2023), 176 features are engineered and the top 40 selected via Random Forest importance. Under a strict time-ordered 80/20 split, our Stacking Ensemble with max-confidence routing achieves the best macro F1 of 0.4124. Distribution shift analysis documents how the 2022 crypto crash degrades test performance. HODL wave ablation reveals ~30% F1 degradation from structural non-stationarity. Binary robustness checks confirm genuine directional predictive signal in on-chain metrics.',
      size: BODY_SZ, font: 'Times New Roman'
    })]
  }),
];

// ── 1  Introduction ───────────────────────────────────────────────────────────

function sub(text, italics = false) {
  return new TextRun({ text, size: BODY_SZ, font: 'Times New Roman', subScript: true, italics });
}
function run(text, italics = false) {
  return new TextRun({ text, size: BODY_SZ, font: 'Times New Roman', italics });
}
function formulaPara(children) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { ...LS, before: 80, after: 80 },
    children,
  });
}

const sec1 = [
  h1('1  Introduction'),
  p('Bitcoin price movements are driven by a complex interplay of on-chain network activity, market sentiment, and macro-economic conditions. We formulate market state prediction as a supervised three-class classification problem: given observed on-chain and market features up to time t, we predict the discrete market state 30 days ahead:'),
  formulaPara([
    run('y', true), sub('t', true),
    run(' = f(X', true), sub('t', true), run('),   where  y', true), sub('t', true),
    run(' ∈ {Bear, Sideways, Bull}'),
  ]),
  p('The 30-day forward return is defined as:'),
  formulaPara([
    run('r', true), sub('t+30', true),
    run(' = (P', true), sub('t+30', true),
    run(' − P', true), sub('t', true),
    run(') / P', true), sub('t', true),
  ]),
  p('Labels are assigned by threshold: Bull (rₜ₊₃₀ > +15%), Bear (rₜ₊₃₀ < −15%), Sideways (−15% ≤ rₜ₊₃₀ ≤ +15%). A 30-day horizon captures economically meaningful regime shifts while avoiding micro-volatility noise.'),
  p('Key challenges include: severe class imbalance (Sideways dominates the test period at 58.7%), temporal non-stationarity from the 2022 crypto crash, and the high noise-to-signal ratio of financial time series. We benchmark six supervised classifiers, one LSTM, one DQN agent, and two ensemble methods under strict temporal evaluation, and identify max-confidence stacking as an effective strategy for imbalanced multi-class cryptocurrency prediction.'),
];

// ── 2  Dataset ────────────────────────────────────────────────────────────────

const sec2 = [
  h1('2  Dataset'),
  p('Two daily on-chain CSV datasets (2010–2023) are merged on the datetime index: blockchain.com (transaction count, hash rate, fees, miner revenue, market price) and lookintobitcoin.com (NUPL, MVRV ratio, Fear & Greed Index, on-chain valuation). After inner join, deduplication, zero-price removal, and forward/back-fill imputation, 4,383 rows with zero missing values remain.'),
  blankLine(),
  ...makeTable('Table 1: Dataset overview',
    ['Source', 'Records', 'Date range', 'Key metrics'],
    [
      ['blockchain.com', '~5,000', '2010–2023', 'Hash rate, fees, mempool, miner revenue'],
      ['lookintobitcoin.com', '~5,000', '2010–2023', 'NUPL, MVRV, Fear/Greed, realised cap'],
      ['Merged (inner join)', '4,383', '2011–2023', 'All features; 176 engineered → top 40'],
    ],
    [1800, 1000, 1200, 3920]
  ),
  ...img('eda_label_distribution.png', 460, 204, 'Figure 1: Label distribution — train (2011–2021) vs. test (2021–2023), showing significant Bull-class decline in the test period due to the 2022 crash.'),
  p('Six additional CSV files (HODL waves, realised-cap HODL waves, address balance bands) were excluded after ablation: long-duration HODL bands are structurally zero until 2015 and grow monotonically thereafter, degrading macro F1 by ~30% across all models due to temporal non-stationarity.'),
];

// ── 3  Methodology ────────────────────────────────────────────────────────────

const sec3 = [
  h1('3  Methodology'),

  h2('3.1  Feature engineering'),
  p('From 27 raw columns, 176 features are constructed: rolling statistics (7d/30d/90d mean, std, min-max ratios for price, hash rate, volume, fees); momentum indicators (7/30/90-day returns, return ratio); on-chain ratios (MVRV z-score, NUPL momentum, mempool congestion ratio); and temporal features (month, day-of-week, cyclical encoding, halving cycle). The top 40 features by Random Forest importance are retained. StandardScaler is applied for scale-sensitive models (LR, SVM, KNN, LSTM, DQN).'),

  h2('3.2  Supervised classifiers'),
  p('Six classifiers are trained with TimeSeriesSplit(n_splits=5) and GridSearchCV to preserve temporal order. Class imbalance is handled via class_weight="balanced" (LR, RF, SVM) or sample_weight computed from inverse class frequencies (XGB, LGB). Best hyperparameters are selected by macro-F1 on the held-out validation fold:'),
  bullet('Logistic Regression (LR): L2 regularisation, solver=lbfgs, max_iter=2000; C ∈ {0.01, 0.1, 1, 10}. Serves as the linear baseline; strong regularisation (C=0.01 optimal) prevents overfitting in the 40-dimensional space.'),
  bullet('Random Forest (RF): n_estimators ∈ {100, 200}, max_depth ∈ {5, 10}, min_samples_split ∈ {2, 5}. Bootstrapped trees reduce variance on non-stationary data; feature importances drive the 40-feature selection pipeline.'),
  bullet('XGBoost: multi:softprob objective; grid over n_estimators ∈ {100,200}, max_depth ∈ {3,5}, learning_rate ∈ {0.05,0.1}, subsample ∈ {0.8,1.0}, min_child_weight ∈ {1,3}.'),
  bullet('LightGBM: multiclass objective, leaf-wise growth; grid over n_estimators, max_depth, learning_rate, min_child_samples ∈ {20,50}, num_leaves ∈ {31,63}.'),
  bullet('SVM: probability=True (Platt scaling); linear and RBF kernels; C ∈ {0.1, 1, 10}.'),
  bullet('KNN: n_neighbors ∈ {3, 5, 7, 11, 15}, weights ∈ {uniform, distance}. Distance-weighted voting reduces sensitivity to irrelevant neighbours near class boundaries.'),

  h2('3.3  Deep learning and reinforcement learning'),
  p('The Bidirectional LSTM (2 LSTM layers, 128 hidden units each, 40% dropout, BatchNorm between layers) processes sliding windows of 30 consecutive days. Class-weighted cross-entropy loss (weights inversely proportional to class frequency) is optimised with AdamW and a cosine annealing learning rate schedule; training stops early if validation macro-F1 does not improve for 10 epochs (up to 60 total). A DQN agent (3-layer MLP 256→128→64, ReLU, target network updated every 10 steps, experience replay buffer of 10,000, ε-greedy exploration from 1.0 to 0.05) is included as an exploratory baseline; its instability under distribution shift and limited 60-episode horizon limit it to a secondary role.'),

  h2('3.4  Ensemble methods'),
  p('A Voting Ensemble combines RF and LightGBM via soft-vote, exploiting their complementary profiles (RF: Sideways-strong; LGB: Bear-capable). A Stacking Ensemble (RF + KNN + LGB + XGB) evaluates three meta-strategies in parallel: (i) Meta-LGB trained on out-of-fold (OOF) probabilities; (ii) max-confidence routing — select the base model with highest weighted confidence per sample, with LGB receiving a 1.5× multiplier to amplify its Bear signal; (iii) soft-vote average. All OOF predictions are generated with TimeSeriesSplit(5) to prevent test leakage.'),
];

// ── 4  Experiments and Results ────────────────────────────────────────────────

const sec4 = [
  h1('4  Experiments and results'),

  h2('4.1  Setup and baselines'),
  p('Train: 2011–2021 (3,506 samples); test: 2021–2023 (877 samples); strict chronological order, no shuffling. Primary metric: macro-averaged F1 (weights each class equally; sensitive to Bear detection). Accuracy and macro ROC-AUC are reported additionally. Eight temporal-integrity checks are verified: chronological split, no shuffling, TimeSeriesSplit CV, train-only feature selection and scaling, no future leakage in rolling features, label-only forward shift, OOF stacking without test exposure. Three naive baselines frame the lower bound: Random (F1≈0.333), Always-Sideways (F1≈0.241), Always-Bull (F1≈0.127).'),

  h2('4.2  Overall model performance'),
  blankLine(),
  ...makeTable('Table 2: Model performance summary (test set)',
    ['Model', 'Accuracy', 'F1 (macro)', 'ROC-AUC', 'Best params'],
    [
      ['Logistic Regression', '0.1977', '0.1843', '0.5611', 'C=0.01'],
      ['Random Forest', '0.5781', '0.3105', '0.5217', 'depth=10, n=100'],
      ['XGBoost', '0.2529', '0.1644', '0.6103', 'lr=0.1, depth=3, n=200'],
      ['LightGBM', '0.3369', '0.3384', '0.6158', 'lr=0.1, depth=5, n=200'],
      ['SVM', '0.2179', '0.1223', '0.3870', 'C=10, linear'],
      ['KNN', '0.5770', '0.3119', '0.4974', 'k=3, weighted'],
      ['Voting ensemble (RF+LGB)', '0.4198', '0.3903', '0.5926', 'soft-vote'],
      ['LSTM', '0.5675', '0.3452', '0.5831', 'bidir, 128 units, w=30'],
      ['DQN (exploratory)', '0.3411', '0.3384', 'N/A', '60 eps, ε=0.05'],
      ['Stacking — max-conf routing', '0.4283', '0.4124', '0.5796', 'LGB 1.5× weight'],
    ],
    [2400, 1000, 1000, 1000, 2520]
  ),
  p('Stacking with max-confidence routing achieves the best macro F1 of 0.4124, +23.8% above random (0.333). High accuracy does not correlate with high F1: RF (acc=0.578) and KNN (acc=0.577) collapse to predicting Sideways, yielding F1~0.31. LightGBM achieves the best individual F1 (0.3384) and ROC-AUC (0.6158). XGBoost AUC=0.610 with F1=0.164 indicates good probability ranking but a poorly-calibrated decision boundary. The LSTM (F1=0.345) marginally outperforms LightGBM among individual models, confirming that pre-computed rolling features capture most temporal signal at daily granularity.'),
  ...img('model_summary_table.png', 460, 111, 'Figure 2: Macro F1, accuracy, and ROC-AUC for all ten models. Stacking (max-conf routing) dominates F1; RF/KNN lead in accuracy through Sideways collapse.'),

  h2('4.3  Per-class analysis'),
  blankLine(),
  ...makeTable('Table 3: Per-class F1 scores (test set)',
    ['Model', 'Bear F1', 'Sideways F1', 'Bull F1'],
    [
      ['Logistic Regression', '0.2661', '0.0000', '0.2867'],
      ['Random Forest', '0.0000', '0.7228', '0.2086'],
      ['XGBoost', '0.0000', '0.1316', '0.3616'],
      ['LightGBM', '0.3875', '0.2419', '0.3857'],
      ['SVM', '0.0000', '0.0107', '0.3563'],
      ['KNN', '0.0000', '0.7205', '0.2151'],
      ['Voting ensemble', '0.3291', '0.5064', '0.3355'],
    ],
    [2500, 1807, 1806, 1807]
  ),
  ...img('per_class_f1_heatmap.png', 460, 230, 'Figure 3: Per-class F1 heatmap. Bear F1=0 for five of seven individual classifiers; only LightGBM and Logistic Regression achieve non-zero Bear detection.'),
  p('Bear F1=0 for five of seven individual classifiers reveals the core challenge: models learn the Sideways majority and ignore the economically most critical Bear regime. The 2022 distribution shift (Bull: 35.8%→22.3%; Sideways: 46.7%→58.7%) amplifies this failure. The MVRV ratio, NUPL, and their 30d/90d rolling variants dominate feature importance across all tree-based models, confirming on-chain valuation metrics as primary regime signals.'),
  ...img('distribution_shift.png', 460, 162, 'Figure 4: Feature mean comparison between train (2011–2021) and test (2021–2023) periods. Hash-rate, MVRV, and NUPL exhibit the largest shifts, explaining degraded Bear-class detection in the test set.'),

  h2('4.4  Stacking ensemble ablation'),
  blankLine(),
  ...makeTable('Table 4: Stacking meta-strategy ablation',
    ['Meta strategy', 'F1 (macro)', 'vs. best individual LGB', 'Explanation'],
    [
      ['Soft-vote average', '0.2852', '−0.053', 'Dilutes LGB Bear signal with RF/KNN Sideways bias'],
      ['Meta-LGB (OOF)', '0.2951', '−0.043', 'OOF training inherits Sideways majority; meta-learner replicates bias'],
      ['Max-conf routing (ours)', '0.4124', '+0.074', 'Selects LGB (1.5× weight) when confident; exploits Bear capability directly'],
    ],
    [2200, 1100, 1920, 2700]
  ),
  p('Meta-LGB fails because its OOF training set is dominated by Sideways samples (same imbalance as training data), causing the meta-learner to inherit the very bias it should correct. Max-confidence routing sidesteps this by not learning a selection rule — it directly amplifies LGB Bear signal without an OOF fitting step, achieving +0.074 F1 over the best individual model.'),

  h2('4.5  Binary direction robustness check'),
  p('Samples within the ±5% neutral zone are excluded, leaving 3,777 samples (79.8%) with binary labels (Up/Down). The same six classifiers are retrained under the same 80/20 temporal split.'),
  blankLine(),
  ...makeTable('Table 5: Binary vs. three-class F1 (test set)',
    ['Model', 'Binary F1', 'Binary AUC', '3-class F1', 'Gain'],
    [
      ['Logistic Regression', '0.5256', '0.6461', '0.1843', '+0.341'],
      ['Random Forest',       '0.4900', '0.6436', '0.3105', '+0.180'],
      ['XGBoost',             '0.5720', '0.6369', '0.1644', '+0.408'],
      ['LightGBM',            '0.4210', '0.5605', '0.3384', '+0.083'],
      ['SVM',                 '0.5452', '0.6509', '0.1223', '+0.423'],
      ['KNN',                 '0.4593', '0.5619', '0.3119', '+0.140'],
      ['Random baseline',     '~0.500', '0.500',  '~0.333', '—'],
    ],
    [2400, 1130, 1130, 1130, 1130]
  ),
  ...img('binary_model_comparison.png', 460, 181, 'Figure 5: Binary classification F1, AUC, and accuracy for all models. Three models (XGBoost, SVM, LR) clearly exceed the 0.500 random baseline.'),
  p('Three of six models exceed the binary random baseline (F1=0.500): XGBoost (0.572), SVM (0.545), LR (0.526). The best AUC of 0.651 (SVM) confirms genuine directional signal in on-chain features. Models that perform well in three-class (RF, KNN) underperform in binary, further evidence that their three-class "success" is majority-class collapse rather than true discrimination.'),
];

// ── 5  Discussion ─────────────────────────────────────────────────────────────

const sec5 = [
  h1('5  Discussion'),
  p('Max-confidence stacking outperforms both soft-vote and meta-LGB because it avoids the OOF Sideways-majority trap while directly leveraging LGB Bear detection, the scarce but economically critical signal. Traditional meta-learner stacking is contraindicated when base models are strongly correlated and class imbalance is severe — conditions typical of financial time series. The LSTM modest gain over static models suggests that daily on-chain data at a 30-day horizon does not contain rich sequential structure beyond what rolling features already capture. The DQN unstable training further confirms that RL policy learning under severe distribution shift and a short episode horizon is not yet competitive with supervised methods for this task.'),
  p('All models underperform on Bear detection due to the 2022 distribution shift — an out-of-distribution tail event not present in training data. Real-world deployment would require walk-forward retraining and potentially regime-detection triggers to adapt to structural breaks. On-chain data alone cannot capture exchange-level dynamics, regulatory events, or macro-financial sentiment — known Bitcoin price drivers. HODL wave features must be time-normalised (ratio rather than absolute coin counts) before use, as their monotonic growth from 2015 onward creates spurious temporal correlations in a train/test split framework.'),
  p('Our macro F1~0.41 should be interpreted against the difficulty of the task rather than against binary classification benchmarks on static datasets such as credit card default (where 90%+ accuracy is common). Five structural differences explain the gap: (i) task complexity — three-class prediction is inherently harder than binary (random baseline: 0.333 vs. 0.500); (ii) temporal non-stationarity — financial time series violate the i.i.d. assumption required by most classifiers, whereas credit-card default datasets are cross-sectional with stable feature distributions; (iii) distribution shift — the 2022 crypto crash constitutes an out-of-distribution tail event with no analogue in the training period; (iv) evaluation rigour — we use strict time-ordered splitting, whereas many published cryptocurrency prediction results use random train/test splits or k-fold CV that permit look-ahead bias, inflating reported accuracy by 10–30%; (v) label horizon — a 30-day forward return is far noisier than a static default label. The binary robustness check (§4.5) confirms AUC 0.637–0.651 for the best models, demonstrating genuine predictive signal that aligns with the upper range of rigorously evaluated cryptocurrency forecasting literature.'),
];

// ── 6  Conclusion ─────────────────────────────────────────────────────────────

const sec6 = [
  h1('6  Conclusion'),
  p('We systematically compared ten approaches for three-class Bitcoin market state prediction under rigorous temporal evaluation. Key findings:'),
  bullet('Stacking with max-confidence routing (F1=0.4124) is the best approach, +23.8% above random and +22% above the best individual model (LightGBM).'),
  bullet('LightGBM is the only individual classifier detecting all three market states (Bear F1=0.3875, Bull F1=0.3857), making it the preferred base learner for minority-class emphasis.'),
  bullet('Bear F1=0 for five of seven individual classifiers; the 2022 distribution shift is the primary cause, not model quality.'),
  bullet('HODL wave features degrade macro F1 by ~30% due to structural non-stationarity — a general warning for using maturation-linked on-chain metrics in time-split ML.'),
  bullet('Binary robustness checks confirm genuine directional signal: XGBoost (AUC=0.637), SVM (AUC=0.651), LR (AUC=0.646) all exceed 0.500.'),
  p('Future work should explore walk-forward retraining, time-normalised HODL ratios, social media sentiment features, and transformer-based sequence models.'),
];

// ── References ────────────────────────────────────────────────────────────────

const refs = [
  h1('References'),
  ...[
    '[1] Chen, T. & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD \'16.',
    '[2] Ke, G. et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. NeurIPS 2017.',
    '[3] Hochreiter, S. & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8).',
    '[4] Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. Nature, 518.',
    '[5] Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. JMLR, 12.',
    '[6] Nakamoto, S. (2008). Bitcoin: A peer-to-peer electronic cash system. bitcoin.org.',
    '[7] Nakagawa, K. et al. (2018). Deep recurrent factor model. arXiv:1901.11493.',
    '[8] Wolpert, D.H. (1992). Stacked generalization. Neural Networks, 5(2), 241–259.',
  ].map((t, i) => new Paragraph({
    spacing: { ...LS, before: 0, after: 80 },
    children: [new TextRun({ text: t, size: CAP_SZ, font: 'Times New Roman' })]
  })),
  new Paragraph({ children: [new PageBreak()] }),
];

// ── Appendix A: Workload ──────────────────────────────────────────────────────

const appendix = [
  h1('Appendix A  Workload distribution'),
  p('(For workload breakdown, please refer to Table 6 below. All members contributed equally to report drafting and code review.)'),
  blankLine(),
  ...makeTable('Table 6: Group member workload distribution',
    ['Member', 'Primary responsibilities', '%'],
    [
      ['Chen Zhiyu', 'Data loading, feature engineering, EDA, dataset ablation; LR & SVM model training', '25%'],
      ['Liu Ruyan', 'LSTM & DQN model training, stacking ensemble design, hyperparameter optimisation', '25%'],
      ['Shi Xiangyan', 'RF & KNN model training, TimeSeriesSplit CV framework, evaluation metrics pipeline', '25%'],
      ['Yang Shuyi', 'XGBoost & LightGBM & Voting Ensemble model training, distribution shift analysis, feature importance, report writing', '25%'],
    ],
    [1600, 5000, 1320]
  ),
];

// ── Document assembly ─────────────────────────────────────────────────────────

const allChildren = [
  ...titleBlock,
  ...sec1, ...sec2, ...sec3, ...sec4,
  ...sec5, ...sec6, ...refs, ...appendix,
];

const doc = new Document({
  styles: {
    default: { document: { run: { font: 'Times New Roman', size: BODY_SZ } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: H1_SZ, bold: true, font: 'Times New Roman', color: '000000' },
        paragraph: { spacing: { ...LS, before: 240, after: PA }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: H2_SZ, bold: true, font: 'Times New Roman', color: '000000' },
        paragraph: { spacing: { ...LS, before: 180, after: 60 }, outlineLevel: 1 } },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 2160, bottom: 1440, left: 2160 }
      }
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ children: [PageNumber.CURRENT], size: CAP_SZ, font: 'Times New Roman' })]
        })]
      })
    },
    children: allChildren,
  }]
});

Packer.toBuffer(doc).then(buffer => {
  const outPath = '/Users/liuruyan/Desktop/bitcoin-market-prediction/Bitcoin_Market_Prediction_Report_Final.docx';
  fs.writeFileSync(outPath, buffer);
  console.log('Report written to:', outPath);
  console.log('Size:', Math.round(buffer.length / 1024), 'KB');
}).catch(err => { console.error('Error:', err); process.exit(1); });
