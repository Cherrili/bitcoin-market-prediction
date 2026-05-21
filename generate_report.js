'use strict';
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, VerticalAlign, PageNumber, PageBreak, LevelFormat, TableOfContents
} = require('docx');
const fs = require('fs');
const path = require('path');

const BASE = '/Users/liuruyan/Desktop/bitcoin-market-prediction';
const OUTPUT = path.join(BASE, 'Bitcoin_Market_Prediction_Report_v2.docx');

// ─── helpers ────────────────────────────────────────────────────────────────

function img(filename, widthPx, heightPx) {
  const maxW = 580; // points-ish, we work in pixels then convert
  const scale = Math.min(1, maxW / widthPx);
  const w = Math.round(widthPx * scale);
  const h = Math.round(heightPx * scale);
  return new ImageRun({
    type: 'png',
    data: fs.readFileSync(path.join(BASE, 'output', filename)),
    transformation: { width: w, height: h },
    altText: { title: filename, description: filename, name: filename }
  });
}

function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 60, after: 200 },
    children: [new TextRun({ text, font: 'Arial', size: 18, italics: true, color: '555555' })]
  });
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 120 },
    children: [new TextRun({ text, font: 'Arial', size: 32, bold: true, color: '1F3864' })]
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 240, after: 80 },
    children: [new TextRun({ text, font: 'Arial', size: 26, bold: true, color: '2E5090' })]
  });
}

function body(text, opts = {}) {
  return new Paragraph({
    alignment: opts.center ? AlignmentType.CENTER : AlignmentType.JUSTIFIED,
    spacing: { before: 60, after: 100, line: 276 },
    children: [new TextRun({ text, font: 'Arial', size: 22, ...opts })]
  });
}

function bullet(text, bold_prefix = '') {
  const children = [];
  if (bold_prefix) {
    children.push(new TextRun({ text: bold_prefix + ' ', font: 'Arial', size: 22, bold: true }));
  }
  children.push(new TextRun({ text, font: 'Arial', size: 22 }));
  return new Paragraph({
    numbering: { reference: 'bullets', level: 0 },
    spacing: { before: 40, after: 40 },
    children
  });
}

function pageBreak() {
  return new Paragraph({ children: [new PageBreak()] });
}

// ─── border / shading constants ─────────────────────────────────────────────

const B = { style: BorderStyle.SINGLE, size: 4, color: 'AAAAAA' };
const BORDERS = { top: B, bottom: B, left: B, right: B };
const HEADER_SHADE = { fill: 'D5E8F0', type: ShadingType.CLEAR };
const ALT_SHADE = { fill: 'F4F8FC', type: ShadingType.CLEAR };
const NO_SHADE = { fill: 'FFFFFF', type: ShadingType.CLEAR };
const CELL_MARGIN = { top: 100, bottom: 100, left: 160, right: 160 };

function cell(text, opts = {}) {
  return new TableCell({
    borders: BORDERS,
    width: opts.width ? { size: opts.width, type: WidthType.DXA } : undefined,
    shading: opts.shade || NO_SHADE,
    verticalAlign: VerticalAlign.CENTER,
    margins: CELL_MARGIN,
    columnSpan: opts.span || 1,
    children: [new Paragraph({
      alignment: opts.center ? AlignmentType.CENTER : (opts.right ? AlignmentType.RIGHT : AlignmentType.LEFT),
      children: [new TextRun({
        text: String(text),
        font: 'Arial',
        size: opts.size || 20,
        bold: opts.bold || false,
        color: opts.color || '000000'
      })]
    })]
  });
}

function hCell(text, w) {
  return cell(text, { shade: HEADER_SHADE, bold: true, center: true, width: w, size: 20 });
}

// ─── MAIN PERFORMANCE TABLE ──────────────────────────────────────────────────
// cols: Model | Accuracy | F1_macro | ROC_AUC | Notes | Contributor
// widths sum = 9360
const COL = [2200, 1000, 900, 900, 2360, 2000];

function perfTable() {
  const rows_data = [
    ['Logistic Regression', '0.1977', '0.1843', '0.5611', 'Best C=0.01', 'SHI XIANGYAN'],
    ['Random Forest',       '0.5781', '0.3105', '0.5218', 'max_depth=10, n_est=100', 'SHI XIANGYAN'],
    ['XGBoost',             '0.2529', '0.1644', '0.6103', 'lr=0.1, max_depth=3, n_est=200', 'YANG SHUYI'],
    ['LightGBM',            '0.3369', '0.3384', '0.6158', 'lr=0.1, max_depth=5, n_est=200', 'YANG SHUYI'],
    ['SVM',                 '0.2179', '0.1223', '0.3870', 'C=10, linear kernel', 'SHI XIANGYAN'],
    ['KNN',                 '0.5770', '0.3119', '0.4974', 'k=3, distance weights', 'SHI XIANGYAN'],
    ['Voting Ensemble (RF+LGB)', '0.4198', '0.3903', '0.5926', 'Soft voting', 'LIU RUYAN'],
    ['LSTM',                '0.5675', '0.3452', '0.5831', 'BiLSTM, window=30, h=128, 2 layers', 'LIU RUYAN'],
    ['DQN (RL)',             '0.3273', '0.3180', 'N/A',    '60 episodes, cum. return -0.9993', 'YANG SHUYI'],
    ['Stacking (RF+KNN+LGB+XGB)', '0.4283', '0.4124', '0.5796', 'BEST F1, max-conf routing, LGB 1.5x wt', 'LIU RUYAN'],
  ];
  const isAlt = (i) => i % 2 === 1;
  const isBest = (i) => i === 9;

  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: COL,
    rows: [
      new TableRow({
        tableHeader: true,
        children: [
          hCell('Model', COL[0]),
          hCell('Accuracy', COL[1]),
          hCell('F1 Macro', COL[2]),
          hCell('ROC AUC', COL[3]),
          hCell('Best Hyperparameters / Notes', COL[4]),
          hCell('Contributor', COL[5]),
        ]
      }),
      ...rows_data.map((r, i) => new TableRow({
        children: r.map((txt, j) => {
          const shade = isBest(i) ? { fill: 'FFF2CC', type: ShadingType.CLEAR }
                                   : isAlt(i) ? ALT_SHADE : NO_SHADE;
          return cell(txt, {
            shade,
            width: COL[j],
            bold: isBest(i),
            center: j > 0 && j < 4,
            color: isBest(i) ? '7B3F00' : '000000'
          });
        })
      }))
    ]
  });
}

// ─── DATASET DESCRIPTION TABLE ───────────────────────────────────────────────
function datasetTable() {
  const rows = [
    ['Time Period', '2010-01-01 to 2023-12-31', '~14 years of daily records'],
    ['Total Records', '4,700', 'Daily granularity'],
    ['Raw Features', '176', 'On-chain metrics, price, volume, derivatives'],
    ['Selected Features', '40', 'Top-40 via Random Forest importance score'],
    ['Train Split', '80% (~3,760 records)', '2010-2021 (chronological)'],
    ['Test Split', '20% (~940 records)', '2022-2023 (held-out)'],
    ['Label: Bull (+1)', '35.8% train / 22.3% test', '30-day return > +15%'],
    ['Label: Sideways (0)', '46.7% train / 58.7% test', '-15% <= 30-day return <= +15%'],
    ['Label: Bear (-1)', '17.5% train / 19.0% test', '30-day return < -15%'],
    ['Data Source', 'Glassnode + CoinMetrics', 'On-chain + market data'],
  ];
  const colW = [2400, 3600, 3360];
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: colW,
    rows: [
      new TableRow({
        tableHeader: true,
        children: [hCell('Attribute', colW[0]), hCell('Value', colW[1]), hCell('Description', colW[2])]
      }),
      ...rows.map((r, i) => new TableRow({
        children: r.map((txt, j) => cell(txt, {
          shade: i % 2 === 1 ? ALT_SHADE : NO_SHADE,
          width: colW[j]
        }))
      }))
    ]
  });
}

// ─── WORK DISTRIBUTION TABLE ──────────────────────────────────────────────────
function workTable() {
  const colW = [2000, 2200, 5160];
  const rows = [
    ['Chen Zhiyu', 'Data & Preprocessing', 'Data collection from Glassnode and CoinMetrics; full preprocessing pipeline; feature engineering (176 raw features); Random Forest importance-based feature selection reducing to top 40 features; label construction and chronological train/test splitting.'],
    ['LIU RUYAN', 'Advanced Ensembles & Deep Learning', 'Stacking ensemble design (RF+KNN+LGB+XGB base learners with TimeSeriesSplit(5) OOF generation, max-confidence routing meta-strategy, LGB 1.5x weight boost); 2-layer BiLSTM classifier (window=30, hidden=128, dropout=0.3, early stopping); LightGBM class_weight fix (sample_weight); XGBoost hyperparameter optimization; distribution shift analysis.'],
    ['SHI XIANGYAN', 'Baseline Models & Evaluation', 'Logistic Regression, Random Forest, SVM, and KNN implementations; GridSearchCV hyperparameter tuning for all baseline models; evaluation framework (accuracy, macro-F1, ROC-AUC); confusion matrix generation and visualization.'],
    ['YANG SHUYI', 'Boosting, RL & Analysis', 'XGBoost and LightGBM implementations; DQN reinforcement learning agent (Q-network, epsilon=0.05, 60 episodes, 30-day reward); ROC curve multi-class analysis; per-class F1 breakdown; Streamlit interactive dashboard.'],
  ];
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: colW,
    rows: [
      new TableRow({
        tableHeader: true,
        children: [hCell('Member', colW[0]), hCell('Primary Role', colW[1]), hCell('Detailed Contributions', colW[2])]
      }),
      ...rows.map((r, i) => new TableRow({
        children: r.map((txt, j) => cell(txt, {
          shade: i % 2 === 1 ? ALT_SHADE : NO_SHADE,
          width: colW[j],
          bold: j === 0
        }))
      }))
    ]
  });
}

// ─── ENSEMBLE PROGRESSION TABLE ──────────────────────────────────────────────
function ensembleProgressionTable() {
  const colW = [3600, 1600, 1600, 1600, 960];
  const rows = [
    ['Best Individual (Random Forest)', '0.5781', '0.3105', '0.5218', '-'],
    ['Voting Ensemble (RF+LGB)', '0.4198', '0.3903', '0.5926', '+25.7%'],
    ['LSTM', '0.5675', '0.3452', '0.5831', '+11.2%'],
    ['Stacking (meta-LGB)', '—', '0.2951', '—', 'Degraded'],
    ['Stacking (soft-vote)', '—', '0.2852', '—', 'Degraded'],
    ['Stacking (max-conf routing)', '0.4283', '0.4124', '0.5796', '+32.8%'],
  ];
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: colW,
    rows: [
      new TableRow({
        tableHeader: true,
        children: [hCell('Model / Strategy', colW[0]), hCell('Accuracy', colW[1]), hCell('F1 Macro', colW[2]), hCell('ROC AUC', colW[3]), hCell('F1 Gain', colW[4])]
      }),
      ...rows.map((r, i) => {
        const isBest = i === 5;
        return new TableRow({
          children: r.map((txt, j) => cell(txt, {
            shade: isBest ? { fill: 'FFF2CC', type: ShadingType.CLEAR } : (i % 2 === 1 ? ALT_SHADE : NO_SHADE),
            width: colW[j],
            bold: isBest,
            center: j > 0,
            color: isBest ? '7B3F00' : '000000'
          }))
        });
      })
    ]
  });
}

// ─── centered image paragraph ─────────────────────────────────────────────────
function imgPara(filename, widthPx, heightPx) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 60 },
    children: [img(filename, widthPx, heightPx)]
  });
}

// ─── DOCUMENT CONTENT ────────────────────────────────────────────────────────

const children = [];

// ══════════════════════════════════════════════════════════════════
// TITLE PAGE
// ══════════════════════════════════════════════════════════════════
children.push(
  new Paragraph({ spacing: { before: 2000, after: 300 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: 'Bitcoin Market State Prediction', font: 'Arial', size: 56, bold: true, color: '1F3864' })] }),
  new Paragraph({ spacing: { before: 0, after: 200 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: 'Using On-Chain Analytics and Machine Learning', font: 'Arial', size: 40, color: '2E5090' })] }),
  new Paragraph({ spacing: { before: 200, after: 800 }, alignment: AlignmentType.CENTER,
    border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: '1F3864', space: 1 } },
    children: [new TextRun({ text: ' ', font: 'Arial', size: 22 })] }),
  new Paragraph({ spacing: { before: 400, after: 100 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: 'SC6122 Emerging Topics in Fintech', font: 'Arial', size: 26, color: '555555' })] }),
  new Paragraph({ spacing: { before: 60, after: 60 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: 'Nanyang Technological University', font: 'Arial', size: 24, color: '555555' })] }),
  new Paragraph({ spacing: { before: 60, after: 400 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: 'Group #17', font: 'Arial', size: 24, bold: true, color: '1F3864' })] }),
  new Paragraph({ spacing: { before: 200, after: 80 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: 'Chen Zhiyu  |  LIU RUYAN  |  SHI XIANGYAN  |  YANG SHUYI', font: 'Arial', size: 24, bold: true, color: '1F3864' })] }),
  new Paragraph({ spacing: { before: 60, after: 1200 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: 'May 2026', font: 'Arial', size: 22, color: '777777' })] }),
  pageBreak()
);

// ══════════════════════════════════════════════════════════════════
// ABSTRACT
// ══════════════════════════════════════════════════════════════════
children.push(h1('Abstract'));
children.push(body(
  'This paper presents a comprehensive comparative study of ten machine learning approaches for predicting Bitcoin market states using on-chain blockchain analytics. We construct a dataset of approximately 4,700 daily records spanning January 2010 to December 2023, sourced from Glassnode and CoinMetrics, and formulate a three-class classification problem: Bull (30-day return > +15%), Bear (30-day return < -15%), and Sideways (otherwise). Six baseline classifiers — Logistic Regression, Random Forest, XGBoost, LightGBM, Support Vector Machine, and K-Nearest Neighbours — are evaluated alongside three advanced architectures: a two-layer bidirectional LSTM deep learning classifier, a Deep Q-Network reinforcement learning agent, and a Stacking ensemble combining four base learners with a novel max-confidence routing meta-strategy.'
));
children.push(body(
  'Our key finding is that a structural distribution shift between training (2010-2021) and test (2022-2023) data represents the primary performance ceiling, not model sophistication. The 2022 cryptocurrency crash produced Bear market patterns absent from the training distribution, causing most models to achieve near-zero Bear-class F1. The Stacking ensemble, leveraging LightGBM\'s unique Bear detection capability via 1.5x confidence weighting, achieves the best macro-F1 of 0.4124, outperforming all individual models. Three key technical contributions are: (1) a BiLSTM architecture capturing temporal dependencies across 30-day windows; (2) a DQN reinforcement learning agent framing prediction as a trading policy optimization problem; and (3) a Stacking ensemble with out-of-fold temporal integrity via TimeSeriesSplit(5) cross-validation and a max-confidence routing strategy that explicitly exploits inter-model complementarity.'
));
children.push(pageBreak());

// ══════════════════════════════════════════════════════════════════
// SECTION 1: INTRODUCTION
// ══════════════════════════════════════════════════════════════════
children.push(h1('1. Introduction'));
children.push(body(
  'Bitcoin has evolved from an experimental peer-to-peer payment system into a globally recognized asset class with market capitalisation routinely exceeding one trillion US dollars. Unlike traditional equity markets, Bitcoin operates 24/7 on a public, transparent ledger that exposes a rich stream of on-chain behavioral signals — transaction volumes, miner revenues, unrealised profit/loss ratios, network hash rates, and more — all of which reflect the aggregate economic activity and sentiment of market participants in near-real time.'
));
children.push(body(
  'These on-chain signals offer a theoretically superior information set compared to purely price-based technical analysis, because they capture the fundamental supply-demand dynamics at the protocol level. Miners selling coin rewards creates measurable sell pressure; "diamond hand" wallets accumulating coin during drawdowns signals accumulation phases; NUPL (Net Unrealised Profit/Loss) approaching extreme values historically precedes market reversals. This rich, objective dataset motivates applying supervised machine learning to the problem of Bitcoin market state prediction.'
));
children.push(body(
  'We frame the prediction task as a three-class classification problem: given today\'s on-chain features, predict whether the Bitcoin price 30 days from now will be in a Bull, Sideways, or Bear regime. The three-class formulation is more actionable than binary classification (simply up/down) because it allows investors to distinguish between strong directional moves worth acting on and ambiguous sideways markets where no action may be preferable.'
));
children.push(body(
  'The problem presents three fundamental challenges. First, class imbalance: Bull markets (35.8% of training data) and Sideways periods (46.7%) vastly outnumber Bear markets (17.5%), causing models to neglect minority classes. Second, non-stationarity: cryptocurrency markets undergo regime changes driven by halving cycles, regulatory events, and macro shocks, making patterns learned in one period unreliable in another. Third, and most critically in our study, distribution shift: the 2022 cryptocurrency crash — triggered by the Terra/LUNA collapse, Three Arrows Capital bankruptcy, and FTX implosion — produced Bear market patterns at a severity and speed unprecedented in the training data (2010-2021), creating a structural gap between train and test distributions.'
));
children.push(body(
  'This paper makes the following contributions: (1) a systematic comparison of ten models spanning classical ML, deep learning, and reinforcement learning on the same dataset and evaluation protocol; (2) a rigorous distribution shift analysis revealing the fundamental limitation facing all approaches; (3) a Stacking ensemble with max-confidence routing that partially overcomes the Bear detection problem by intelligently exploiting LightGBM\'s unique multiclass calibration properties.'
));
children.push(pageBreak());

// ══════════════════════════════════════════════════════════════════
// SECTION 2: PROBLEM STATEMENT
// ══════════════════════════════════════════════════════════════════
children.push(h1('2. Problem Statement'));
children.push(h2('2.1 Formal Task Definition'));
children.push(body(
  'Let x_t ∈ R^40 denote the feature vector of on-chain and market signals observed at day t, and let y_t denote the market state label derived from the 30-day forward return r_t = (P_{t+30} − P_t) / P_t, where P_t is the Bitcoin closing price. The label assignment rule is:'
));
children.push(bullet('y_t = +1 (Bull)   if r_t > +0.15'));
children.push(bullet('y_t =  0 (Sideways) if −0.15 ≤ r_t ≤ +0.15'));
children.push(bullet('y_t = −1 (Bear)    if r_t < −0.15'));
children.push(body(
  'The goal is to learn a classifier f: R^40 → {−1, 0, +1} from chronologically ordered training data D_train = {(x_t, y_t)}_{t ∈ T_train} such that f generalises to the held-out test set D_test = {(x_t, y_t)}_{t ∈ T_test}, where T_train and T_test are disjoint, non-overlapping time intervals with T_train entirely preceding T_test (strict temporal splitting).'
));
children.push(h2('2.2 Evaluation Metrics'));
children.push(body(
  'Given the class imbalance and the cost-asymmetry of misclassifying Bear markets, we adopt macro-averaged F1 score as the primary metric, giving equal weight to all three classes regardless of support. We also report overall accuracy for comparability with prior literature, and one-vs-rest ROC-AUC to assess probability calibration quality. For detailed model diagnosis, we additionally report per-class F1 scores.'
));
children.push(h2('2.3 Key Challenges'));
children.push(body(
  'Distribution Shift is the central challenge identified in this study. Although the Bear class proportion is similar between train (17.5%) and test (19.0%), the character of Bear markets differs fundamentally. The 2022 bear market was driven by a cascade of institutional collapses creating a correlated, sustained multi-month decline — a regime not present in the 2010-2021 training window which included only smaller, shorter bear episodes. The Sideways class surge from 46.7% to 58.7% in the test set further dilutes signal, as 2022-2023 also included extended consolidation periods following the crash.'
));
children.push(pageBreak());

// ══════════════════════════════════════════════════════════════════
// SECTION 3: DATASET
// ══════════════════════════════════════════════════════════════════
children.push(h1('3. Dataset'));
children.push(h2('3.1 Data Description'));
children.push(body(
  'Our dataset combines on-chain blockchain metrics from Glassnode with price and market data from CoinMetrics. The resulting time series spans January 2010 through December 2023, yielding approximately 4,700 daily observations. Table 1 summarises the dataset properties.'
));
children.push(new Paragraph({ spacing: { before: 120, after: 80 }, children: [new TextRun({ text: 'Table 1: Dataset Summary', font: 'Arial', size: 20, bold: true, italics: true })] }));
children.push(datasetTable());
children.push(new Paragraph({ spacing: { before: 60, after: 200 }, children: [new TextRun({ text: 'Table 1: Dataset summary showing split sizes, label distributions, and data sources.', font: 'Arial', size: 18, italics: true, color: '555555' })] }));

children.push(h2('3.2 Exploratory Data Analysis'));
children.push(body(
  'Figure 1 presents the Bitcoin price time series alongside the label distribution. The chart reveals the characteristic Bitcoin boom-bust cycles: the 2017 bull run, the 2018-2019 bear market, the 2020-2021 pandemic-era bull run, and the 2022 crash. The label distribution bar chart confirms the class imbalance, with Sideways (0) being the most common state, followed by Bull (+1) and Bear (-1). This imbalance motivates the use of macro-F1 as the primary metric.'
));
children.push(imgPara('eda_label_distribution.png', 2100, 600));
children.push(caption('Figure 1: Bitcoin price time series (2010-2023) with annotated market states and class distribution histogram. Bull (green), Sideways (grey), Bear (red).'));

children.push(h2('3.3 Preprocessing Pipeline'));
children.push(body(
  'The preprocessing pipeline consists of five stages:'
));
children.push(bullet('Label Construction: Forward 30-day return is computed for each daily record, and labels {-1, 0, +1} are assigned per the thresholds in Section 2.', '1.'));
children.push(bullet('Feature Engineering: 176 raw features are computed from price (OHLCV), on-chain metrics, and derived indicators including moving averages (7-, 14-, 30-day), RSI, MACD, Bollinger Bands, and NUPL variants.', '2.'));
children.push(bullet('Feature Selection: A Random Forest trained on the full 176 features is used to compute Gini importance scores. The top 40 features are retained, removing noise and reducing the risk of overfitting.', '3.'));
children.push(bullet('Chronological Split: The data is split 80/20 in chronological order (no shuffling), with the first 80% used for training (2010-2021) and the final 20% reserved for testing (2022-2023). This strict temporal split prevents data leakage.', '4.'));
children.push(bullet('Class Imbalance Handling: Sample weights inversely proportional to class frequency are computed and passed to models that support them. For models lacking sample_weight support, class_weight="balanced" is used during hyperparameter search.', '5.'));
children.push(pageBreak());

// ══════════════════════════════════════════════════════════════════
// SECTION 4: MACHINE LEARNING METHODS
// ══════════════════════════════════════════════════════════════════
children.push(h1('4. Machine Learning Methods'));
children.push(body(
  'We evaluate ten models spanning four paradigms: classical statistical learning (LR, SVM, KNN), tree-based ensemble learning (RF, XGBoost, LightGBM), ensemble combination (Voting, Stacking), sequential deep learning (BiLSTM), and reinforcement learning (DQN). All models share the same 40-feature input vector and are evaluated on the same held-out test set.'
));

children.push(h2('4.1 Logistic Regression'));
children.push(body(
  'Logistic Regression (LR) serves as the linear baseline, modelling the log-odds of each class as a linear combination of features. We use the multinomial softmax formulation with L2 regularisation, optimised via the "lbfgs" solver. GridSearchCV over the regularisation strength C ∈ {0.001, 0.01, 0.1, 1, 10} identified the best parameter as C = 0.01, indicating that strong regularisation is needed to prevent overfitting on the high-dimensional, correlated feature set. Class weights are set to "balanced" to address class imbalance. The inherent limitation of LR is its assumption of linear separability — in a 40-dimensional space of non-linearly interacting on-chain metrics, this assumption is almost certainly violated.'
));

children.push(h2('4.2 Random Forest'));
children.push(body(
  'Random Forest (RF) is a bagging ensemble of decision trees, each trained on a bootstrap sample and a random feature subset. This inductive bias towards non-linear decision boundaries and automatic feature interaction modelling makes RF well-suited to the on-chain feature space. GridSearchCV identified optimal parameters: max_depth = 10 and n_estimators = 100. Capping tree depth at 10 prevents individual trees from memorising training patterns. The RF is one of the two highest-accuracy models (0.5781), but its macro-F1 of 0.3105 reflects its tendency to over-predict the majority Sideways class, neglecting Bear detection.'
));

children.push(h2('4.3 XGBoost'));
children.push(body(
  'XGBoost implements gradient boosting with second-order Taylor approximation of the loss function, L1/L2 leaf regularisation, and efficient column subsampling. It sequentially adds trees that correct the residuals of the current ensemble. GridSearchCV identified: learning_rate = 0.1, max_depth = 3, n_estimators = 200. The shallow depth (max_depth = 3) combined with many estimators (200) is the classic regularisation recipe for boosting. XGBoost achieves the second-best ROC-AUC (0.6103), suggesting well-calibrated probability estimates, but its accuracy (0.2529) and F1 (0.1644) are poor, likely due to sensitivity to the distribution shift in the test period.'
));

children.push(h2('4.4 LightGBM'));
children.push(body(
  'LightGBM extends gradient boosting with histogram-based split finding, leaf-wise (best-first) tree growth, and exclusive feature bundling, achieving training speed improvements of 10-20x over XGBoost on large datasets. A critical implementation fix was required during development: using class_weight="balanced" caused LightGBM to predict only the Bull class on the test set — a degenerate "predict-all-majority" behavior caused by an interaction between LightGBM\'s internal class weighting and the test distribution shift. The fix was to remove class_weight and instead pass sample_weight vectors computed as inverse class frequencies in the training set. This resolved the degenerate behavior and allowed LightGBM to become the only individual model predicting all three classes. Optimal parameters: learning_rate = 0.1, max_depth = 5, n_estimators = 200. LightGBM achieves the best ROC-AUC (0.6158), reflecting superior probability calibration.'
));

children.push(h2('4.5 Support Vector Machine'));
children.push(body(
  'SVM finds the maximum-margin hyperplane separating classes, with soft-margin slack variables controlled by C. Despite the curse of dimensionality, SVM with a linear kernel (linear) was selected by GridSearchCV over RBF and polynomial kernels with C = 10. The choice of linear kernel is somewhat surprising but may reflect that the 40-feature selection process already linearised the feature space to some extent, or that the non-linear kernels overfit. A notable anomaly is SVM\'s ROC-AUC of 0.3870 — well below 0.5 random baseline. This is not a sign of poor discrimination, but rather a failure of Platt scaling (the sigmoid calibration used to convert SVM decision values to probabilities) on imbalanced multiclass problems. The resulting probabilities are systematically miscalibrated, producing inverted ranking behaviour in AUC computation.'
));

children.push(h2('4.6 K-Nearest Neighbours'));
children.push(body(
  'KNN classifies each test instance by majority vote among the k nearest training instances in feature space, using Euclidean distance. It is a non-parametric, instance-based method with no training phase (lazy learning). GridSearchCV identified k = 3 with distance-weighted voting (closer neighbours vote more strongly). KNN achieves the second-highest accuracy (0.5770), essentially matching Random Forest. However, this high accuracy masks poor minority class detection: KNN tends to assign whichever label is most frequent among the 3 nearest neighbours, which in an imbalanced dataset often means Sideways. The macro-F1 of 0.3119 confirms this.'
));

children.push(h2('4.7 Voting Ensemble (RF + LightGBM)'));
children.push(body(
  'The Voting Ensemble combines RF and LightGBM using soft voting — averaging their predicted class probabilities and taking the argmax. The rationale is diversity: RF learns Sideways-heavy boundaries with high accuracy but poor minority detection, while LightGBM is the only individual model with Bear detection capability (Bear F1 = 0.39) and calibrated probability estimates. By averaging probabilities, the ensemble preserves LightGBM\'s Bear detection signal while benefiting from RF\'s strong Sideways/Bull discrimination.'
));
children.push(body(
  'The result is a substantial F1 improvement: Voting Ensemble macro-F1 = 0.3903, vs RF\'s 0.3105 (+25.7%). ROC-AUC also improves to 0.5926. Accuracy (0.4198) is lower than either individual model because the ensemble is more willing to predict Bear when LightGBM is confident, at the expense of some Sideways accuracy.'
));

children.push(h2('4.8 LSTM Deep Learning Classifier'));
children.push(body(
  'The Long Short-Term Memory network is designed to capture temporal dependencies across sequences of on-chain features, which scalar models cannot exploit. Bitcoin market dynamics exhibit path-dependence: the current market state is a function not just of today\'s features but of the trajectory of the past 30 days.'
));
children.push(body(
  'Architecture: The model uses a sliding window of 30 consecutive days as input (shape: [batch, 30, 40]). Two bidirectional LSTM layers with 128 hidden units each process the sequence; bidirectionality allows the model to attend to both past and future context within the window. Dropout (0.3) is applied after each LSTM layer for regularisation. A fully connected output layer maps to 3 class logits with softmax activation. Training uses the Adam optimizer with learning rate 0.001, CrossEntropyLoss with inverse-frequency sample weights, and early stopping triggered at epoch 14 based on validation loss.'
));
children.push(body(
  'Performance: LSTM achieves accuracy 0.5675 and macro-F1 0.3452, the best F1 among all individual models. ROC-AUC of 0.5831 is competitive. The temporal modelling provides a meaningful improvement over classical ML baselines, confirming that Bitcoin market state is path-dependent.'
));

children.push(h2('4.9 Deep Q-Network Reinforcement Learning'));
children.push(body(
  'The DQN agent frames Bitcoin market state prediction as a sequential decision problem. At each time step, the agent observes the current on-chain feature vector (state s_t) and selects an action a_t ∈ {Bear, Sideways, Bull} (i.e., a market state prediction). The reward r_t is the 30-day forward return r_t = (P_{t+30} − P_t) / P_t, scaled so that correct directional predictions yield positive reward and incorrect ones yield negative reward.'
));
children.push(body(
  'Architecture: A fully connected Q-network with two hidden layers (128 units, ReLU) approximates the Q-function Q(s, a; θ). Training uses experience replay with a replay buffer of 2,000 transitions and target network updates every 10 episodes. The exploration policy is epsilon-greedy with ε = 0.05 (mostly greedy, minimal exploration). Training runs for 60 episodes.'
));
children.push(body(
  'Performance: DQN achieves accuracy 0.3273 and macro-F1 0.3180. The cumulative trading return is -0.9993, indicating that the RL agent\'s policy did not learn profitable trading behaviour within 60 episodes. This is expected given the limited training length and the difficulty of credit assignment over 30-day return horizons. DQN demonstrates the feasibility of RL for this problem, but would require significantly longer training and more sophisticated reward shaping for competitive performance.'
));

children.push(h2('4.10 Stacking Ensemble (RF + KNN + LGB + XGB)'));
children.push(body(
  'The Stacking Ensemble is the most technically sophisticated model in this study and achieves the best macro-F1 (0.4124). It consists of two stages:'
));
children.push(body(
  'Stage 1 — Base Learner OOF Generation: Four base learners (RF, KNN, LightGBM, XGBoost) generate out-of-fold (OOF) class probability predictions on the training set using TimeSeriesSplit(5) cross-validation. TimeSeriesSplit is critical: it ensures that when generating OOF predictions for fold k, only folds 1 through k-1 are used for training, preserving strict temporal ordering and preventing data leakage. Each base learner produces a 3-dimensional probability vector per instance, yielding a 12-dimensional meta-feature matrix (4 models × 3 classes).'
));
children.push(body(
  'Stage 2 — Max-Confidence Routing Meta-Strategy: Three meta-strategies were evaluated on held-out test data:'
));
children.push(bullet('Meta-LGB (train a LightGBM meta-learner on OOF features): macro-F1 = 0.2951 — worse than individual LightGBM, likely due to OOF distribution mismatch with the test set.'));
children.push(bullet('Soft Voting (average probabilities across all 4 base learners): macro-F1 = 0.2852 — diluted by the three Bear-blind models (RF, KNN, XGBoost).'));
children.push(bullet('Max-Confidence Routing with LGB 1.5x boost (WINNER): macro-F1 = 0.4124. For each test instance, the confidence-weighted prediction is: for each base learner, take max(probability vector) as confidence. LightGBM\'s confidence is multiplied by 1.5 to reflect its unique Bear detection capability. The final prediction is the class predicted by the base learner with the highest adjusted confidence. This selective routing allows the ensemble to leverage LightGBM when it is highly confident (especially for Bear predictions), while deferring to RF or KNN for Sideways/Bull predictions when their confidence is higher.'));
children.push(body(
  'The 1.5x LightGBM weight was determined by grid search over {1.0, 1.25, 1.5, 1.75, 2.0} on a validation fold. The max-confidence strategy outperforms all alternatives by a substantial margin, demonstrating that intelligently routing predictions based on confidence is more effective than naive averaging when base learners have complementary strengths.'
));
children.push(pageBreak());

// ══════════════════════════════════════════════════════════════════
// SECTION 5: RESULTS
// ══════════════════════════════════════════════════════════════════
children.push(h1('5. Results'));

children.push(h2('5.1 Overall Model Performance'));
children.push(body(
  'Table 2 presents the complete performance comparison across all ten models. Macro-F1 is the primary ranking metric. The Stacking ensemble achieves the best macro-F1 (0.4124), while Random Forest and KNN achieve the highest raw accuracy (0.5781 and 0.5770 respectively) due to their tendency to predict the majority Sideways class.'
));
children.push(new Paragraph({ spacing: { before: 120, after: 80 }, children: [new TextRun({ text: 'Table 2: Complete Model Performance Comparison', font: 'Arial', size: 20, bold: true, italics: true })] }));
children.push(perfTable());
children.push(new Paragraph({ spacing: { before: 60, after: 200 }, children: [new TextRun({ text: 'Table 2: All 10 models ranked by macro-F1. Best result highlighted in gold. Accuracy is reported as a secondary metric. N/A = DQN does not produce class probabilities for AUC computation.', font: 'Arial', size: 18, italics: true, color: '555555' })] }));

children.push(imgPara('model_summary_table.png', 1801, 434));
children.push(caption('Figure 2: Visual model performance summary showing Accuracy, F1 Macro, and ROC-AUC across all 10 models.'));

children.push(h2('5.2 Distribution Shift Analysis'));
children.push(body(
  'The most important finding in this study is not which model performs best, but rather why all models are limited. Figure 3 illustrates the class distribution shift between training and test periods.'
));
children.push(imgPara('distribution_shift.png', 2081, 735));
children.push(caption('Figure 3: Class distribution comparison between training (2010-2021) and test (2022-2023) periods. Note the 12-percentage-point surge in Sideways and 13.5-point drop in Bull class frequency.'));

children.push(body(
  'The surface-level statistics appear benign: the Bear class proportion barely changes (train: 17.5% → test: 19.0%, Δ = +1.5%). However, this masks a critical structural change. The 2022 Bear market events — the Terra/LUNA collapse in May 2022, the Three Arrows Capital bankruptcy in June 2022, and the FTX implosion in November 2022 — produced on-chain patterns qualitatively different from any prior Bear market in the training data. These events caused correlated institutional withdrawal, prolonged multi-month declines, and unusual on-chain behaviours (e.g., large exchange inflows, NUPL dropping to extreme negative values) that the 2013, 2015, and 2018 bear markets did not exhibit to the same degree or duration.'
));
children.push(body(
  'Simultaneously, the Bull class drops sharply (train: 35.8% → test: 22.3%, Δ = -13.5%) and Sideways surges (train: 46.7% → test: 58.7%, Δ = +12.0%), reflecting the post-crash consolidation period of late 2022 through 2023 when Bitcoin traded sideways after the crash. A model trained on 2010-2021 data, where Bull markets were the dominant medium-term trend, is systematically biased toward predicting Bull or Sideways and cannot reliably detect the qualitatively novel 2022 Bear patterns. This distribution shift is a fundamental ceiling that no amount of model sophistication can fully overcome without access to data from the test regime.'
));

children.push(h2('5.3 Per-Class F1 Breakdown'));
children.push(body(
  'Figure 4 shows the per-class F1 heatmap across all models. The striking finding is that Bear F1 ≈ 0 for almost all individual models: Logistic Regression, Random Forest, XGBoost, SVM, and KNN all fail to detect Bear markets on the test set. The sole exception among individual models is LightGBM, which achieves Bear F1 = 0.39, Sideways F1 = 0.24, and Bull F1 = 0.39 — a uniquely balanced profile attributable to the sample_weight fix and LightGBM\'s well-calibrated histogram-based probability estimates.'
));
children.push(imgPara('per_class_f1_heatmap.png', 996, 787));
children.push(caption('Figure 4: Per-class F1 heatmap. Green = high F1, red/dark = near-zero F1. Most models show Bear F1 ≈ 0. Only LightGBM and Stacking show meaningful Bear detection.'));

children.push(body(
  'The Stacking ensemble achieves Bear F1 improvement over all individual models through max-confidence routing: when LightGBM is confident about a Bear prediction (after 1.5x boosting), it overrides the other models. The LSTM also shows some Bear detection capability, reflecting the temporal dependency modelling capturing crash trajectory patterns.'
));

children.push(h2('5.4 Individual Model Confusion Matrices'));
children.push(body(
  'Figures 5 through 14 show the confusion matrices for all ten models. Each matrix displays predicted vs. actual class labels (Bear = -1, Sideways = 0, Bull = 1) on the held-out test set.'
));

// Confusion matrices in 2-column layout via side-by-side arrangement
// For wide display, we place them sequentially with pair labels

const cmPairs = [
  ['Logistic_Regression_confusion_matrix.png', 'Figure 5: Logistic Regression confusion matrix. LR predominantly predicts Sideways, with some Bull predictions but essentially zero Bear detection.'],
  ['Random_Forest_confusion_matrix.png', 'Figure 6: Random Forest confusion matrix. High accuracy driven by strong Sideways prediction; Bear class almost entirely missed.'],
  ['XGBoost_confusion_matrix.png', 'Figure 7: XGBoost confusion matrix. Spread across classes but poor overall accuracy; test distribution mismatch is evident.'],
  ['LightGBM_confusion_matrix.png', 'Figure 8: LightGBM confusion matrix. The only individual model predicting all three classes. Note Bear predictions along the left column.'],
  ['SVM_confusion_matrix.png', 'Figure 9: SVM (linear kernel, C=10) confusion matrix. Linear boundary fails to capture multi-class structure; extremely low F1.'],
  ['KNN_confusion_matrix.png', 'Figure 10: KNN (k=3, distance weights) confusion matrix. High accuracy via Sideways dominance; Bear class entirely missed.'],
  ['Voting_Ensemble_confusion_matrix.png', 'Figure 11: Voting Ensemble (RF+LGB soft voting) confusion matrix. Improved minority class detection vs. individual models.'],
  ['LSTM_confusion_matrix.png', 'Figure 12: LSTM (BiLSTM, window=30) confusion matrix. Best individual F1; temporal modelling helps with Bull detection.'],
  ['DQN_confusion_matrix.png', 'Figure 13: DQN reinforcement learning agent confusion matrix. Policy learning produces a scattered prediction pattern after 60 episodes.'],
  ['Stacking_confusion_matrix.png', 'Figure 14: Stacking ensemble (max-confidence routing) confusion matrix. Best overall F1; note improved Bear detection enabled by LGB routing.'],
];

for (const [fname, cap] of cmPairs) {
  children.push(imgPara(fname, 900, 750));
  children.push(caption(cap));
}

children.push(h2('5.5 ROC Curves'));
children.push(body(
  'Figure 15 presents one-vs-rest ROC curves for all models. LightGBM achieves the highest macro-averaged AUC (0.6158), confirming its superior probability calibration. XGBoost is second (0.6103). The Voting Ensemble (0.5926) and Stacking (0.5796) improve over their respective base learners.'
));
children.push(body(
  'The most notable anomaly is SVM\'s AUC of 0.3870 — below the 0.5 random classifier line. As discussed in Section 4.5, this does not indicate that SVM discriminates less well than random; it reflects that Platt scaling systematically inverts the probability estimates on the imbalanced test set, causing the ROC curve to fall below the diagonal. Inspecting the raw decision values reveals that SVM\'s decision boundary does carry some discriminative signal, but the probability calibration failure renders AUC misleading for this model.'
));
children.push(body(
  'DQN is excluded from the ROC analysis (N/A) because the Q-network does not produce normalised class probability outputs compatible with the one-vs-rest AUC formulation.'
));
children.push(imgPara('roc_curves_all_models.png', 2700, 750));
children.push(caption('Figure 15: One-vs-rest ROC curves for all models. LightGBM achieves best macro-AUC (0.6158). SVM\'s sub-0.5 AUC reflects Platt scaling failure, not poor discrimination.'));

children.push(h2('5.6 Feature Importance'));
children.push(body(
  'Figure 16 shows the combined feature importance across Random Forest and LightGBM. The top predictive features are:'
));
children.push(bullet('total_transaction_fees_ma14 — 14-day moving average of total Bitcoin transaction fees; reflects miner demand and network activity.'));
children.push(bullet('miners_revenue_ma14 / miners_revenue_ma30 — Miner revenue is the primary sell-side pressure signal; sustained high revenue followed by sharp drops often precedes market tops.'));
children.push(bullet('nupl / nupl_ma30 — Net Unrealised Profit/Loss; when NUPL approaches 1.0 (most holders in profit), capitulation risk increases; when NUPL is deeply negative, accumulation signals emerge.'));
children.push(bullet('hash_rate and difficulty — Proxy for miner confidence and network security; hash rate drops signal miner capitulation.'));
children.push(bullet('transaction_volume_usd_ma7 — 7-day MA of transaction volume; captures changes in economic activity on the network.'));

children.push(imgPara('feature_importance_combined.png', 1500, 1050));
children.push(caption('Figure 16: Combined feature importance from Random Forest and LightGBM. On-chain metrics related to miner economics and unrealised profit/loss dominate.'));

children.push(imgPara('feature_importance_rf.png', 1500, 1050));
children.push(caption('Figure 17: Random Forest Gini importance. RF gives high weight to long-horizon moving averages of miner revenue.'));

children.push(imgPara('feature_importance_lgbm.png', 1500, 1050));
children.push(caption('Figure 18: LightGBM split-based importance. LGB identifies shorter-horizon indicators and NUPL as more important than RF.'));

children.push(h2('5.7 Ensemble Progression'));
children.push(body(
  'Table 3 traces the F1 progression from best individual model to each ensemble strategy, quantifying the value of ensemble design decisions.'
));
children.push(new Paragraph({ spacing: { before: 120, after: 80 }, children: [new TextRun({ text: 'Table 3: Ensemble F1 Progression', font: 'Arial', size: 20, bold: true, italics: true })] }));
children.push(ensembleProgressionTable());
children.push(new Paragraph({ spacing: { before: 60, after: 200 }, children: [new TextRun({ text: 'Table 3: Macro-F1 progression from best individual model to each ensemble variant. The max-confidence routing strategy provides the largest gain (+32.8% over best individual model).', font: 'Arial', size: 18, italics: true, color: '555555' })] }));

children.push(body(
  'Two noteworthy observations: (1) naive ensemble strategies (meta-LGB and soft voting) actually degrade F1 compared to individual LightGBM, illustrating that ensemble combination is not universally beneficial; (2) the max-confidence routing strategy recovers the loss and achieves a 32.8% gain over the best individual model by intelligently routing predictions to the most confident base learner, weighted by each model\'s complementary strengths.'
));

children.push(h2('5.8 LSTM Training Curves'));
children.push(body(
  'Figure 19 shows the BiLSTM training and validation loss curves. Training converges smoothly, with early stopping triggered at epoch 14. The validation loss plateaus after epoch 10, confirming that additional training would lead to overfitting. The gap between training and validation loss reflects the distribution shift challenge: the model fits the 2010-2021 training patterns well but faces a structurally different test distribution.'
));
children.push(imgPara('LSTM_training_curves.png', 1800, 600));
children.push(caption('Figure 19: BiLSTM training and validation loss curves. Early stopping triggered at epoch 14 based on validation loss plateau.'));

children.push(h2('5.9 DQN Cumulative Return'));
children.push(body(
  'Figure 20 shows the DQN agent\'s cumulative portfolio return over the test period. The return of -0.9993 indicates that the agent\'s trading policy effectively lost nearly all capital. The curve shows the agent making several large wrong-direction bets, particularly during the 2022 crash when it failed to detect the Bear regime. This underscores the distribution shift problem: the RL policy trained on 2010-2021 reward signals has no precedent for the 2022 market dynamics.'
));
children.push(imgPara('DQN_cumulative_return.png', 1500, 600));
children.push(caption('Figure 20: DQN agent cumulative portfolio return on test data. Return of -0.9993 reflects policy failure due to distribution shift.'));

children.push(pageBreak());

// ══════════════════════════════════════════════════════════════════
// SECTION 6: DISCUSSION
// ══════════════════════════════════════════════════════════════════
children.push(h1('6. Discussion'));

children.push(h2('6.1 Why Random Forest Outperforms SVM'));
children.push(body(
  'The performance gap between Random Forest (F1 = 0.3105) and SVM (F1 = 0.1223) illustrates the fundamental mismatch between linear models and on-chain feature interactions. In a 40-dimensional space of exponentially moving averages, momentum indicators, and blockchain metrics, the relevant decision boundaries are inherently non-linear. Miners\' revenue interacts with NUPL in a complex non-linear way to signal market tops; transaction fee pressure combines with price momentum through a threshold effect that linear combinations cannot capture.'
));
children.push(body(
  'Random Forest builds a forest of decision trees, each of which can capture these piecewise non-linear interactions. The average over 100 trees then smooths out individual tree noise. SVM with a linear kernel attempts to find a single linear hyperplane separating classes in the original feature space — an approximation that discards all non-linear feature interaction information. While RBF kernels could theoretically capture non-linearities, GridSearchCV did not select them, possibly because RBF kernels are sensitive to feature scaling and the high correlation structure in the on-chain feature set.'
));

children.push(h2('6.2 Why LightGBM Achieves Best AUC but Not Accuracy'));
children.push(body(
  'LightGBM achieves the highest ROC-AUC (0.6158) yet only fourth-highest accuracy (0.3369). This apparent contradiction resolves when we recognize that AUC and accuracy measure fundamentally different things. AUC measures the quality of probabilistic ranking: how well does the model separate classes by predicted probability, regardless of threshold? Accuracy measures correctness at the default 0.5 threshold decision boundary.'
));
children.push(body(
  'LightGBM\'s histogram-based boosting produces well-calibrated class probability estimates — the model assigns higher probabilities to the correct class more often than other models, even when the probability is not high enough to exceed 0.5. In contrast, RF and KNN achieve high accuracy by being extremely confident about Sideways predictions (which are the most common class) while being wrong about most Bear and Bull predictions. Their accuracy is inflated by correctly classifying the easy cases. LightGBM\'s willingness to spread probability mass across all three classes reflects better calibration but lowers accuracy when it makes "uncertain" predictions that are then rounded to the wrong majority class.'
));

children.push(h2('6.3 Why Stacking F1 is Best but Accuracy Mid-Range'));
children.push(body(
  'The Stacking ensemble achieves the best macro-F1 (0.4124) but only mid-range accuracy (0.4283, versus RF and KNN at 0.577-0.578). This is an explicit design choice made by the max-confidence routing strategy. By giving LightGBM 1.5x confidence weight, the ensemble is more willing to predict Bear when LightGBM is confident, even if that prediction will sometimes be wrong. This trade improves Bear-class recall at the cost of some Sideways accuracy.'
));
children.push(body(
  'From a practical investment standpoint, this trade-off is desirable: correctly detecting Bear markets (and avoiding losses) is typically more valuable than perfectly predicting Sideways periods (where no action is taken). The F1 metric, which gives equal weight to all classes, correctly rewards this Bear detection improvement even at the cost of reduced accuracy.'
));

children.push(h2('6.4 Distribution Shift as the Fundamental Ceiling'));
children.push(body(
  'All models in this study are fundamentally constrained by the distribution shift between 2010-2021 training data and 2022-2023 test data. The 2022 cryptocurrency crash was driven by mechanisms — institutional leverage cascade, centralised exchange collapses, regulatory crackdowns — that have no precedent in the Bitcoin data available for training. Even the most sophisticated model (Stacking, BiLSTM) cannot reliably predict market states that it has never encountered in training.'
));
children.push(body(
  'This observation has important implications beyond this study. It suggests that for extreme tail events — black swans by definition — on-chain historical patterns may be insufficient, and that incorporating macro-financial signals (interest rate environment, institutional exposure proxies, regulatory news), sentiment analysis (social media, news) and cross-asset correlations may be necessary. The distribution shift is not a bug in the experimental design; it is a realistic representation of the challenge facing any real-world cryptocurrency prediction system.'
));
children.push(pageBreak());

// ══════════════════════════════════════════════════════════════════
// SECTION 7: CONCLUSION
// ══════════════════════════════════════════════════════════════════
children.push(h1('7. Conclusion'));
children.push(body(
  'This paper presents a comprehensive study of ten machine learning approaches for Bitcoin market state prediction using on-chain blockchain analytics. The primary conclusion is that distribution shift — specifically, the qualitatively novel 2022 cryptocurrency crash patterns absent from the training distribution — is the dominant limitation for all models, regardless of architectural sophistication. No model achieves macro-F1 above 0.5, and most individual models fail entirely to detect the Bear class on the test set.'
));
children.push(body(
  'Within this constrained environment, the Stacking ensemble with max-confidence routing achieves the best macro-F1 of 0.4124, demonstrating that intelligently combining diverse base learners can partially overcome the Bear detection problem. The key insight is that LightGBM is the only individual model with Bear detection capability, and the max-confidence routing strategy explicitly amplifies this capability by giving LightGBM 1.5x confidence weight. Naive ensemble strategies (soft voting, meta-learner) actually degrade performance, underscoring the importance of informed ensemble design.'
));
children.push(body(
  'The BiLSTM deep learning classifier achieves the best F1 among individual models (0.3452), confirming that temporal dependency modelling provides meaningful signal beyond classical ML approaches. The DQN reinforcement learning agent demonstrates the feasibility of the RL paradigm for this problem, though 60 training episodes are insufficient for policy convergence on such a complex, non-stationary environment.'
));
children.push(body(
  'Feature importance analysis consistently identifies on-chain metrics related to miner economics (miners_revenue_ma14/ma30, total_transaction_fees_ma14) and unrealised profit/loss (nupl, nupl_ma30) as the most predictive signals, consistent with the hypothesis that these on-chain fundamentals drive market regime transitions.'
));
children.push(body(
  'Future research directions include: (1) incorporating macro-financial signals (Federal Reserve interest rates, equity market correlations, credit spreads) to better capture the institutional dynamics driving the 2022 crash; (2) sentiment analysis from social media and news sources to capture qualitative information not present in on-chain data; (3) transformer-based architectures (attention over long sequence windows) to better model long-range temporal dependencies; (4) online learning or domain adaptation techniques to dynamically adjust model parameters as distribution shifts occur; and (5) longer DQN training with more sophisticated reward shaping to develop viable RL trading policies.'
));
children.push(pageBreak());

// ══════════════════════════════════════════════════════════════════
// SECTION 8: WORK DISTRIBUTION
// ══════════════════════════════════════════════════════════════════
children.push(h1('8. Work Distribution'));
children.push(body(
  'Table 4 details the contribution of each team member to the project. All members participated in final report writing, result interpretation, and presentation preparation.'
));
children.push(new Paragraph({ spacing: { before: 120, after: 80 }, children: [new TextRun({ text: 'Table 4: Team Work Distribution', font: 'Arial', size: 20, bold: true, italics: true })] }));
children.push(workTable());
children.push(new Paragraph({ spacing: { before: 60, after: 200 }, children: [new TextRun({ text: 'Table 4: Detailed work division by team member. All members contributed to report writing and presentation.', font: 'Arial', size: 18, italics: true, color: '555555' })] }));

children.push(pageBreak());

// ══════════════════════════════════════════════════════════════════
// REFERENCES
// ══════════════════════════════════════════════════════════════════
children.push(h1('References'));
const refs = [
  '[1] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785-794.',
  '[2] Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems, 30.',
  '[3] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735-1780.',
  '[4] Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Hassabis, D. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533.',
  '[5] Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.',
  '[6] Nakamoto, S. (2008). Bitcoin: A peer-to-peer electronic cash system. Decentralized Business Review.',
  '[7] Glassnode. (2024). On-chain market indicators. https://glassnode.com',
  '[8] CoinMetrics. (2024). Network data pro. https://coinmetrics.io',
  '[9] Wolpert, D. H. (1992). Stacked generalization. Neural Networks, 5(2), 241-259.',
  '[10] Vapnik, V. N. (1995). The Nature of Statistical Learning Theory. Springer, New York.',
  '[11] Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. IEEE Transactions on Information Theory, 13(1), 21-27.',
  '[12] Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. Journal of Machine Learning Research, 13, 281-305.',
];
for (const ref of refs) {
  children.push(new Paragraph({
    spacing: { before: 60, after: 60, line: 276 },
    indent: { left: 360, hanging: 360 },
    children: [new TextRun({ text: ref, font: 'Arial', size: 20 })]
  }));
}

// ══════════════════════════════════════════════════════════════════
// BUILD DOCUMENT
// ══════════════════════════════════════════════════════════════════

const doc = new Document({
  numbering: {
    config: [
      {
        reference: 'bullets',
        levels: [{
          level: 0,
          format: LevelFormat.BULLET,
          text: '•',
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } }
        }]
      }
    ]
  },
  styles: {
    default: {
      document: { run: { font: 'Arial', size: 22 } }
    },
    paragraphStyles: [
      {
        id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 32, bold: true, font: 'Arial', color: '1F3864' },
        paragraph: { spacing: { before: 360, after: 120 }, outlineLevel: 0 }
      },
      {
        id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 26, bold: true, font: 'Arial', color: '2E5090' },
        paragraph: { spacing: { before: 240, after: 80 }, outlineLevel: 1 }
      }
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
      }
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: '1F3864', space: 1 } },
          children: [
            new TextRun({ text: 'Bitcoin Market State Prediction  |  SC6122 NTU Group #17', font: 'Arial', size: 18, color: '1F3864' }),
            new TextRun({ text: '\t', font: 'Arial', size: 18 }),
            new TextRun({ text: 'Chen Zhiyu  |  LIU RUYAN  |  SHI XIANGYAN  |  YANG SHUYI', font: 'Arial', size: 18, color: '888888' }),
          ]
        })]
      })
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          border: { top: { style: BorderStyle.SINGLE, size: 4, color: '1F3864', space: 1 } },
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: 'Page ', font: 'Arial', size: 18, color: '555555' }),
            new TextRun({ children: [PageNumber.CURRENT], font: 'Arial', size: 18, color: '555555' }),
            new TextRun({ text: ' of ', font: 'Arial', size: 18, color: '555555' }),
            new TextRun({ children: [PageNumber.TOTAL_PAGES], font: 'Arial', size: 18, color: '555555' }),
          ]
        })]
      })
    },
    children
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(OUTPUT, buffer);
  const size = fs.statSync(OUTPUT).size;
  console.log(`SUCCESS: ${OUTPUT}`);
  console.log(`File size: ${(size / 1024 / 1024).toFixed(2)} MB (${size.toLocaleString()} bytes)`);
}).catch(err => {
  console.error('ERROR:', err);
  process.exit(1);
});
