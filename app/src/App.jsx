import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BadgeDollarSign,
  CheckCircle2,
  ChevronDown,
  CreditCard,
  FolderOpen,
  Globe2,
  Landmark,
  Loader2,
  Lock,
  Play,
  RefreshCw,
  Server,
  ShieldCheck,
  SlidersHorizontal,
  TrendingUp
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

const API_BASE =
  import.meta.env.VITE_API_BASE ||
  (import.meta.env.DEV ? "/api" : "http://127.0.0.1:8000");
const REQUEST_TIMEOUT_MS = 8000;

const defaultForm = {
  amount: 125,
  currency: "USD",
  product: "W",
  card_type: "debit",
  card_brand: "visa",
  email_domain_match: true,
  payer_free_email: true,
  receiver_free_email: true,
  hour_of_day_local: 10,
  day_of_week: 2,
  use_live_fx: true,
  advanced: {
    tx_count_1h: 1,
    tx_count_24h: 4,
    geo_velocity_kmh: 5,
    distance_km: 2,
    days_since_last_tx: 3,
    account_age_days: 365,
    history_count: 2,
    history_fraud_rate: 0.03,
    prior_fraud_count: 0,
    chargeback_count: 0,
    merchant_frequency: 25,
    identity_missing_rate: 0.2,
    device_present: true,
    mobile_device: true,
    card_device_mismatch: false,
    suspicious_identity_signal: 0
  }
};

const transactionPresets = {
  reliable: {
    label: "Reliable",
    detail: "Small daytime purchase with matched identity signals",
    reasons: ["Low amount", "Matched email", "Normal velocity"],
    form: defaultForm
  },
  suspicious: {
    label: "Suspicious",
    detail: "High-value late transaction with velocity and identity risk",
    reasons: ["High value", "Email mismatch", "Prior fraud"],
    form: {
      amount: 9800,
      currency: "EUR",
      product: "C",
      card_type: "credit",
      card_brand: "mastercard",
      email_domain_match: false,
      payer_free_email: true,
      receiver_free_email: true,
      hour_of_day_local: 2,
      day_of_week: 6,
      use_live_fx: true,
      advanced: {
        tx_count_1h: 18,
        tx_count_24h: 96,
        geo_velocity_kmh: 1420,
        distance_km: 7300,
        days_since_last_tx: 0,
        account_age_days: 3,
        history_count: 1,
        history_fraud_rate: 0.72,
        prior_fraud_count: 3,
        chargeback_count: 4,
        merchant_frequency: 2,
        identity_missing_rate: 0.9,
        device_present: true,
        mobile_device: true,
        card_device_mismatch: true,
        suspicious_identity_signal: 1
      }
    }
  }
};

const products = [
  ["W", "Retail"],
  ["H", "Home"],
  ["C", "Checkout"],
  ["S", "Service"],
  ["R", "Recurring"]
];

const dayLabels = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

const numericRules = {
  amount: { min: 0, max: 1_000_000_000, fallback: defaultForm.amount, integer: false },
  hour_of_day_local: { min: 0, max: 23, fallback: defaultForm.hour_of_day_local, integer: true },
  day_of_week: { min: 0, max: 6, fallback: defaultForm.day_of_week, integer: true },
  tx_count_1h: { min: 0, max: 500, fallback: defaultForm.advanced.tx_count_1h, integer: false },
  tx_count_24h: { min: 0, max: 5000, fallback: defaultForm.advanced.tx_count_24h, integer: false },
  geo_velocity_kmh: { min: 0, max: 2000, fallback: defaultForm.advanced.geo_velocity_kmh, integer: false },
  distance_km: { min: 0, max: 20000, fallback: defaultForm.advanced.distance_km, integer: false },
  days_since_last_tx: { min: 0, max: 365, fallback: defaultForm.advanced.days_since_last_tx, integer: false },
  account_age_days: { min: 0, max: 10000, fallback: defaultForm.advanced.account_age_days, integer: false },
  history_count: { min: 0, max: 1_000_000, fallback: defaultForm.advanced.history_count, integer: false },
  history_fraud_rate: { min: 0, max: 1, fallback: defaultForm.advanced.history_fraud_rate, integer: false },
  prior_fraud_count: { min: 0, max: 10000, fallback: defaultForm.advanced.prior_fraud_count, integer: false },
  chargeback_count: { min: 0, max: 100, fallback: defaultForm.advanced.chargeback_count, integer: false },
  merchant_frequency: { min: 0, max: 10_000_000, fallback: defaultForm.advanced.merchant_frequency, integer: false },
  identity_missing_rate: { min: 0, max: 1, fallback: defaultForm.advanced.identity_missing_rate, integer: false },
  suspicious_identity_signal: { min: 0, max: 1, fallback: defaultForm.advanced.suspicious_identity_signal, integer: false }
};

function App() {
  const [models, setModels] = useState([]);
  const [selected, setSelected] = useState("not_loaded");
  const [health, setHealth] = useState(null);
  const [form, setForm] = useState(defaultForm);
  const [result, setResult] = useState(null);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [loadingModels, setLoadingModels] = useState(false);
  const [loadingPrediction, setLoadingPrediction] = useState(false);
  const [message, setMessage] = useState(null);
  const [apiState, setApiState] = useState("checking");
  const [runInfo, setRunInfo] = useState(null);

  const selectedModel = useMemo(
    () => models.find((model) => stripExt(model.checkpoint) === selected) || models.find((model) => model.selected),
    [models, selected]
  );
  const recommended = useMemo(() => models.find((model) => model.recommended), [models]);

  useEffect(() => {
    refreshModels();
  }, []);

  async function api(path, options = {}) {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    try {
      const response = await fetch(`${API_BASE}${path}`, {
        headers: { "Content-Type": "application/json", ...(options.headers || {}) },
        signal: controller.signal,
        ...options
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.detail || `${response.status} ${response.statusText}`);
      }
      return data;
    } catch (error) {
      throw new Error(apiErrorMessage(error));
    } finally {
      window.clearTimeout(timeout);
    }
  }

  async function refreshModels() {
    setLoadingModels(true);
    setMessage(null);
    try {
      const [healthData, modelData] = await Promise.all([
        api("/health"),
        api("/models")
      ]);
      setHealth(healthData);
      setModels(modelData.models || []);
      setRunInfo(modelData.run || healthData.run || null);
      setSelected(modelData.selected || healthData.model_version || "not_loaded");
      setApiState(healthData.model_version === "not_loaded" ? "no_model" : "online");
    } catch (error) {
      setHealth(null);
      setModels([]);
      setRunInfo(null);
      setSelected("not_loaded");
      setApiState("offline");
      setMessage({ type: "error", text: error.message });
    } finally {
      setLoadingModels(false);
    }
  }

  async function useModel(checkpoint) {
    setLoadingModels(true);
    setMessage(null);
    try {
      const data = await api("/models/select", {
        method: "POST",
        body: JSON.stringify({ checkpoint })
      });
      setSelected(data.model_version);
      await refreshModels();
      setMessage({ type: "ok", text: `${checkpoint} loaded` });
      setApiState("online");
    } catch (error) {
      setMessage({ type: "error", text: error.message });
    } finally {
      setLoadingModels(false);
    }
  }

  async function submitPrediction(event) {
    event.preventDefault();
    setLoadingPrediction(true);
    setMessage(null);
    try {
      const normalized = normalizeForm(form);
      setForm(normalized.form);
      const payload = numericPayload(normalized.form);
      const prediction = await api("/predict-demo", {
        method: "POST",
        body: JSON.stringify(payload)
      });
      setResult(prediction);
      setApiState("online");
      if (normalized.messages.length) {
        setMessage({ type: "ok", text: `Adjusted ${normalized.messages.join(", ")}` });
      }
    } catch (error) {
      setMessage({ type: "error", text: error.message });
    } finally {
      setLoadingPrediction(false);
    }
  }

  function updateField(key, value) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  function commitField(key) {
    const rule = numericRules[key];
    if (!rule) return;
    setForm((current) => {
      const sanitized = sanitizeNumber(current[key], rule);
      if (sanitized.changed) {
        setMessage({ type: "ok", text: `Adjusted ${humanLabel(key)}` });
      }
      return { ...current, [key]: sanitized.value };
    });
  }

  function updateAdvanced(key, value) {
    setForm((current) => ({
      ...current,
      advanced: { ...current.advanced, [key]: value }
    }));
  }

  function commitAdvanced(key) {
    const rule = numericRules[key];
    if (!rule) return;
    setForm((current) => {
      const sanitized = sanitizeNumber(current.advanced[key], rule);
      if (sanitized.changed) {
        setMessage({ type: "ok", text: `Adjusted ${humanLabel(key)}` });
      }
      return {
        ...current,
        advanced: {
          ...current.advanced,
          [key]: sanitized.value
        }
      };
    });
  }

  function applyPreset(name) {
    const preset = transactionPresets[name];
    if (!preset) return;
    setForm(cloneForm(preset.form));
    setResult(null);
    setMessage({ type: "ok", text: `${preset.label} transaction loaded` });
    setAdvancedOpen(name === "suspicious");
  }

  const metricBars = chartMetrics(selectedModel);
  const riskClass = result ? `risk-${result.risk_band}` : "risk-idle";
  const canScore = apiState === "online" && health?.model_version !== "not_loaded";
  const status = statusFor(apiState, loadingModels, health);

  return (
    <main className="shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">
            <Landmark size={22} />
          </div>
          <div>
            <p className="eyebrow">Fraud Console</p>
            <h1>Transaction screening</h1>
          </div>
        </div>
        <div className="top-actions">
          <StatusPill
            icon={status.icon}
            tone={status.tone}
            label={status.label}
          />
          <IconButton
            label="Refresh models"
            icon={RefreshCw}
            loading={loadingModels}
            onClick={refreshModels}
          />
        </div>
      </header>

      {message && (
        <div className={`notice ${message.type}`}>
          {message.type === "error" ? <AlertTriangle size={17} /> : <CheckCircle2 size={17} />}
          <span>{message.text}</span>
        </div>
      )}

      <section className="summary-band">
        <MetricTile label="Selected" value={selected || "not_loaded"} icon={Server} />
        <MetricTile label="Run" value={runInfo?.run || "default"} icon={FolderOpen} />
        <MetricTile label="Threshold" value={formatNumber(health?.threshold ?? selectedModel?.threshold, 4)} icon={Activity} />
        <MetricTile label="Recommended" value={recommended?.checkpoint || "None"} icon={ShieldCheck} />
      </section>

      <div className="workspace">
        <section className="panel model-panel">
          <PanelHeading
            icon={ShieldCheck}
            title="Model selection"
            action={
              recommended && (
                <button className="small-command" onClick={() => useModel(recommended.checkpoint)}>
                  <ShieldCheck size={15} />
                  Recommended
                </button>
              )
            }
          />
          <div className="metric-chart" aria-label="Selected model metrics">
            {metricBars.length ? (
              <ResponsiveContainer width="100%" height={170}>
                <BarChart data={metricBars} margin={{ left: -18, right: 8, top: 8, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="name" tickLine={false} axisLine={false} />
                  <YAxis domain={[0, 1]} tickLine={false} axisLine={false} />
                  <Tooltip formatter={(value) => formatNumber(value, 4)} />
                  <Bar dataKey="value" radius={[5, 5, 0, 0]} fill="#0f8b63" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <EmptyState
                title={apiState === "offline" ? "API unavailable" : "No metrics loaded"}
                detail={apiState === "offline" ? "Start the backend, then refresh." : "A compatible checkpoint has not been loaded."}
              />
            )}
          </div>

          <div className="model-list">
            {models.length === 0 && (
              <div className="model-list-empty">
                {apiState === "offline" ? "No registry connection." : "No compatible checkpoints found."}
              </div>
            )}
            {models.slice(0, 10).map((model) => (
              <button
                key={model.checkpoint}
                className={`model-row ${model.selected ? "selected" : ""}`}
                onClick={() => useModel(model.checkpoint)}
              >
                <span>
                  <strong>{model.checkpoint}</strong>
                  <small>{model.reason}</small>
                </span>
                <span className="model-metrics">
                  <MetricInline label="AUPRC" value={model.metrics?.auprc} />
                  <MetricInline label="AUROC" value={model.metrics?.auroc} />
                  <MetricInline label="F1" value={model.metrics?.f1} />
                </span>
              </button>
            ))}
          </div>
        </section>

        <form className="panel transaction-panel" onSubmit={submitPrediction}>
          <PanelHeading
            icon={CreditCard}
            title="Transaction"
            action={
              <button type="submit" className="primary-command" disabled={loadingPrediction || !canScore}>
                {loadingPrediction && <Loader2 size={17} className="spin" />}
                Score
              </button>
            }
          />

          <div className="preset-strip" aria-label="Transaction presets">
            {Object.entries(transactionPresets).map(([name, preset]) => (
              <button
                type="button"
                key={name}
                className={`preset-card ${name}`}
                onClick={() => applyPreset(name)}
              >
                <span>
                  {name === "reliable" ? <CheckCircle2 size={17} /> : <AlertTriangle size={17} />}
                  <strong>{preset.label}</strong>
                </span>
                <small>{preset.detail}</small>
                <em>{preset.reasons.join(" | ")}</em>
              </button>
            ))}
          </div>

          <div className="form-grid">
            <Field label="Amount">
              <input
                type="number"
                min={numericRules.amount.min}
                max={numericRules.amount.max}
                step="0.01"
                value={form.amount}
                onChange={(event) => updateField("amount", event.target.value)}
                onBlur={() => commitField("amount")}
              />
            </Field>
            <Field label="Currency" icon={Globe2}>
              <select value={form.currency} onChange={(event) => updateField("currency", event.target.value)}>
                {["USD", "EUR", "GBP", "SGD", "JPY", "AUD"].map((currency) => (
                  <option key={currency} value={currency}>{currency}</option>
                ))}
              </select>
            </Field>
            <Field label="Hour">
              <input
                type="number"
                min={numericRules.hour_of_day_local.min}
                max={numericRules.hour_of_day_local.max}
                value={form.hour_of_day_local}
                onChange={(event) => updateField("hour_of_day_local", event.target.value)}
                onBlur={() => commitField("hour_of_day_local")}
              />
            </Field>
            <Field label="Day">
              <select value={form.day_of_week} onChange={(event) => updateField("day_of_week", event.target.value)}>
                {dayLabels.map((label, index) => (
                  <option key={label} value={index}>{label}</option>
                ))}
              </select>
            </Field>
          </div>

          <Segmented
            label="Product"
            value={form.product}
            options={products}
            onChange={(value) => updateField("product", value)}
          />
          <Segmented
            label="Card type"
            value={form.card_type}
            options={[
              ["debit", "Debit"],
              ["credit", "Credit"],
              ["charge card", "Charge"],
              ["debit or credit", "Mixed"]
            ]}
            onChange={(value) => updateField("card_type", value)}
          />
          <div className="form-grid compact-grid">
            <Field label="Card brand">
              <select value={form.card_brand} onChange={(event) => updateField("card_brand", event.target.value)}>
                {["visa", "mastercard", "american express", "discover", "other"].map((brand) => (
                  <option key={brand} value={brand}>{titleCase(brand)}</option>
                ))}
              </select>
            </Field>
          </div>

          <div className="toggle-grid">
            <Toggle label="Email match" checked={form.email_domain_match} onChange={(value) => updateField("email_domain_match", value)} />
            <Toggle label="Payer free email" checked={form.payer_free_email} onChange={(value) => updateField("payer_free_email", value)} />
            <Toggle label="Receiver free email" checked={form.receiver_free_email} onChange={(value) => updateField("receiver_free_email", value)} />
            <Toggle label="Live exchange rate" checked={form.use_live_fx} onChange={(value) => updateField("use_live_fx", value)} />
          </div>

          <button type="button" className="advanced-toggle" onClick={() => setAdvancedOpen((open) => !open)}>
            <SlidersHorizontal size={17} />
            Advanced signals
            <ChevronDown size={17} className={advancedOpen ? "open" : ""} />
          </button>

          <div className={`advanced-region ${advancedOpen ? "open" : ""}`}>
            <div className="advanced-grid">
              <NumberField label="Transactions in 1 hour" value={form.advanced.tx_count_1h} rule={numericRules.tx_count_1h} onChange={(value) => updateAdvanced("tx_count_1h", value)} onBlur={() => commitAdvanced("tx_count_1h")} />
              <NumberField label="Transactions in 24 hours" value={form.advanced.tx_count_24h} rule={numericRules.tx_count_24h} onChange={(value) => updateAdvanced("tx_count_24h", value)} onBlur={() => commitAdvanced("tx_count_24h")} />
              <NumberField label="Geographic velocity" value={form.advanced.geo_velocity_kmh} rule={numericRules.geo_velocity_kmh} onChange={(value) => updateAdvanced("geo_velocity_kmh", value)} onBlur={() => commitAdvanced("geo_velocity_kmh")} />
              <NumberField label="Distance km" value={form.advanced.distance_km} rule={numericRules.distance_km} onChange={(value) => updateAdvanced("distance_km", value)} onBlur={() => commitAdvanced("distance_km")} />
              <NumberField label="Days since last transaction" value={form.advanced.days_since_last_tx} rule={numericRules.days_since_last_tx} onChange={(value) => updateAdvanced("days_since_last_tx", value)} onBlur={() => commitAdvanced("days_since_last_tx")} />
              <NumberField label="Account age" value={form.advanced.account_age_days} rule={numericRules.account_age_days} onChange={(value) => updateAdvanced("account_age_days", value)} onBlur={() => commitAdvanced("account_age_days")} />
              <NumberField label="History count" value={form.advanced.history_count} rule={numericRules.history_count} onChange={(value) => updateAdvanced("history_count", value)} onBlur={() => commitAdvanced("history_count")} />
              <NumberField label="History fraud rate" value={form.advanced.history_fraud_rate} rule={numericRules.history_fraud_rate} step="0.01" onChange={(value) => updateAdvanced("history_fraud_rate", value)} onBlur={() => commitAdvanced("history_fraud_rate")} />
              <NumberField label="Prior fraud count" value={form.advanced.prior_fraud_count} rule={numericRules.prior_fraud_count} onChange={(value) => updateAdvanced("prior_fraud_count", value)} onBlur={() => commitAdvanced("prior_fraud_count")} />
              <NumberField label="Chargebacks" value={form.advanced.chargeback_count} rule={numericRules.chargeback_count} onChange={(value) => updateAdvanced("chargeback_count", value)} onBlur={() => commitAdvanced("chargeback_count")} />
              <NumberField label="Merchant frequency" value={form.advanced.merchant_frequency} rule={numericRules.merchant_frequency} onChange={(value) => updateAdvanced("merchant_frequency", value)} onBlur={() => commitAdvanced("merchant_frequency")} />
              <NumberField label="Identity missing" value={form.advanced.identity_missing_rate} rule={numericRules.identity_missing_rate} step="0.01" onChange={(value) => updateAdvanced("identity_missing_rate", value)} onBlur={() => commitAdvanced("identity_missing_rate")} />
              <NumberField label="Identity signal" value={form.advanced.suspicious_identity_signal} rule={numericRules.suspicious_identity_signal} step="0.01" onChange={(value) => updateAdvanced("suspicious_identity_signal", value)} onBlur={() => commitAdvanced("suspicious_identity_signal")} />
            </div>
            <div className="toggle-grid compact">
              <Toggle label="Device present" checked={form.advanced.device_present} onChange={(value) => updateAdvanced("device_present", value)} />
              <Toggle label="Mobile device" checked={form.advanced.mobile_device} onChange={(value) => updateAdvanced("mobile_device", value)} />
              <Toggle label="Device mismatch" checked={form.advanced.card_device_mismatch} onChange={(value) => updateAdvanced("card_device_mismatch", value)} />
            </div>
          </div>
        </form>

        <section className={`panel result-panel ${riskClass}`}>
          <PanelHeading icon={TrendingUp} title="Decision" />
          {result ? (
            <div className="result-content">
              <div className="score-ring" style={{ "--score": `${clamp01(result.fraud_probability) * 100}%` }}>
                <div className="score-text">
                  <strong>{formatPercent(result.fraud_probability)}</strong>
                  <span>{result.decision_label}</span>
                </div>
              </div>
              <div className="decision-lines">
                <InfoLine label="Risk band" value={result.risk_band} />
                <InfoLine label="Threshold" value={formatNumber(result.metadata?.threshold, 4)} />
                <InfoLine label="Model" value={result.model_version} />
                <InfoLine label="USD amount" value={`$${formatMoney(result.metadata?.amount_usd)}`} />
                <InfoLine label="Exchange rate" value={`${result.metadata?.fx_source || "n/a"} @ ${formatNumber(result.metadata?.fx_rate, 5)}`} />
                <InfoLine label="Stale rate" value={result.metadata?.stale_fx_flag ? "Yes" : "No"} />
              </div>
            </div>
          ) : (
            <div className="result-placeholder">
              {apiState === "offline" ? <AlertTriangle size={30} /> : <Lock size={30} />}
              <span>{apiState === "offline" ? "API unavailable" : "No transaction scored"}</span>
              <small>{canScore ? "Ready for scoring" : status.detail}</small>
            </div>
          )}
        </section>
      </div>
    </main>
  );
}

function PanelHeading({ icon: Icon, title, action }) {
  return (
    <div className="panel-heading">
      <div>
        <Icon size={18} />
        <h2>{title}</h2>
      </div>
      {action && <div className="panel-action">{action}</div>}
    </div>
  );
}

function Field({ label, icon: Icon, children }) {
  return (
    <label className="field">
      <span>{Icon && <Icon size={14} />}{label}</span>
      {children}
    </label>
  );
}

function NumberField({ label, value, onChange, onBlur, rule, step = "1" }) {
  return (
    <Field label={label}>
      <input
        type="number"
        min={rule.min}
        max={rule.max}
        step={step}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        onBlur={onBlur}
      />
    </Field>
  );
}

function Segmented({ label, value, options, onChange }) {
  return (
    <div className="segmented-block">
      <span>{label}</span>
      <div className="segmented">
        {options.map(([optionValue, text]) => (
          <button
            type="button"
            key={optionValue}
            className={value === optionValue ? "active" : ""}
            onClick={() => onChange(optionValue)}
          >
            {text}
          </button>
        ))}
      </div>
    </div>
  );
}

function Toggle({ label, checked, onChange }) {
  return (
    <label className="toggle">
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
      <span className="switch" />
      <span>{label}</span>
    </label>
  );
}

function EmptyState({ title, detail }) {
  return (
    <div className="empty-state">
      <strong>{title}</strong>
      <span>{detail}</span>
    </div>
  );
}

function IconButton({ label, icon: Icon, onClick, loading }) {
  return (
    <button className="icon-button" aria-label={label} title={label} onClick={onClick}>
      {loading ? <Loader2 size={18} className="spin" /> : <Icon size={18} />}
    </button>
  );
}

function StatusPill({ icon: Icon, label, tone }) {
  return (
    <span className={`status-pill ${tone}`}>
      <Icon size={15} />
      {label}
    </span>
  );
}

function MetricTile({ icon: Icon, label, value }) {
  return (
    <div className="metric-tile">
      <Icon size={18} />
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function MetricInline({ label, value }) {
  return (
    <span>
      <small>{label}</small>
      <strong>{value == null ? "n/a" : formatNumber(value, 3)}</strong>
    </span>
  );
}

function InfoLine({ label, value }) {
  return (
    <div className="info-line">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function numericPayload(value) {
  const advanced = Object.fromEntries(
    Object.entries(value.advanced).map(([key, item]) => [
      key,
      typeof item === "boolean" ? item : Number(item)
    ])
  );
  return {
    ...value,
    amount: Number(value.amount),
    hour_of_day_local: Number(value.hour_of_day_local),
    day_of_week: Number(value.day_of_week),
    advanced,
    feature_overrides: {}
  };
}

function normalizeForm(value) {
  const messages = [];
  const form = cloneForm(value);
  for (const key of ["amount", "hour_of_day_local", "day_of_week"]) {
    const sanitized = sanitizeNumber(form[key], numericRules[key]);
    form[key] = sanitized.value;
    if (sanitized.changed) messages.push(humanLabel(key));
  }
  for (const key of Object.keys(form.advanced)) {
    const rule = numericRules[key];
    if (!rule) continue;
    const sanitized = sanitizeNumber(form.advanced[key], rule);
    form.advanced[key] = sanitized.value;
    if (sanitized.changed) messages.push(humanLabel(key));
  }
  if (Number(form.advanced.tx_count_24h) < Number(form.advanced.tx_count_1h)) {
    form.advanced.tx_count_24h = form.advanced.tx_count_1h;
    messages.push("24h transaction count");
  }
  return { form, messages: [...new Set(messages)].slice(0, 5) };
}

function sanitizeNumber(value, rule) {
  const raw = typeof value === "string" ? value.trim() : value;
  let numeric = Number(raw);
  if (!Number.isFinite(numeric)) {
    numeric = rule.fallback;
  }
  numeric = Math.max(rule.min, Math.min(rule.max, numeric));
  if (rule.integer) {
    numeric = Math.round(numeric);
  }
  const changed = Number(value) !== numeric || raw === "";
  return { value: numeric, changed };
}

function humanLabel(key) {
  return key
    .replace(/_/g, " ")
    .replace(/\btx\b/g, "transaction")
    .replace(/\b1h\b/g, "1h")
    .replace(/\b24h\b/g, "24h");
}

function cloneForm(value) {
  return {
    ...value,
    advanced: { ...value.advanced }
  };
}

function statusFor(apiState, loading, health) {
  if (loading && apiState === "checking") {
    return {
      icon: Loader2,
      tone: "neutral",
      label: "Checking API",
      detail: "Connecting to model server"
    };
  }
  if (apiState === "online" && health?.status === "ok") {
    return {
      icon: CheckCircle2,
      tone: "good",
      label: "API online",
      detail: "Model loaded"
    };
  }
  if (apiState === "no_model") {
    return {
      icon: AlertTriangle,
      tone: "warn",
      label: "No model",
      detail: "Load a checkpoint before scoring"
    };
  }
  return {
    icon: AlertTriangle,
    tone: "warn",
    label: "API offline",
    detail: "Start backend on port 8000"
  };
}

function apiErrorMessage(error) {
  if (error?.name === "AbortError") {
    return "API timeout";
  }
  const text = String(error?.message || error || "");
  if (text === "Failed to fetch" || text.includes("NetworkError")) {
    return "API unreachable";
  }
  return text || "API request failed";
}

function chartMetrics(model) {
  if (!model?.metrics) return [];
  return ["auprc", "auroc", "f1"].flatMap((key) => {
    const value = model.metrics[key];
    return value == null ? [] : [{ name: key.toUpperCase(), value }];
  });
}

function stripExt(name) {
  return (name || "").replace(/\.pt$/, "");
}

function formatNumber(value, digits = 3) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(digits) : "n/a";
}

function formatPercent(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? `${(numeric * 100).toFixed(1)}%` : "n/a";
}

function formatMoney(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : "n/a";
}

function titleCase(value) {
  return value.replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function clamp01(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  return Math.max(0, Math.min(1, numeric));
}

export default App;
