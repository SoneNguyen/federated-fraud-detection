import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BadgeDollarSign,
  CheckCircle2,
  ChevronDown,
  CreditCard,
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

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

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

const products = [
  ["W", "Retail"],
  ["H", "Home"],
  ["C", "Checkout"],
  ["S", "Service"],
  ["R", "Recurring"]
];

const dayLabels = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

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

  const selectedModel = useMemo(
    () => models.find((model) => stripExt(model.checkpoint) === selected) || models.find((model) => model.selected),
    [models, selected]
  );
  const recommended = useMemo(() => models.find((model) => model.recommended), [models]);

  useEffect(() => {
    refreshModels();
  }, []);

  async function api(path, options = {}) {
    const response = await fetch(`${API_BASE}${path}`, {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.detail || `${response.status} ${response.statusText}`);
    }
    return data;
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
      setSelected(modelData.selected || healthData.model_version || "not_loaded");
    } catch (error) {
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
      const payload = numericPayload(form);
      const prediction = await api("/predict-demo", {
        method: "POST",
        body: JSON.stringify(payload)
      });
      setResult(prediction);
    } catch (error) {
      setMessage({ type: "error", text: error.message });
    } finally {
      setLoadingPrediction(false);
    }
  }

  function updateField(key, value) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  function updateAdvanced(key, value) {
    setForm((current) => ({
      ...current,
      advanced: { ...current.advanced, [key]: value }
    }));
  }

  const metricBars = chartMetrics(selectedModel);
  const riskClass = result ? `risk-${result.risk_band}` : "risk-idle";

  return (
    <main className="shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">
            <Landmark size={22} />
          </div>
          <div>
            <p className="eyebrow">Vault Fraud Console</p>
            <h1>Transaction screening</h1>
          </div>
        </div>
        <div className="top-actions">
          <StatusPill
            icon={health?.status === "ok" ? CheckCircle2 : AlertTriangle}
            tone={health?.status === "ok" ? "good" : "warn"}
            label={health?.status === "ok" ? "API online" : "API offline"}
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
                  Use recommended
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
              <div className="empty-state">No metrics loaded</div>
            )}
          </div>

          <div className="model-list">
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
              <button type="submit" className="primary-command" disabled={loadingPrediction}>
                {loadingPrediction ? <Loader2 size={17} className="spin" /> : <Play size={17} />}
                Score
              </button>
            }
          />

          <div className="form-grid">
            <Field label="Amount">
              <input
                type="number"
                min="0"
                step="0.01"
                value={form.amount}
                onChange={(event) => updateField("amount", event.target.value)}
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
                min="0"
                max="23"
                value={form.hour_of_day_local}
                onChange={(event) => updateField("hour_of_day_local", event.target.value)}
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
          <Field label="Card brand">
            <select value={form.card_brand} onChange={(event) => updateField("card_brand", event.target.value)}>
              {["visa", "mastercard", "american express", "discover", "other"].map((brand) => (
                <option key={brand} value={brand}>{titleCase(brand)}</option>
              ))}
            </select>
          </Field>

          <div className="toggle-grid">
            <Toggle label="Email match" checked={form.email_domain_match} onChange={(value) => updateField("email_domain_match", value)} />
            <Toggle label="Payer free email" checked={form.payer_free_email} onChange={(value) => updateField("payer_free_email", value)} />
            <Toggle label="Receiver free email" checked={form.receiver_free_email} onChange={(value) => updateField("receiver_free_email", value)} />
            <Toggle label="Live FX" checked={form.use_live_fx} onChange={(value) => updateField("use_live_fx", value)} />
          </div>

          <button type="button" className="advanced-toggle" onClick={() => setAdvancedOpen((open) => !open)}>
            <SlidersHorizontal size={17} />
            Advanced signals
            <ChevronDown size={17} className={advancedOpen ? "open" : ""} />
          </button>

          <div className={`advanced-region ${advancedOpen ? "open" : ""}`}>
            <div className="advanced-grid">
              <NumberField label="Tx count 1h" value={form.advanced.tx_count_1h} min="0" max="500" onChange={(value) => updateAdvanced("tx_count_1h", value)} />
              <NumberField label="Tx count 24h" value={form.advanced.tx_count_24h} min="0" max="5000" onChange={(value) => updateAdvanced("tx_count_24h", value)} />
              <NumberField label="Geo velocity" value={form.advanced.geo_velocity_kmh} min="0" max="2000" onChange={(value) => updateAdvanced("geo_velocity_kmh", value)} />
              <NumberField label="Distance km" value={form.advanced.distance_km} min="0" max="20000" onChange={(value) => updateAdvanced("distance_km", value)} />
              <NumberField label="Days since tx" value={form.advanced.days_since_last_tx} min="0" max="365" onChange={(value) => updateAdvanced("days_since_last_tx", value)} />
              <NumberField label="Account age" value={form.advanced.account_age_days} min="0" max="10000" onChange={(value) => updateAdvanced("account_age_days", value)} />
              <NumberField label="History count" value={form.advanced.history_count} min="0" max="1000000" onChange={(value) => updateAdvanced("history_count", value)} />
              <NumberField label="History fraud rate" value={form.advanced.history_fraud_rate} min="0" max="1" step="0.01" onChange={(value) => updateAdvanced("history_fraud_rate", value)} />
              <NumberField label="Prior fraud count" value={form.advanced.prior_fraud_count} min="0" max="10000" onChange={(value) => updateAdvanced("prior_fraud_count", value)} />
              <NumberField label="Chargebacks" value={form.advanced.chargeback_count} min="0" max="100" onChange={(value) => updateAdvanced("chargeback_count", value)} />
              <NumberField label="Merchant frequency" value={form.advanced.merchant_frequency} min="0" max="10000000" onChange={(value) => updateAdvanced("merchant_frequency", value)} />
              <NumberField label="Identity missing" value={form.advanced.identity_missing_rate} min="0" max="1" step="0.01" onChange={(value) => updateAdvanced("identity_missing_rate", value)} />
              <NumberField label="Identity signal" value={form.advanced.suspicious_identity_signal} min="0" max="1" step="0.01" onChange={(value) => updateAdvanced("suspicious_identity_signal", value)} />
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
                <InfoLine label="FX" value={`${result.metadata?.fx_source || "n/a"} @ ${formatNumber(result.metadata?.fx_rate, 5)}`} />
                <InfoLine label="FX stale" value={result.metadata?.stale_fx_flag ? "Yes" : "No"} />
              </div>
            </div>
          ) : (
            <div className="result-placeholder">
              <Lock size={30} />
              <span>No transaction scored</span>
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
      {action}
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

function NumberField({ label, value, onChange, min, max, step = "1" }) {
  return (
    <Field label={label}>
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(event.target.value)}
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
