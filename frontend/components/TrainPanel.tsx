"use client";
import { useEffect, useRef, useState } from "react";
import {
  getDataset, getGrade, getTrainingStatus, saveCaptions, trainAsset,
  validateAdapter,
  type Asset, type DatasetInfo, type GradeInfo, type ValidationResult,
} from "@/lib/api";

export default function TrainPanel({ asset, onTrained }: { asset: Asset; onTrained: () => void }) {
  const count = asset.reference_images?.length || (asset.reference_image ? 1 : 0);
  const [open, setOpen] = useState(false);
  const [ds, setDs] = useState<DatasetInfo | null>(null);
  const [grade, setGrade] = useState<GradeInfo | null>(null);
  const [edits, setEdits] = useState<Record<string, string>>({});
  const [steps, setSteps] = useState(400);
  const [msg, setMsg] = useState("");
  const [busy, setBusy] = useState(false);
  const [validation, setValidation] = useState<ValidationResult | null>(null);

  async function load() {
    try {
      const [d, g] = await Promise.all([getDataset(asset.id), getGrade(asset.id)]);
      setDs(d); setGrade(g);
    } catch (err) {
      setMsg(err instanceof Error ? err.message : "Failed to load dataset");
    }
  }
  useEffect(() => { if (open) void load(); }, [open, asset.id]); // eslint-disable-line react-hooks/exhaustive-deps

  async function onSaveCaptions() {
    setBusy(true); setMsg("Saving captions…");
    try {
      const r = await saveCaptions(asset.id, edits);
      setMsg(`Saved ${r.updated} caption override${r.updated === 1 ? "" : "s"}. Training will use your words.`);
      setEdits({});
      await load();
    } catch (err) {
      setMsg(err instanceof Error ? err.message : "Save failed");
    } finally { setBusy(false); }
  }

  async function onTrain() {
    const okGo = window.confirm(
      `Start paid Modal GPU training (${steps} steps)?\n\nFree pre-flight checks above are done locally — this is the only step that spends credits.`);
    if (!okGo) return;
    setBusy(true); setMsg("Submitting…");
    try {
      const r = await trainAsset(asset.id, steps);
      setMsg(r.message || `Status: ${r.status}`);
      onTrained();
    } catch (err) {
      setMsg(err instanceof Error ? err.message : "Training failed");
    } finally { setBusy(false); }
  }

  async function onStatus(silent = false) {
    if (busy) return;
    if (!silent) { setBusy(true); setMsg("Checking… (a finished job also runs the 3-probe likeness check, ~1 min)"); }
    try {
      const r = await getTrainingStatus(asset.id);
      const v = r.validation && typeof r.validation === "object" && "passed" in r.validation ? (r.validation as ValidationResult) : null;
      setValidation(v);
      setMsg(v ? (v.passed ? "Adapter passed all likeness probes ✅" : "Adapter failed likeness probes — see reasons below.") : (r.message || `Status: ${r.status}`));
      onTrained();
    } catch (err) {
      if (!silent) setMsg(err instanceof Error ? err.message : "Status check failed");
    } finally { if (!silent) setBusy(false); }
  }

  const statusRef = useRef(onStatus);
  statusRef.current = onStatus;
  const busyRef = useRef(busy);
  busyRef.current = busy;
  useEffect(() => {
    if (!open || asset.train_status !== "training") return;
    let attempts = 0;
    const id = setInterval(() => {
      attempts += 1;
      if (attempts > 60) {
        clearInterval(id);
        setMsg("Still training after ~30 min — leave this open and hit Check status later.");
        return;
      }
      if (busyRef.current) return;
      void statusRef.current(true);
    }, 30000);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, asset.id, asset.train_status]);

  async function onValidate() {
    const okGo = window.confirm(
      "Re-run the likeness acceptance gate? This renders 3 test images through Modal (a few cents of GPU).");
    if (!okGo) return;
    setBusy(true); setMsg("Validating — rendering 3 probes…");
    try {
      const r = await validateAdapter(asset.id);
      const v = r.validation && typeof r.validation === "object" && "passed" in r.validation ? (r.validation as ValidationResult) : null;
      setValidation(v);
      setMsg(v ? (v.passed ? "Adapter passed all likeness probes ✅" : "Adapter failed likeness probes — see reasons below.") : `Status: ${r.status}`);
      onTrained();
    } catch (err) {
      setMsg(err instanceof Error ? err.message : "Validation failed");
    } finally { setBusy(false); }
  }

  return (
    <div className="mt-1">
      <button onClick={() => setOpen(!open)} title="Grade, captions and paid training"
        className="font-medium text-teal-800 underline">
        {open ? "Hide LoRA tools" : `Train LoRA on Modal (${count}/10 recommended)`}
      </button>
      {open && (
        <div className="mt-2 space-y-3 rounded-md border border-zinc-200 bg-zinc-50 p-3">
          {/* Grade */}
          <div>
            <h4 className="text-xs font-bold uppercase tracking-wide text-zinc-600">
              Dataset grade {grade ? (grade.ready ? "· ready ✅" : "· needs work ⚠️") : ""}
            </h4>
            {grade ? (
              <ul className="mt-1 space-y-1">
                {grade.checks.map((c, i) => (
                  <li key={i} className={`text-xs ${c.level === "error" ? "text-red-700" : c.level === "warn" ? "text-amber-700" : "text-teal-800"}`}>
                    {c.level === "ok" ? "✓" : c.level === "warn" ? "⚠" : "✕"} {c.message}
                  </li>
                ))}
              </ul>
            ) : <p className="mt-1 text-xs text-zinc-500">Loading grade…</p>}
            <a href={`http://localhost:8000/api/assets/${asset.id}/dataset.zip`} download
              className="mt-1 inline-block text-xs text-zinc-500 underline">Download dataset .zip</a>
          </div>

          {/* Captions */}
          <div>
            <h4 className="text-xs font-bold uppercase tracking-wide text-zinc-600">
              Captions — fix the words before you pay
            </h4>
            <p className="mt-0.5 text-xs text-zinc-500">
              Trigger token <code className="font-mono">{ds?.trigger || "…"}</code> is auto-appended.
              Override any caption below (e.g. correct the true shirt/overall colours).
            </p>
            {ds ? ds.captions.map((cap, i) => (
              <div key={ds.fingerprints[i] || i} className="mt-2">
                <p className="text-xs text-zinc-600">
                  <span className="font-mono text-[11px] text-zinc-400">#{i + 1} </span>
                  {cap} {ds.custom[i] && <span className="text-teal-700">(custom)</span>}
                </p>
                <input
                  value={edits[ds.fingerprints[i]] ?? ""}
                  onChange={(e) => setEdits({ ...edits, [ds.fingerprints[i]]: e.target.value })}
                  placeholder="Type a corrected caption, or leave blank to keep the auto one"
                  className="mt-1 min-h-9 w-full rounded-md border border-zinc-300 bg-white px-2 text-xs"
                />
              </div>
            )) : <p className="mt-1 text-xs text-zinc-500">Loading captions…</p>}
            {Object.keys(edits).some((k) => edits[k].trim()) && (
              <button onClick={onSaveCaptions} disabled={busy}
                className="mt-2 min-h-9 rounded-md bg-teal-800 px-3 text-xs font-semibold text-white disabled:opacity-40">
                Save caption overrides
              </button>
            )}
          </div>

          {/* Train */}
          <div className="flex flex-wrap items-center gap-2 border-t border-zinc-200 pt-2">
            <label className="text-xs font-medium text-zinc-600">
              Steps
              <select value={steps} onChange={(e) => setSteps(Number(e.target.value))}
                className="ml-1 min-h-9 rounded-md border border-zinc-300 bg-white px-2 text-xs">
                <option value={100}>100 · quick test (~$1)</option>
                <option value={200}>200 · balanced (~$2)</option>
                <option value={400}>400 · full (~$3–4)</option>
              </select>
            </label>
            <button onClick={onTrain} disabled={busy}
              className="min-h-9 rounded-md bg-zinc-900 px-3 text-xs font-semibold text-white disabled:opacity-40">
              Start paid training
            </button>
            {asset.train_status === "training" && (
              <button onClick={() => onStatus()} disabled={busy} className="text-xs font-medium text-teal-800 underline">
                Check status (free, auto-checks every 30s)
              </button>
            )}
            {(asset.train_status === "ready" || asset.train_status === "needs_review") && asset.adapter_ref && (
              <button onClick={onValidate} disabled={busy} title="Renders 3 test images through Modal (a few cents of GPU)"
                className="text-xs font-medium text-teal-800 underline disabled:opacity-40">
                Re-validate likeness
              </button>
            )}
          </div>
          {(msg || asset.train_status) && (
            <p className="text-xs text-zinc-600" role="status">
              LoRA: {asset.train_status || "none"}{msg ? ` — ${msg}` : ""}
              {asset.train_status === "ready" && asset.adapter_ref ? ` · ready ✅` : ""}
              {asset.train_status === "needs_review" ? ` · held for review ⚠️` : ""}
            </p>
          )}
          {validation && (
            <div className="rounded-md border border-zinc-200 bg-white p-2">
              <h4 className="text-xs font-bold uppercase tracking-wide text-zinc-600">
                Likeness gate {validation.passed ? "· passed ✅" : "· failed ⚠️"}
              </h4>
              <ul className="mt-1 space-y-1">
                {validation.probes.map((p) => (
                  <li key={p.name} className={`text-xs ${p.passed ? "text-teal-800" : "text-red-700"}`}>
                    {p.passed ? "✓" : "✕"} {p.name}
                    {!p.passed && p.reasons.length > 0 && `: ${p.reasons.join("; ")}`}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

